"""Neural training and per-stock experiment orchestration."""
import gc
import json
import random

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from .config import fingerprint, json_save, sha_bytes
from .models import ARMS, BASELINES, NEURAL, arm_features, arima_probabilities, make_model
from .rqa import TrainPreprocessor


def seed_everything(seed):
    """Seed Python, NumPy and PyTorch RNGs for one experiment run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def metrics(y, p):
    """Calculate classification metrics from labels and positive-class probabilities."""
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    y = np.asarray(y, int)
    p = np.asarray(p, float)
    if len(y) == 0 or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('Metrics require nonempty targets and finite [0,1] probabilities')
    pred = (p > .5).astype(int)
    return dict(accuracy=accuracy_score(y, pred), f1=f1_score(y, pred, zero_division=0),
                auc=roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan,
                auc_reason='' if len(np.unique(y)) == 2 else 'Single-class test targets', n=len(y))


class Trainer:
    def __init__(self, config):
        self.config = config

    def seed_everything(self, seed):
        seed_everything(seed)

    def _train_attempt(self, name, seed, x, r, y, fit_idx, eval_idx, epochs, select, device, physical):
        self.seed_everything(seed)
        model = make_model(name, self.config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.CONFIG['lr'],
                                     weight_decay=self.config.CONFIG['weight_decay'])
        rng = np.random.default_rng(seed)
        history = []
        best = float('inf')
        best_epoch = 1
        stale = 0
        for epoch in range(1, epochs + 1):
            model.train()
            order = rng.permutation(fit_idx)
            total = 0.
            for start in range(0, len(order), self.config.CONFIG['batch_size']):
                effective = order[start:start + self.config.CONFIG['batch_size']]
                optimizer.zero_grad(set_to_none=True)
                for offset in range(0, len(effective), physical):
                    idx = effective[offset:offset + physical]
                    xb = torch.as_tensor(x[idx], device=device)
                    rb = torch.as_tensor(r[idx], device=device)
                    yb = torch.as_tensor(y[idx], dtype=torch.long, device=device)
                    logits = model(xb, rb)
                    loss = F.cross_entropy(logits, yb, reduction='sum') / len(effective)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f'Nonfinite training loss in {name}')
                    loss.backward()
                    total += loss.item() * len(effective)
                optimizer.step()
            train_loss = total / len(order)
            if not select and epoch < epochs:
                history.append(dict(epoch=epoch, train_loss=train_loss, validation_loss=None))
                continue
            model.eval()
            probability = []
            eval_loss = 0.
            with torch.no_grad():
                for start in range(0, len(eval_idx), physical):
                    idx = eval_idx[start:start + physical]
                    logits = model(torch.as_tensor(x[idx], device=device),
                                   torch.as_tensor(r[idx], device=device))
                    if select:
                        loss = F.cross_entropy(
                            logits, torch.as_tensor(y[idx], dtype=torch.long, device=device), reduction='sum')
                        eval_loss += float(loss.item())
                    probability.extend(logits.softmax(1)[:, 1].cpu().numpy())
            eval_loss /= len(eval_idx)
            if not np.isfinite(eval_loss):
                raise FloatingPointError(f'Nonfinite evaluation loss in {name}')
            history.append(dict(epoch=epoch, train_loss=train_loss,
                                validation_loss=eval_loss if select else None))
            if select:
                if eval_loss < best - 1e-6:
                    best = eval_loss
                    best_epoch = epoch
                    stale = 0
                else:
                    stale += 1
                if stale >= self.config.PATIENCE:
                    break
        # Final-refit target-period losses are not exposed for checkpoint selection.
        return dict(state={k: v.detach().cpu() for k, v in model.state_dict().items()},
                    probability=np.asarray(probability), history=history,
                    epochs=best_epoch if select else epochs, device=device,
                    physical_batch=physical, effective_batch=self.config.CONFIG['batch_size'])

    def train_neural(self, name, seed, x, r, y, fit_idx, eval_idx, epochs, select=False):
        options = ([('cuda', 16), ('cuda', 2), ('cpu', 2)]
                   if self.config.DEVICE == 'cuda' else [('cpu', 16)])
        for device, physical in options:
            try:
                return self._train_attempt(name, seed, x, r, y, fit_idx, eval_idx,
                                           epochs, select, device, physical)
            except torch.cuda.OutOfMemoryError:
                print(f'{name}: CUDA OOM at physical batch {physical}; restarting the complete fit.')
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        raise RuntimeError('All training attempts failed')

    def prepare(self, data, rqa, split, selection):
        indices = split['fit'] if selection else split['final']
        last_year = 2020 if selection else 2021
        preprocessor = TrainPreprocessor(self.config).fit(data, indices, rqa, last_year)
        x, r = preprocessor.transform(data['x'], rqa)
        return x, r, preprocessor

    @staticmethod
    def safe_manifest(value):
        if isinstance(value, dict):
            return {k: Trainer.safe_manifest(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [Trainer.safe_manifest(v) for v in value]
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            return None
        if isinstance(value, np.generic):
            return value.item()
        return value

    def experiment_stock(self, stock, data, rqa, split, backend):
        signature = dict(config=self.config.CONFIG, versions=self.config.VERSIONS, backend=backend,
                         x_hash=sha_bytes(data['x'].tobytes()), rqa_hash=sha_bytes(rqa.tobytes()),
                         targets=data['meta'][['date', 'input_start', 'input_end', 'y']]
                         .astype(str).to_dict('list'),
                         unique_minutes_hash=sha_bytes(
                             pd.util.hash_pandas_object(data['minutes'], index=True).values.tobytes()))
        key = fingerprint(signature)
        run_dir = self.config.PROCESSED / 'runs' / stock / key
        run_dir.mkdir(parents=True, exist_ok=True)
        json_save(run_dir / 'manifest.json', self.safe_manifest(signature))
        selection_x, selection_r, selection_pre = self.prepare(data, rqa, split, True)
        final_x, final_r, final_pre = self.prepare(data, rqa, split, False)
        json_save(run_dir / 'preprocessing.json',
                  dict(selection=selection_pre.state(), final=final_pre.state()))
        y = data['meta'].y.to_numpy(int)
        test = split['test']
        rows = []
        predictions = []
        histories = {}
        for name in [*BASELINES, *ARMS]:
            stem = name.replace(' ', '_')
            selected_file = run_dir / f'{stem}_selection.json'
            if name in NEURAL:
                if selected_file.exists():
                    selection = json.loads(selected_file.read_text())
                    epochs = int(selection['epochs'])
                else:
                    selection = self.train_neural(
                        name, 0, selection_x, arm_features(selection_r, name, self.config), y,
                        split['fit'], split['val'], self.config.MAX_EPOCHS, True)
                    epochs = selection['epochs']
                    selection = {k: v for k, v in selection.items() if k not in {'state', 'probability'}}
                    json_save(selected_file, selection)
                histories[name] = selection['history']
            else:
                epochs = None
            seeds = [0] if name == 'ARIMA' else self.config.SEEDS
            for seed in seeds:
                path = run_dir / f'{stem}_seed{seed}.npz'
                info_path = run_dir / f'{stem}_seed{seed}.json'
                if path.exists() or info_path.exists():
                    if not (path.exists() and info_path.exists()):
                        raise RuntimeError(f'Incomplete experiment cache: {path}')
                    info = json.loads(info_path.read_text())
                    if info['sha256'] != sha_bytes(path.read_bytes()):
                        raise RuntimeError(f'Invalid experiment hash: {path}')
                    with np.load(path, allow_pickle=False) as cached:
                        probability = cached['probability']
                        if not np.array_equal(cached['indices'], test):
                            raise RuntimeError('Cached prediction dates mismatch')
                else:
                    if name == 'ARIMA':
                        probability = arima_probabilities(data['x'][test], self.config)
                        info = dict(epochs=None, device='cpu', runs=1)
                    elif name in {'RF', 'XGBoost'}:
                        estimator = (RandomForestClassifier(
                            n_estimators=300, max_depth=6, min_samples_leaf=5, max_features='sqrt',
                            n_jobs=-1, random_state=seed) if name == 'RF' else
                            XGBClassifier(n_estimators=300, max_depth=3, learning_rate=.05,
                                          subsample=.8, colsample_bytree=.8, objective='binary:logistic',
                                          eval_metric='logloss', tree_method='hist', random_state=seed,
                                          n_jobs=4))
                        flattened_fit = final_x[split['final']].reshape(len(split['final']), -1)
                        estimator.fit(flattened_fit, y[split['final']])
                        flattened_test = final_x[test].reshape(len(test), -1)
                        probability = estimator.predict_proba(flattened_test)[:, 1]
                        info = dict(epochs=None, device='cpu')
                    else:
                        fitted = self.train_neural(
                            name, seed, final_x, arm_features(final_r, name, self.config), y,
                            split['final'], test, epochs)
                        probability = fitted.pop('probability')
                        state = fitted.pop('state')
                        torch.save(state, run_dir / f'{stem}_seed{seed}.pt')
                        info = fitted
                    # Validate BEFORE recording successful completion.
                    metrics(y[test], probability)
                    np.savez_compressed(path, probability=probability, indices=test)
                    info['sha256'] = sha_bytes(path.read_bytes())
                    json_save(info_path, info)
                row = dict(stock=stock, model=name, seed=seed, **metrics(y[test], probability))
                rows.append(row)
                for idx, p in zip(test, probability):
                    predictions.append(dict(stock=stock, date=data['meta'].date.iloc[idx], model=name,
                                            seed=seed, y=int(y[idx]), probability=float(p)))
                print(self.config.LABEL, stock, name, 'seed', seed,
                      {k: round(row[k], 4) for k in ['accuracy', 'f1', 'auc']})
        # Sanity references, not replacements for any paper baseline.
        prior = float(y[split['final']].mean())
        for name, probability in [('Always up', np.ones(len(test))),
                                  ('Training majority', np.full(len(test), float(prior >= .5)))]:
            rows.append(dict(stock=stock, model=name, seed=0, **metrics(y[test], probability)))
        return pd.DataFrame(rows), pd.DataFrame(predictions), histories, run_dir
