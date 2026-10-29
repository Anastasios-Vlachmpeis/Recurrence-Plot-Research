"""Paper-comparison summaries, artifacts and diagnostic figures."""
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from IPython.display import display
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import roc_curve

from recplay.embedding import embed
from .config import fingerprint, json_save
from .models import BASELINES
from .references import PAPER_REFERENCE
from .rqa import RQA_NAMES


class Reporter:
    def __init__(self, config, data_service):
        self.config = config
        self.data = data_service

    def summarize_results(self, raw):
        per_stock = raw.groupby(['stock', 'model'])[['accuracy', 'f1', 'auc']].agg(['mean', 'std', 'count'])
        expected = 1 if self.config.SMOKE else 7
        records = []
        macro = []
        for (stock, model), row in per_stock.iterrows():
            for measure in ['accuracy', 'f1', 'auc']:
                records.append(dict(stock=stock, model=model, metric=measure,
                                    reconstruction=row[measure]['mean'],
                                    reconstruction_seed_sd=row[measure]['std'],
                                    reconstruction_stock_sd=np.nan,
                                    n_seeds=int(row[measure]['count'])))
        detail = pd.DataFrame(records)
        for (model, measure), group in detail.groupby(['model', 'metric']):
            complete = len(group) == expected and np.isfinite(group.reconstruction).all()
            mean = group.reconstruction.mean() if complete else np.nan
            sd = group.reconstruction.std(ddof=1) if complete else np.nan
            macro.append(dict(stock='Average', model=model, metric=measure, reconstruction=mean,
                              reconstruction_seed_sd=group.reconstruction_seed_sd.mean(),
                              reconstruction_stock_sd=sd, n_stocks=len(group), complete=complete))
        macro = pd.DataFrame(macro)
        observed = pd.concat([detail, macro], ignore_index=True)
        comparison = PAPER_REFERENCE.merge(observed, on=['stock', 'model', 'metric'], how='left')
        comparison['difference'] = comparison.reconstruction - comparison.published
        if self.config.SMOKE:
            # Never imply that a synthetic stock was compared empirically with the paper's seven stocks.
            comparison['reconstruction'] = np.nan
            comparison['difference'] = np.nan
            comparison['status'] = 'No historical estimates: smoke mechanics only'
        return per_stock, macro, comparison

    def write_reports(self, raw, predictions, coverage, target_coverage, runs, histories,
                      datasets, rqas, backend):
        key = fingerprint(dict(config=self.config.CONFIG, runs={stock: path.name for stock, path in runs.items()}))
        destination = self.config.RESULTS / key
        destination.mkdir(parents=True, exist_ok=True)
        prefix = 'smoke_' if self.config.SMOKE else ''
        per_stock, macro, comparison = self.summarize_results(raw)
        exported = {'predictions': predictions, 'raw_metrics': raw, 'paper_reference': PAPER_REFERENCE,
                    'comparison_table_v': comparison, 'coverage': coverage,
                    'target_coverage': target_coverage,
                    'per_stock_seed_summary': per_stock.reset_index(), 'macro_summary': macro}
        branch = macro[macro.model.isin(['Dual-close', 'RQA-only', 'Transformer-only'])].copy()
        branch['published_accuracy'] = branch.model.map(
            {'Dual-close': .6256, 'RQA-only': .5955, 'Transformer-only': .6075})
        branch.loc[branch.metric != 'accuracy', 'published_accuracy'] = np.nan
        variable = macro[macro.model.str.startswith('Dual-')].copy()
        reference = dict(zip(['Dual-close', 'Dual-open', 'Dual-high', 'Dual-low', 'Dual-vol', 'Dual-all'],
                             [.6256, .6157, .6085, .5984, .5897, .5906]))
        reference_sd = dict(zip(reference, [.0225, .0226, .0225, .0245, .0235, .0216]))
        variable['published_accuracy'] = variable.model.map(reference)
        variable['published_reported_sd'] = variable.model.map(reference_sd)
        variable.loc[variable.metric != 'accuracy', ['published_accuracy', 'published_reported_sd']] = np.nan
        exported['branch_ablations'] = branch
        exported['table_vi'] = variable
        for name, table in exported.items():
            table.to_csv(destination / f'{prefix}{name}.csv', index=False)
        manifest = dict(label=self.config.LABEL, mode=self.config.MODE, config=self.config.CONFIG,
                        versions=self.config.VERSIONS, backend=backend, device=self.config.DEVICE,
                        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                        generated_utc=datetime.now(timezone.utc).isoformat(),
                        stock_run_directories={stock: str(path.relative_to(self.config.ROOT))
                                               for stock, path in runs.items()},
                        complete_seven_stock_result=not self.config.SMOKE and len(runs) == 7,
                        complete_auc=bool(macro.loc[macro.metric == 'auc', 'complete'].all()),
                        uncertainty=(f'Sample SD across {len(self.config.SEEDS)} seed(s); separate SD across '
                                     'stock means. Paper ± unspecified.'),
                        study='10.1063/5.0233275',
                        data_source='synthetic_fixture' if self.config.SMOKE else 'Tushare stk_mins, unadjusted')
        json_save(destination / f'{prefix}manifest.json', manifest)
        print(self.config.LABEL, '\nOutputs:', destination)
        print('Per-stock seed means / sample SDs (internal accuracy fraction):')
        display(per_stock.round(4))
        print('Macro mean across stock means (not pooled predictions):')
        display(macro.round(4))
        print('Published Table V versus historical reconstruction:')
        display(comparison.query("stock=='Average'").round(4))
        print('Branch removal; F1/AUC are additional diagnostics, not digitized Figure 4 values:')
        display(branch.round(4))
        print('Table VI RQA inputs:')
        display(variable.query("metric=='accuracy'").round(4))

        # Only diagnostic figures; numeric comparison tables are not duplicated as bar charts.
        stock = next(iter(datasets))
        data = datasets[stock]
        sample = self.data.split_samples(data, self.config.SMOKE)['fit'][0]
        series = data['x'][sample, :, self.config.FEATURES.index('close')]
        distances = squareform(pdist(embed(series, self.config.M, self.config.TAU)))
        epsilon = self.config.EPS_FRACTION * distances.max()
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(distances < epsilon, cmap='binary', origin='lower', interpolation='nearest')
        ax.set(title=f'{self.config.LABEL}\n{stock}: example close RP (training window)',
               xlabel='Delay state', ylabel='Delay state')
        fig.tight_layout()
        fig.savefig(destination / f'{prefix}recurrence.png', dpi=110)
        plt.show()
        print('Training-window RQA features:')
        display(pd.Series(rqas[stock][sample, 1], index=RQA_NAMES).to_frame('value'))

        fig, ax = plt.subplots(figsize=(6, 4))
        for model in ['Dual-close', 'Non-stationary Transformer']:
            history = histories[stock][model]
            ax.plot([entry['epoch'] for entry in history],
                    [entry['validation_loss'] for entry in history], label=f'{model} validation')
        ax.set(title=f'{self.config.LABEL}\nSeed-0 epoch selection', xlabel='Epoch',
               ylabel='2021 validation cross-entropy')
        ax.legend()
        fig.tight_layout()
        fig.savefig(destination / f'{prefix}loss.png', dpi=110)
        plt.show()

        fig, ax = plt.subplots(figsize=(6, 4))
        present = predictions[predictions.stock == stock]
        for model in [*BASELINES, 'Dual-close']:
            # Additional ROC diagnostic is a seed-mean ensemble, not the reported seed-mean metric.
            group = present[present.model == model].groupby('date').agg(
                y=('y', 'first'), probability=('probability', 'mean'))
            if group.y.nunique() == 2:
                fpr, tpr, _ = roc_curve(group.y, group.probability)
                ax.plot(fpr, tpr, label=model)
        ax.plot([0, 1], [0, 1], '--', color='gray')
        ax.set(title=f'{self.config.LABEL}\n{stock}: seed-mean probability ROC diagnostic',
               xlabel='False positive rate', ylabel='True positive rate')
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(destination / f'{prefix}roc.png', dpi=110)
        plt.show()
        if self.config.SMOKE:
            print('Mechanics completed. These synthetic scores say NOTHING about stock predictability or replication.')
        else:
            pivot = macro.pivot(index='model', columns='metric', values='reconstruction')
            for model in [*BASELINES, 'RQA-only', 'Transformer-only']:
                print('Dual-close minus', model, ':',
                      (pivot.loc['Dual-close'] - pivot.loc[model]).round(4).to_dict())
            print('No statistical significance or profitability claim follows from these differences.')
        return destination, raw, macro, comparison
