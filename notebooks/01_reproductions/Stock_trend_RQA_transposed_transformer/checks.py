"""Behavioral verification for the reconstruction, not market-performance evidence."""
import copy, gc
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from .models import make_model, arm_features, DeStationaryAttention
from .training import metrics, seed_everything
from .rqa import RQA_ATTRS


class ReproductionChecks:
    def __init__(self, config, data, rqa, trainer):
        self.config=config
        self.data=data
        self.rqa=rqa
        self.trainer=trainer

    def correctness_checks(self, backend):
        self.rqa.check_fixture('cpu')
        self.rqa.check_fixture(backend)
        f, c, d = self.data.synthetic_minutes()
        a = self.data.audit_and_build_samples(f, c, d, 'SYNTHETIC')
        assert a['x'].shape == (83, 1200, 5)
        assert (a['meta'].input_end.dt.tz_localize(None).dt.normalize() < a['meta'].date).all()
        assert a['meta'].loc[a['meta'].target_close == a['meta'].previous_close, 'y'].eq(1).all()
        i = 7
        target = a['meta'].date.iloc[i]
        mutated = f.copy()
        mask = pd.to_datetime(mutated.trade_time).dt.normalize().eq(target)
        mutated.loc[mask, 'close'] = a['meta'].previous_close.iloc[i] + 3
        mutated.loc[mask, 'high'] = mutated.loc[mask, ['open', 'close']].max(axis=1) + 0.1
        mutated.loc[mask, 'low'] = mutated.loc[mask, ['open', 'close']].min(axis=1) - 0.1
        b = self.data.audit_and_build_samples(mutated, c, d, 'SYNTHETIC')
        np.testing.assert_array_equal(a['x'][i], b['x'][i])
        assert a['meta'].y.iloc[i] != b['meta'].y.iloc[i]
        unchanged_input = b['x'][i, :, 1].copy()
        auction = f.iloc[[0]].copy()
        auction['trade_time'] = pd.to_datetime(auction.trade_time) - pd.Timedelta(minutes=1)
        with_auction = self.data.audit_and_build_samples(pd.concat([f, auction]), c, d, 'SYNTHETIC')
        np.testing.assert_array_equal(with_auction['x'], a['x'])
        bad = f.iloc[[0]].copy()
        bad['vol'] += 1
        try:
            self.data.audit_and_build_samples(pd.concat([f, bad]), c, d, 'SYNTHETIC')
        except ValueError:
            pass
        else:
            raise AssertionError('Conflicting duplicate was accepted')
        missing = f.loc[pd.to_datetime(f.trade_time) != self.data.expected_slots(c.cal_date.iloc[8])[216].tz_localize(None)]
        b = self.data.audit_and_build_samples(missing, c, d, 'SYNTHETIC')
        assert len(b['meta']) == len(a['meta']) - 6
        assert a['audit'].status.eq('valid').all()
        sample = a['x'][i, :, self.config.FEATURES.index('close')]
        actual = self.rqa.features(sample, backend)
        direct, _ = self.rqa.result(sample, 'cpu')
        np.testing.assert_allclose(actual, [getattr(direct, k) for k in RQA_ATTRS], rtol=2e-05, atol=1e-05, equal_nan=True)
        np.testing.assert_allclose(actual, self.rqa.features(unchanged_input, backend), equal_nan=True)
        score = metrics([0, 1, 0, 1], [0.1, 0.8, 0.4, 0.9])
        assert all((score[k] == 1 for k in ['accuracy', 'f1', 'auc']))
        assert np.isnan(metrics([1, 1], [0.1, 0.9])['auc'])
        assert metrics([0, 1], [0.5, 0.5])['accuracy'] == 0.5
        seed_everything(0)
        x = torch.randn(2, 1200, 5, device=self.config.DEVICE)
        r = torch.randn(2, 13, device=self.config.DEVICE)
        model = make_model('Dual-close', self.config).to(self.config.DEVICE)
        seen = []
        h = model.transformer.layers[0].self_attn.register_forward_pre_hook(lambda m, args: seen.append(args[0].shape[1]))
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-06)
        before = model.rqa.net[0].weight.detach().clone()
        out = model(x, r)
        assert out.shape == (2, 2)
        F.cross_entropy(out, torch.tensor([0, 1], device=self.config.DEVICE)).backward()
        for branch in [model.rqa, model.transformer]:
            assert sum((p.grad.abs().sum().item() for p in branch.parameters() if p.grad is not None)) > 0
        optimizer.step()
        assert not torch.equal(before, model.rqa.net[0].weight)
        assert seen == [5]
        h.remove()
        attention = DeStationaryAttention(self.config).to(self.config.DEVICE).eval()
        z = torch.randn(2, 7, 256, device=self.config.DEVICE)
        out1 = attention(z, torch.ones(2, 1, device=self.config.DEVICE), torch.zeros(2, 7, device=self.config.DEVICE))
        out2 = attention(z, torch.full((2, 1), 2.0, device=self.config.DEVICE), torch.arange(7, device=self.config.DEVICE).expand(2, -1).float())
        assert not torch.allclose(out1, out2)
        del model, optimizer, attention, out, out1, out2, x, r, z
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print('Correctness checks passed: RQA, chronology, missing sessions, labels, metrics, both branches and attention.')

    def check_train_isolation(self, data, rqa, split):
        """Mutation check: held-out observations cannot enter preprocessing or epoch selection."""
        sx, sr, pre = self.trainer.prepare(data, rqa, split, True)
        other = copy.deepcopy(data)
        mask = other['meta'].date.dt.year.eq(2022).to_numpy()
        other['x'][mask] *= 3
        other['meta'].loc[mask, 'y'] = 1 - other['meta'].loc[mask, 'y']
        other['minutes'].loc[other['minutes'].index.year == 2022] *= 3
        orqa = rqa.copy()
        orqa[mask] *= 3
        ox, orr, opre = self.trainer.prepare(other, orqa, split, True)
        assert pre.state() == opre.state()
        np.testing.assert_array_equal(sx[split['fit']], ox[split['fit']])
        np.testing.assert_array_equal(sr[split['val']], orr[split['val']])
        y = data['meta'].y.to_numpy()
        a = self.trainer.train_neural('Dual-close', 0, sx, arm_features(sr, 'Dual-close', self.config), y, split['fit'], split['val'], 2, True)
        b = self.trainer.train_neural('Dual-close', 0, ox, arm_features(orr, 'Dual-close', self.config), other['meta'].y.to_numpy(), split['fit'], split['val'], 2, True)
        assert a['epochs'] == b['epochs']
        np.testing.assert_allclose(a['probability'], b['probability'])
        print('Held-out mutation did not change fit state, training features or selected epochs.')
