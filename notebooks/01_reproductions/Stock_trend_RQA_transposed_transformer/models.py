"""Neural and statistical model definitions for the stock-trend reconstruction."""

import math
import warnings

import numpy as np
import torch
from scipy.stats import norm
from statsmodels.tsa.arima.model import ARIMA
from torch import nn
from torch.nn import functional as F

from .config import FEATURES


def _window_size(config):
    return config.WINDOW_DAYS * config.BARS_PER_DAY


def classifier_head(n_features):
    return nn.Sequential(nn.Linear(n_features, 20), nn.GELU(), nn.Dropout(.3), nn.Linear(20, 2))


class RQABranch(nn.Module):
    def __init__(self, config, n_features=13):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_features, 128), nn.GELU(), nn.Dropout(.2), nn.Linear(128, 32))

    def forward(self, r):
        return self.net(r)


class TransposedTransformerBranch(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Linear(_window_size(config), 256, bias=False)
        self.variable_bias = nn.Parameter(torch.zeros(len(FEATURES), 256))
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(256, 8, 512, .3, activation='gelu',
                                       batch_first=True, norm_first=False)
            for _ in range(2)
        ])
        self.reduce = nn.Linear(len(FEATURES) * 256, 64)

    def forward(self, x):
        x = self.embedding(x.transpose(1, 2)) + self.variable_bias
        for layer in self.layers:
            x = layer(x)
        return self.reduce(x.flatten(1))


class DualBranchClassifier(nn.Module):
    def __init__(self, config, rqa_input='close'):
        super().__init__()
        self.rqa_input = rqa_input
        self.rqa = RQABranch(config, 65 if rqa_input == 'all' else 13)
        self.transformer = TransposedTransformerBranch(config)
        self.head = classifier_head(96)

    def forward(self, x, r):
        return self.head(torch.cat([self.rqa(r), self.transformer(x)], dim=1))


class RQAOnlyClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.rqa = RQABranch(config)
        self.head = classifier_head(32)

    def forward(self, x, r):
        return self.head(self.rqa(r))


class TransformerOnlyClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transformer = TransposedTransformerBranch(config)
        self.head = classifier_head(64)

    def forward(self, x, r):
        return self.head(self.transformer(x))


class LSTMClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.lstm = nn.LSTM(len(FEATURES), 128, 2, batch_first=True, dropout=.3)
        self.head = classifier_head(128)

    def forward(self, x, r):
        sequence, _ = self.lstm(x)
        return self.head(sequence[:, -1])


class DLinearClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        window_size = _window_size(config)
        self.seasonal = nn.Linear(window_size, window_size)
        self.trend = nn.Linear(window_size, window_size)
        nn.init.constant_(self.seasonal.weight, 1 / window_size)
        nn.init.constant_(self.trend.weight, 1 / window_size)
        self.head = nn.Linear(window_size * len(FEATURES), 2)
        self.window_size = window_size

    def forward(self, x, r):
        xc = x.transpose(1, 2)
        half = (25 - 1) // 2
        trend = F.avg_pool1d(F.pad(xc, (half, half), mode='replicate'), 25, stride=1)
        out = self.seasonal(xc - trend) + self.trend(trend)
        return self.head(out.transpose(1, 2).reshape(len(x), -1))


class Projector(nn.Module):
    def __init__(self, config, output_dim):
        super().__init__()
        self.series_conv = nn.Conv1d(_window_size(config), 1, 3, padding=1,
                                     padding_mode='circular', bias=False)
        self.backbone = nn.Sequential(
            nn.Linear(len(FEATURES) * 2, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, output_dim, bias=False)
        )

    def forward(self, x, stats):
        return self.backbone(torch.cat([self.series_conv(x), stats], dim=1).flatten(1))


class DeStationaryAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.q = nn.Linear(256, 256)
        self.k = nn.Linear(256, 256)
        self.v = nn.Linear(256, 256)
        self.projection = nn.Linear(256, 256)
        self.dropout = nn.Dropout(.3)

    def forward(self, x, tau, delta):
        batch, length, _ = x.shape
        q = self.q(x).view(batch, length, 8, 32).transpose(1, 2)
        k = self.k(x).view(batch, length, 8, 32).transpose(1, 2)
        v = self.v(x).view(batch, length, 8, 32).transpose(1, 2)
        scores = (torch.matmul(q, k.transpose(-2, -1)) * tau[:, None, None, :]
                  + delta[:, None, None, :]) / math.sqrt(32)
        attention = self.dropout(torch.softmax(scores, dim=-1))
        return self.projection(torch.matmul(attention, v).transpose(1, 2).reshape(batch, length, 256))


class DeStationaryLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = DeStationaryAttention(config)
        self.conv1 = nn.Conv1d(256, 512, 1)
        self.conv2 = nn.Conv1d(512, 256, 1)
        self.norm1 = nn.LayerNorm(256)
        self.norm2 = nn.LayerNorm(256)
        self.drop = nn.Dropout(.3)

    def forward(self, x, tau, delta):
        x = self.norm1(x + self.drop(self.attention(x, tau, delta)))
        y = self.drop(F.gelu(self.conv1(x.transpose(1, 2))))
        y = self.drop(self.conv2(y).transpose(1, 2))
        return self.norm2(x + y)


class NonstationaryTransformerClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        window_size = _window_size(config)
        self.token = nn.Conv1d(len(FEATURES), 256, 3, padding=1,
                               padding_mode='circular', bias=False)
        nn.init.kaiming_normal_(self.token.weight, mode='fan_in', nonlinearity='leaky_relu')
        pos = torch.arange(window_size, dtype=torch.float32).unsqueeze(1)
        rate = torch.exp(torch.arange(0, 256, 2) * (-math.log(10000) / 256))
        pe = torch.zeros(1, window_size, 256)
        pe[0, :, 0::2] = torch.sin(pos * rate)
        pe[0, :, 1::2] = torch.cos(pos * rate)
        self.register_buffer('position', pe)
        self.tau = Projector(config, 1)
        self.delta = Projector(config, window_size)
        self.layers = nn.ModuleList([DeStationaryLayer(config) for _ in range(2)])
        self.norm = nn.LayerNorm(256)
        self.drop = nn.Dropout(.3)
        self.head = nn.Linear(window_size * 256, 2)

    def forward(self, x, r):
        raw = x.detach()
        mean = raw.mean(1, keepdim=True)
        std = torch.sqrt((raw - mean).var(1, keepdim=True, unbiased=False) + 1e-5)
        tau = self.tau(raw, std).clamp(max=80).exp()
        delta = self.delta(raw, mean)
        out = self.drop(self.token(x.transpose(1, 2)).transpose(1, 2) + self.position)
        for layer in self.layers:
            out = layer(out, tau, delta)
        return self.head(self.drop(F.gelu(self.norm(out))).flatten(1))


BASELINES = ['ARIMA', 'RF', 'XGBoost', 'LSTM', 'Non-stationary Transformer', 'DLinear']
ARMS = ['Dual-close', 'Dual-open', 'Dual-high', 'Dual-low', 'Dual-vol', 'Dual-all',
        'RQA-only', 'Transformer-only']
NEURAL = ['LSTM', 'Non-stationary Transformer', 'DLinear', *ARMS]


def make_model(name, config):
    if name.startswith('Dual-'):
        return DualBranchClassifier(config, name.split('-', 1)[1])
    return {
        'RQA-only': RQAOnlyClassifier,
        'Transformer-only': TransformerOnlyClassifier,
        'LSTM': LSTMClassifier,
        'DLinear': DLinearClassifier,
        'Non-stationary Transformer': NonstationaryTransformerClassifier,
    }[name](config)


def arm_features(rqa, name, config):
    if name == 'Dual-all':
        return rqa.reshape(len(rqa), 65)
    series = name.split('-', 1)[1] if name.startswith('Dual-') else 'close'
    return rqa[:, FEATURES.index(series), :]


def arima_probabilities(windows, config):
    probabilities = []
    for x in windows:
        close = x[:, FEATURES.index('close')].astype(float)
        fitted = ARIMA(close, order=(5, 1, 0), trend='n').fit()
        if not fitted.mle_retvals.get('converged', True):
            warnings.warn('ARIMA optimizer did not converge; inspect this reconstructed baseline.', RuntimeWarning)
        forecast = fitted.get_forecast(steps=240)
        mean = float(np.asarray(forecast.predicted_mean)[-1])
        se = float(np.asarray(forecast.se_mean)[-1])
        if not np.isfinite([mean, se]).all() or se < 0:
            raise RuntimeError('Invalid ARIMA forecast distribution')
        probabilities.append(float(norm.sf((close[-1] - mean) / se)) if se > 0
                             else float(mean >= close[-1]))
    return np.asarray(probabilities)
