"""PyRQA feature extraction and fit-only preprocessing for the reconstruction."""

import time

import numpy as np
from scipy.spatial.distance import pdist
from pyrqa.analysis_type import Classic
from pyrqa.computation import RQAComputation
from pyrqa.metric import EuclideanMetric
from pyrqa.neighbourhood import FixedRadius
from pyrqa.settings import Settings
from pyrqa.time_series import TimeSeries
from pyrqa.variants.rqa.radius.baseline import Baseline

from recplay.embedding import embed
from .config import fingerprint, sha_bytes


RQA_NAMES = ['RR', 'DET', 'L', 'Lmax', 'DIV', 'Lentr', 'LAM', 'TT', 'Vmax', 'Ventr', 'W', 'Wmax', 'Wentr']
RQA_ATTRS = [
    'recurrence_rate', 'determinism', 'average_diagonal_line', 'longest_diagonal_line',
    'divergence', 'entropy_diagonal_lines', 'laminarity', 'trapping_time', 'longest_vertical_line',
    'entropy_vertical_lines', 'average_white_vertical_line', 'longest_white_vertical_line',
    'entropy_white_vertical_lines',
]


class RQAExtractor:
    def __init__(self, config):
        self.config = config
        self.opencl = None

    def choose_backend(self):
        import pyopencl as cl
        from pyrqa.opencl import OpenCL

        try:
            platforms = cl.get_platforms()
        except cl.LogicError:
            platforms = []
        options = []
        for platform_index, platform in enumerate(platforms):
            for device_index, device in enumerate(platform.get_devices()):
                options.append((not bool(device.type & cl.device_type.GPU), platform_index,
                                device_index, device.name))
        if not options:
            print('No OpenCL device: using the genuine PyRQA CPU Baseline (slower).')
            return 'cpu'
        _, platform_index, device_index, name = min(options)
        self.opencl = OpenCL(platform_id=platform_index, device_ids=(device_index,))
        print('PyRQA OpenCL device:', name)
        return 'opencl'

    def result(self, series, backend, *, m=None, tau=None, epsilon=None):
        m = self.config.M if m is None else m
        tau = self.config.TAU if tau is None else tau
        values = np.asarray(series, dtype=np.float32)
        if values.ndim != 1 or len(values) - (m - 1) * tau < 2 or not np.isfinite(values).all():
            raise ValueError('RQA requires a finite 1D series with at least two delay states')
        eps = (float(self.config.EPS_FRACTION * pdist(embed(values, m, tau)).max())
               if epsilon is None else float(epsilon))
        settings = Settings(
            TimeSeries(values, embedding_dimension=m, time_delay=tau, dtype=np.float32),
            analysis_type=Classic,
            neighbourhood=FixedRadius(eps),
            similarity_measure=EuclideanMetric,
            theiler_corrector=1,
        )
        if backend == 'cpu':
            result = Baseline(settings, verbose=False).run()
        elif backend == 'opencl':
            if self.opencl is None:
                raise RuntimeError('OpenCL must be initialized before RQA')
            result = RQAComputation.create(settings, verbose=False, opencl=self.opencl).run()
        else:
            raise ValueError('Unknown RQA backend')
        result.min_diagonal_line_length = 2
        result.min_vertical_line_length = 2
        result.min_white_vertical_line_length = 2
        return result, eps

    def features(self, series, backend, *, m=None, tau=None, epsilon=None):
        result, _ = self.result(series, backend, m=m, tau=tau, epsilon=epsilon)
        values = np.array([getattr(result, name) for name in RQA_ATTRS], dtype=np.float32)
        values[~np.isfinite(values)] = np.nan
        return values

    def check_fixture(self, backend):
        fixture = [.1, .5, 1.3, .7, .8, 1.4, 1.6, 1.2, .4, 1.1, .8, .2, 1.3]
        expected = np.array([
            .371901, .411765, 2.333333, 3, .333333, .636514, .4, 2.571429, 4, .955700,
            2.538462, 6, .839796,
        ])
        observed = self.features(fixture, backend, m=2, tau=2, epsilon=.65)
        np.testing.assert_allclose(observed, expected, atol=1e-5, rtol=1e-5)
        print('PyRQA published reference fixture passed:', backend)

    def extract_all(self, data, backend, cache_dir):
        x = data['x']
        dates = data['meta'].date.dt.strftime('%Y-%m-%d').tolist()
        out = np.empty((len(x), 5, 13), np.float32)
        epsilons = np.empty((len(x), 5), float)
        spec = dict(
            m=self.config.M, tau=self.config.TAU, epsilon_fraction=self.config.EPS_FRACTION,
            theiler=1, min_lines=2, backend=backend, pyrqa=self.config.VERSIONS['PyRQA'],
            implementation=self.config.CONFIG['implementation'],
            source_sha256=self.config.CONFIG['source_sha256'],
        )
        started = time.monotonic()
        for i, window in enumerate(x):
            key = fingerprint(dict(spec=spec, date=dates[i], input_hash=sha_bytes(window.tobytes())))
            path = cache_dir / f'{key}.npz'
            if path.exists():
                with np.load(path, allow_pickle=False) as cached:
                    features = cached['features']
                    eps = cached['epsilons']
                    if (features.shape != (5, 13) or eps.shape != (5,)
                            or str(cached['hash']) != sha_bytes(features.tobytes() + eps.tobytes())):
                        raise RuntimeError(f'Invalid feature cache: {path}')
                out[i], epsilons[i] = features, eps
            else:
                for j in range(5):
                    result, eps = self.result(window[:, j], backend)
                    values = np.array([getattr(result, name) for name in RQA_ATTRS], np.float32)
                    values[~np.isfinite(values)] = np.nan
                    out[i, j] = values
                    epsilons[i, j] = eps
                cache_dir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    path, features=out[i], epsilons=epsilons[i],
                    hash=sha_bytes(out[i].tobytes() + epsilons[i].tobytes()),
                )
            if i == 0:
                duration = time.monotonic() - started
                print(f'First five-series window: {duration:.2f}s; estimated total '
                      f'{duration * len(x) / 60:.1f} min (cache hits and startup affect this estimate).')
            if (i + 1) % 50 == 0:
                print('RQA windows:', i + 1, '/', len(x))
        return out, epsilons


class TrainPreprocessor:
    def __init__(self, config):
        self.config = config

    def fit(self, data, indices, rqa, last_year):
        rows = data['minutes'].loc[data['minutes'].index.year <= last_year,
                                   self.config.FEATURES].to_numpy(float)
        if len(rows) == 0:
            raise ValueError('No unique fit-year minute rows')
        self.mean = rows.mean(0)
        self.scale = rows.std(0)
        self.scale[self.scale == 0] = 1.
        r = rqa[indices].reshape(len(indices), -1).astype(float)
        self.undefined = (~np.isfinite(r)).sum(0)
        self.median = np.array([
            np.median(column[np.isfinite(column)]) if np.isfinite(column).any() else 0.
            for column in r.T
        ])
        clean = np.where(np.isfinite(r), r, self.median)
        self.rmean = clean.mean(0)
        self.rscale = clean.std(0)
        self.rscale[self.rscale == 0] = 1.
        print('Fit-only undefined RQA entries:', int(self.undefined.sum()))
        return self

    def transform(self, x, rqa):
        x = ((x - self.mean) / self.scale).astype(np.float32)
        shape = rqa.shape
        r = rqa.reshape(len(rqa), -1).astype(float)
        r = np.where(np.isfinite(r), r, self.median)
        r = ((r - self.rmean) / self.rscale).astype(np.float32).reshape(shape)
        return x, r

    def state(self):
        return {key: getattr(self, key).tolist()
                for key in ['mean', 'scale', 'median', 'rmean', 'rscale', 'undefined']}
