"""Time-delay embedding and data-driven choice of its delay (AMI) and dimension (false nearest neighbours)."""
import numpy as np


def embed(x, m: int, tau: int) -> np.ndarray:
    """Delay-embed a series into states of dimension m with delay tau.

    A 1-D input of length N gives shape (N - (m - 1) * tau, m). A 2-D input of shape (N, d)
    gives (N - (m - 1) * tau, m * d), lags outermost. State t is dated by its latest sample,
    index t + (m - 1) * tau of the input.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    n = len(x) - (m - 1) * tau
    return np.concatenate([x[k * tau : k * tau + n] for k in range(m)], axis=1)


def average_mutual_information(x, max_lag: int, bins: int = 16) -> np.ndarray:
    """Mutual information (nats) between x_t and x_{t+lag} for lag = 0..max_lag, on equiprobable bins."""
    x = np.asarray(x, dtype=float)
    b = np.argsort(np.argsort(x, kind="stable")) * bins // len(x)
    out = []
    for lag in range(max_lag + 1):
        p = np.histogram2d(b[: len(b) - lag], b[lag:], bins=bins, range=[[0, bins], [0, bins]])[0]
        p /= p.sum()
        indep = np.outer(p.sum(axis=1), p.sum(axis=0))
        nz = p > 0
        out.append(float((p[nz] * np.log(p[nz] / indep[nz])).sum()))
    return np.array(out)


def first_local_minimum(values) -> int | None:
    """Smallest index k >= 1 with values[k-1] > values[k] <= values[k+1]; None if there is none."""
    for k in range(1, len(values) - 1):
        if values[k - 1] > values[k] <= values[k + 1]:
            return k
    return None


def false_nearest_fraction(x, tau: int, max_m: int, rtol: float = 15.0, atol: float = 2.0,
                           theiler: int = 1) -> np.ndarray:
    """Share of false nearest neighbours (Kennel et al. 1992) for m = 1..max_m at delay tau.

    The nearest neighbour of each m-dimensional state (excluding states closer than `theiler` in time) is
    false if adding coordinate m + 1 moves it more than rtol times their distance, or the (m + 1)-dimensional
    distance exceeds atol times the std of x.
    """
    x = np.asarray(x, dtype=float)
    scale = x.std()
    out = []
    for m in range(1, max_m + 1):
        S = embed(x, m + 1, tau)
        A, extra = S[:, :m], S[:, m]
        D = np.sqrt(((A[:, None, :] - A[None, :, :]) ** 2).sum(axis=-1))
        idx = np.arange(len(D))
        D[np.abs(idx[:, None] - idx[None, :]) < theiler] = np.inf
        nn = D.argmin(axis=1)
        r = D[idx, nn]
        step = np.abs(extra - extra[nn])
        false = (step > rtol * np.maximum(r, 1e-12)) | (np.sqrt(r ** 2 + step ** 2) > atol * scale)
        out.append(float(false.mean()))
    return np.array(out)
