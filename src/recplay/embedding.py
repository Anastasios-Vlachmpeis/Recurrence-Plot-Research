"""Time-delay embedding."""
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
