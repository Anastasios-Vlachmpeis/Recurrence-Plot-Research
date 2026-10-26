"""Surrogate series for null models."""
import numpy as np


def iaaft(x, rng: np.random.Generator, n_iter: int = 200) -> np.ndarray:
    """IAAFT surrogate (Schreiber & Schmitz 1996): exactly x's values, approximately its amplitude spectrum.

    Stops early once the rank order no longer changes.
    """
    x = np.asarray(x, dtype=float)
    amp = np.abs(np.fft.rfft(x))
    values = np.sort(x)
    s = rng.permutation(x)
    ranks = None
    for _ in range(n_iter):
        s = np.fft.irfft(amp * np.exp(1j * np.angle(np.fft.rfft(s))), len(x))
        new = np.argsort(np.argsort(s))
        s = values[new]
        if ranks is not None and np.array_equal(new, ranks):
            break
        ranks = new
    return s
