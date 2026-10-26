"""Distance/divergence matrices and thresholded recurrence plots."""
import numpy as np
import pandas as pd

from recplay.embedding import embed


def euclidean_distances(states: np.ndarray) -> np.ndarray:
    """Pairwise Euclidean distances between the rows of a 2-D state array."""
    return np.sqrt(((states[:, None, :] - states[None, :, :]) ** 2).sum(axis=-1))


def recurrence(series: pd.Series, m: int, tau: int, rr: float):
    """Euclidean RP of a z-scored, delay-embedded series.

    ε is the rr-quantile of the off-diagonal distances, so the recurrence rate is about rr
    (ties at ε can push it above). Returns (D, R, eps, dates); each state is dated by its latest sample.
    """
    x = series.to_numpy(dtype=float)
    D = euclidean_distances(embed((x - x.mean()) / x.std(), m, tau))
    eps = np.quantile(D[np.triu_indices(len(D), k=1)], rr)
    return D, D <= eps, eps, series.index[(m - 1) * tau :]


def threshold_by_rate(d: np.ndarray, rr: float, theiler: int = 1) -> np.ndarray:
    """RP from a distance or divergence matrix, with the main diagonal and the Theiler band set to False.

    ε is chosen so the recurrence rate over pairs with |i - j| >= theiler equals rr.
    """
    n = len(d)
    band = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :]) < theiler
    eps = np.quantile(d[np.triu_indices(n, theiler)], rr)
    R = d <= eps
    R[band] = False
    return R


def recurrence_matrix(states: np.ndarray, rr: float, theiler: int = 1) -> np.ndarray:
    """Euclidean RP of given states with the main diagonal and the Theiler band set to False.

    ε is chosen so the recurrence rate over pairs with |i - j| >= theiler equals rr.
    """
    return threshold_by_rate(euclidean_distances(states), rr, theiler)


def joint_matrix(state_list, rr: float, theiler: int = 1) -> np.ndarray:
    """Joint RP: the AND of each system's own recurrence_matrix, each thresholded separately."""
    R = recurrence_matrix(state_list[0], rr, theiler)
    for states in state_list[1:]:
        R = R & recurrence_matrix(states, rr, theiler)
    return R


def symmetric_beta_matrix(states: np.ndarray, beta: float) -> np.ndarray:
    """Coordinate-summed mean of forward and reverse β-divergences (Févotte & Idier, §2.1).

    S_β(x, y) = (x - y)(x^(β-1) - y^(β-1)) / (2(β - 1)) for any real β ≠ 1, and (x - y)(ln x - ln y) / 2 at β = 1.
    """
    states = np.asarray(states, dtype=float)
    if states.ndim != 2 or not np.isfinite(states).all() or np.any(states <= 0):
        raise ValueError("β-divergence states must be a finite, strictly positive 2D array")
    if not np.isfinite(beta):
        raise ValueError("β must be finite")
    transformed = np.log(states) if beta == 1 else states ** (beta - 1) / (beta - 1)
    divergence = 0.5 * (
        (states[:, None, :] - states[None, :, :])
        * (transformed[:, None, :] - transformed[None, :, :])
    ).sum(axis=-1)
    if not np.isfinite(divergence).all():
        raise ValueError("β-divergence overflow; check the input scale")
    return divergence


def beta_recurrence(series: pd.Series, beta: float, m: int, tau: int, rr: float):
    """β-divergence RP of a percentage-valued series encoded as exp(v / 100) > 0.

    For log returns in percent this is the gross return. Returns (R, eps, dates).
    """
    positive = np.exp(series.to_numpy(dtype=float) / 100)
    if m < 1 or tau < 1 or len(positive) - (m - 1) * tau < 2:
        raise ValueError("Embedding requires positive m, tau and at least two states")
    states = embed(positive, m, tau)
    divergence = symmetric_beta_matrix(states, beta)
    eps = float(np.quantile(divergence[np.triu_indices(len(states), k=1)], rr))
    return divergence <= eps, eps, series.index[(m - 1) * tau :]
