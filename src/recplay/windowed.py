"""Sliding-window RQA and RRc features. The row for month t uses only the W months ending at t.

`series` maps a column prefix to a column of `data` (e.g. {"cpi": "INFLATION"}); `joints` maps a prefix to
two series prefixes whose RPs are AND-ed (e.g. {"jun": ("cpi", "un")}). `distance(z, m, tau)` turns a
z-scored window into a pairwise distance or divergence matrix of its embedded states.
"""
import numpy as np
import pandas as pd

from recplay.connectivity import omega_ranks, rr_critical
from recplay.embedding import embed
from recplay.recurrence import euclidean_distances, symmetric_beta_matrix, threshold_by_rate
from recplay.rqa import rqa
from recplay.surrogates import iaaft


def window_z(x: np.ndarray, end: int, W: int) -> np.ndarray:
    """x[end-W+1 : end+1] z-scored with its own mean and std (zeros if constant)."""
    w = x[end - W + 1 : end + 1]
    sd = w.std()
    return (w - w.mean()) / sd if sd > 0 else np.zeros_like(w)


def euclidean_distance(z: np.ndarray, m: int, tau: int) -> np.ndarray:
    return euclidean_distances(embed(z, m, tau))


def beta_distance(z: np.ndarray, m: int, tau: int, beta: float) -> np.ndarray:
    """Symmetric β-divergence between embedded exp(z) states (exp makes them strictly positive)."""
    return symmetric_beta_matrix(embed(np.exp(z), m, tau), beta)


def rqa_frame(data: pd.DataFrame, series: dict, joints: dict, W: int, m: int, tau: int, rr: float,
              distance, theiler: int = 1) -> pd.DataFrame:
    """rqa() of every single-series RP (thresholded at rate rr) and every joint RP, columns '<prefix>_<measure>'.

    Also RR_last: the recurrence rate of the window's last state (the current month) with the rest of the window.
    """
    arrays = {p: data[c].to_numpy(dtype=float) for p, c in series.items()}
    rows = []
    for end in range(W - 1, len(data)):
        R = {p: threshold_by_rate(distance(window_z(x, end, W), m, tau), rr, theiler) for p, x in arrays.items()}
        R.update({j: R[a] & R[b] for j, (a, b) in joints.items()})
        rows.append({f"{p}_{k}": v for p, Rp in R.items()
                     for k, v in {**rqa(Rp), "RR_last": Rp[-1].sum() / (len(Rp) - theiler)}.items()})
    return pd.DataFrame(rows, index=data.index[W - 1 :])


def window_surrogates(data: pd.DataFrame, series: dict, W: int, n_surr: int) -> dict:
    """n_surr IAAFT surrogates of every z-scored window: {"index": window end dates, prefix: (windows, n_surr, W)}.

    Each window's generator is seeded by (W, end date, column name), so a window gets the same surrogates
    whatever data surrounds it.
    """
    ends = data.index[W - 1 :]
    out = {"index": ends}
    for p, c in series.items():
        x = data[c].to_numpy(dtype=float)
        windows = []
        for end, date in zip(range(W - 1, len(data)), ends):
            z = window_z(x, end, W)
            rng = np.random.default_rng([W, date.toordinal(), *c.encode()])
            windows.append(np.stack([iaaft(z, rng) for _ in range(n_surr)]))
        out[p] = np.stack(windows)
    return out


def critical_rates(D: dict, joints: dict, theiler: int, delta: float) -> dict:
    """RRc of every single-series distance matrix and of every joint RP (ordered by the max of Ω-ranks)."""
    out = {p: rr_critical(d, theiler, delta) for p, d in D.items()}
    for j, (a, b) in joints.items():
        out[j] = rr_critical(np.maximum(omega_ranks(D[a], theiler), omega_ranks(D[b], theiler)), theiler, delta)
    return out


def rrc_frame(data: pd.DataFrame, series: dict, joints: dict, W: int, m: int, tau: int, distance,
              surrogates: dict, theiler: int = 1, delta: float = 0.05) -> pd.DataFrame:
    """Per RP: RRc, its mean over the surrogates (RRc_sur), dRRc = RRc - RRc_sur, and RRc_pct = share of
    surrogates with RRc <= the observed one. Joint surrogates pair the s-th surrogate of each series."""
    arrays = {p: data[c].to_numpy(dtype=float) for p, c in series.items()}
    n_surr = surrogates[next(iter(series))].shape[1]
    rows = []
    for end in range(W - 1, len(data)):
        pos = surrogates["index"].get_loc(data.index[end])
        obs = critical_rates({p: distance(window_z(x, end, W), m, tau) for p, x in arrays.items()},
                             joints, theiler, delta)
        sur = [critical_rates({p: distance(surrogates[p][pos, s], m, tau) for p in series}, joints, theiler, delta)
               for s in range(n_surr)]
        row = {}
        for p, value in obs.items():
            null = np.array([s[p] for s in sur])
            row |= {f"{p}_RRc": value, f"{p}_RRc_sur": null.mean(), f"{p}_dRRc": value - null.mean(),
                    f"{p}_RRc_pct": (null <= value).mean()}
        rows.append(row)
    return pd.DataFrame(rows, index=data.index[W - 1 :])
