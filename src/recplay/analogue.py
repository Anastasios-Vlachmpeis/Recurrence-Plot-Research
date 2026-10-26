"""Analogue (method-of-analogues) features: the current state's recurrences with the whole past, and what followed them."""
import numpy as np
import pandas as pd

from recplay.embedding import embed
from recplay.recurrence import euclidean_distances


def analogue_frame(x: pd.Series, target: pd.Series, m: int, tau: int, rr: float, horizons,
                   min_candidates: int = 20) -> pd.DataFrame:
    """For every month t, the past months s whose delay-embedded state of x is recurrent with the state at t.

    Candidates are s <= t - h, so the outcome target[s+h] - target[s] is known at t. Recurrent means within ε_t,
    the rr-quantile of the candidates' Euclidean distances to the state at t (one column of the RP of the whole
    history, thresholded at rate rr). Columns per horizon h: analog_up_h{h}, the share of recurrent months
    followed by a strict rise in target over h months, and analog_chg_h{h}, their mean change. NaN while there
    are fewer than min_candidates candidates.
    """
    if not x.index.equals(target.index):
        raise ValueError("x and target need the same index")
    off = (m - 1) * tau  # state k is dated by month k + off
    D = euclidean_distances(embed(x.to_numpy(dtype=float), m, tau))
    y = target.to_numpy(dtype=float)
    n = len(x)
    out = {}
    for h in horizons:
        change = np.full(n, np.nan)
        change[: n - h] = y[h:] - y[: n - h]
        up, chg = np.full(n, np.nan), np.full(n, np.nan)
        for t in range(off, n):
            cand = np.arange(off, t - h + 1)
            if len(cand) < min_candidates:
                continue
            d = D[t - off, cand - off]
            c = change[cand[d <= np.quantile(d, rr)]]
            up[t], chg[t] = (c > 0).mean(), c.mean()
        out[f"analog_up_h{h}"], out[f"analog_chg_h{h}"] = up, chg
    return pd.DataFrame(out, index=x.index)
