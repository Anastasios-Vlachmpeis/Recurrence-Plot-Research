"""Joint-recurrence dependence between systems: independence ratios, joint covariance and correlation.

Every function takes `rps`, a dict of equally sized boolean RPs keyed by series name, and counts only
pairs (i, j) with j - i >= min_sep (min_sep > 1 is a Theiler window).
"""
from itertools import combinations

import numpy as np
import pandas as pd


def combos(cols) -> list[tuple[str, ...]]:
    """All pairs, then all triples, of the given series names."""
    return [combo for k in (2, 3) for combo in combinations(cols, k)]


def independence_table(rps: dict, cols, min_sep: int = 1) -> pd.DataFrame:
    """Joint RR vs the product of individual RRs (its value if the systems recurred independently)."""
    n = len(rps[cols[0]])
    idx = np.triu_indices(n, k=min_sep)
    rr = {c: rps[c][idx].mean() for c in cols}
    rows = []
    for combo in combos(cols):
        joint_rr = np.logical_and.reduce([rps[c] for c in combo])[idx].mean()
        expected_rr = np.prod([rr[c] for c in combo])
        rows.append({"systems": " × ".join(combo),
                     **{f"RR {c}": rr[c] for c in combo},
                     "joint RR": joint_rr,
                     "expected if independent": expected_rr,
                     "ratio": joint_rr / expected_rr})
    table = pd.DataFrame(rows).set_index("systems")
    return table[[f"RR {c}" for c in cols] + ["joint RR", "expected if independent", "ratio"]]


def dependence_table(rps: dict, cols, min_sep: int = 1) -> pd.DataFrame:
    """Joint covariance and correlation of the 0/1 recurrence indicators.

    Pairs: Cov_XY = q_XY - q_X q_Y, divided by σ_X σ_Y (phi coefficient).
    Triples: the third-order joint cumulant E[(R_X - q_X)(R_Y - q_Y)(R_Z - q_Z)], which is 0 whenever one
    system is independent of the other two; divided by the product of L3 norms, bounded in [-1, 1] by Hölder.
    """
    n = len(rps[cols[0]])
    idx = np.triu_indices(n, k=min_sep)
    centred = {c: rps[c][idx] - rps[c][idx].mean() for c in cols}
    rows = []
    for combo in combos(cols):
        parts = [centred[c] for c in combo]
        cov = np.prod(parts, axis=0).mean()
        if len(combo) == 2:
            scale = np.prod([np.sqrt(np.mean(p ** 2)) for p in parts])
        else:
            scale = np.prod([np.mean(np.abs(p) ** 3) ** (1 / 3) for p in parts])
        rows.append({"systems": " × ".join(combo), "order": len(combo),
                     "joint covariance": cov, "joint correlation": cov / scale})
    return pd.DataFrame(rows).set_index("systems")
