"""Critical recurrence rate RRc: the RR at which recurrences first link the start of the record to its end.

Domain Ω = {(i, j): j - i >= theiler}. A = the first k rows of Ω, B = the last k columns, k = max(1, ceil(δN)).
ε_c is the smallest ε at which one 8-connected component of {(i, j) ∈ Ω: D_ij <= ε} meets both A and B;
RRc = |{(i, j) ∈ Ω: D_ij <= ε_c}| / |Ω|, so ties at ε_c count as recurrent.
"""
from math import ceil

import numpy as np
from scipy.stats import rankdata


def rr_critical(D: np.ndarray, theiler: int = 1, delta: float = 0.05) -> float:
    """RRc of a distance-like matrix (only its upper triangle at k = theiler is read).

    Pixels are activated in order of increasing D and merged with active 8-neighbours by union-find
    (Kruskal / 0-dimensional persistence); each root carries two flags, touches A and touches B.
    """
    n = len(D)
    I, J = np.triu_indices(n, theiler)
    if len(I) == 0:
        raise ValueError("empty domain: theiler >= N")
    vals = D[I, J]
    order = np.argsort(vals, kind="stable")
    sorted_vals = vals[order]
    k = max(1, ceil(delta * n))
    w = n + 2  # padded grid, so neighbours never need bounds checks
    parent = list(range(w * w))
    active = bytearray(w * w)
    flags = bytearray(w * w)  # bit 1: touches A, bit 2: touches B
    offsets = (-w - 1, -w, -w + 1, -1, 1, w - 1, w, w + 1)
    for rank, (i, j) in enumerate(zip(I[order].tolist(), J[order].tolist())):
        p = (i + 1) * w + j + 1
        active[p] = 1
        flags[p] = (i < k) | ((j >= n - k) << 1)
        for o in offsets:
            q = p + o
            if active[q]:
                while parent[q] != q:
                    parent[q] = parent[parent[q]]
                    q = parent[q]
                if q != p:
                    parent[q] = p
                    flags[p] |= flags[q]
        if flags[p] == 3:
            return float(np.searchsorted(sorted_vals, sorted_vals[rank], side="right") / len(vals))
    raise AssertionError("Ω is 8-connected, so the last pixel always spans")


def omega_ranks(D: np.ndarray, theiler: int = 1) -> np.ndarray:
    """Rank of each pair's D within Ω (ties share the highest rank); 0 outside Ω.

    Pair (i, j) is in the RP thresholded at rate q iff its rank <= q|Ω|, so the elementwise max of two such
    matrices orders pairs by when they enter the joint RP; rr_critical of it is the joint RRc.
    """
    n = len(D)
    I, J = np.triu_indices(n, theiler)
    out = np.zeros(D.shape)
    out[I, J] = rankdata(D[I, J], method="max")
    return out
