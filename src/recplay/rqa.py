"""Recurrence quantification: line-length distributions, DET, L, ENTR, LAM, K2."""
import numpy as np


def _runs(seqs) -> np.ndarray:
    out = []
    for s in seqs:
        s = np.concatenate([[0], s.astype(np.int8), [0]])
        dd = np.diff(s)
        out.append(np.flatnonzero(dd == -1) - np.flatnonzero(dd == 1))
    return np.concatenate(out) if out else np.array([], int)


def line_lengths(R: np.ndarray):
    """Diagonal line lengths (upper triangle; R is symmetric) and vertical line lengths."""
    n = len(R)
    diag = _runs(np.diagonal(R, k) for k in range(1, n))
    vert = _runs(R[:, j] for j in range(n))
    return diag, vert


def k2_lines(diag: np.ndarray, n: int, ls=np.arange(1, 7), min_count: int = 10) -> float:
    """Per-step decay rate of C_l, the share of pairs that stay recurrent for l consecutive steps.

    For IID data C_l = RR**l, so the rate is -ln(RR); for chaos it estimates K2 (Thiel et al. 2003).
    """
    counts = np.array([np.clip(diag - l + 1, 0, None).sum() for l in ls], float)
    pairs = np.array([sum(max(n - k - l + 1, 0) for k in range(1, n)) for l in ls], float)
    ok = counts >= min_count
    if ok.sum() < 3:
        return np.nan
    return -np.polyfit(ls[ok], np.log(counts[ok] / pairs[ok]), 1)[0]


def rqa(R: np.ndarray) -> dict:
    """RR, DET, L, ENTR, LAM, DET(lmin=3)/DET(lmin=2) and K2_lines of a recurrence matrix."""
    diag, vert = line_lengths(R)
    pts = diag.sum()
    det = lambda lmin: diag[diag >= lmin].sum() / pts if pts else np.nan
    long = diag[diag >= 2]
    p = np.bincount(long)[2:]
    p = p[p > 0] / p.sum()
    return {"RR": R.mean(), "DET": det(2), "L": long.mean() if len(long) else 0.0,
            "ENTR": float(-(p * np.log(p)).sum()) if len(p) else 0.0,
            "LAM": vert[vert >= 2].sum() / max(vert.sum(), 1),
            "DET3_DET2": det(3) / max(det(2), 1e-12), "K2_lines": k2_lines(diag, len(R))}
