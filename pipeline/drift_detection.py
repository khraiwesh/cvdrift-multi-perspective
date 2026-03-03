# ============================================================
# pipeline/drift_detection.py — PELT change-point detection
# ============================================================
"""Detect and consolidate change points using the PELT algorithm (ruptures)."""

import warnings
from typing import List, Optional

import numpy as np

try:
    import ruptures as rpt
except ImportError as e:
    raise ImportError("Install ruptures: pip install ruptures") from e


def _default_pen_value(n: int, scale: float = 3.0) -> float:
    """Compute the default PELT penalty value: scale * ln(n)."""
    n = max(2, int(n))
    return float(scale * np.log(n))


def detect_drift_pelt(
    series_values: np.ndarray,
    pen_scale: float = 3.0,
    model: str = "l2",
) -> List[int]:
    """
    PELT on a 1-D series (standardised internally).

    Returns CP indices in the *series index space* (1..len-1), excluding last point.
    """
    x = np.asarray(series_values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 6:
        return []

    mu_x = float(np.mean(x))
    sigma_x = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
    if sigma_x < 1e-12:
        return []

    x_std = (x - mu_x) / sigma_x

    try:
        algo = rpt.Pelt(model=str(model)).fit(x_std)
    except Exception:
        algo = rpt.Pelt(model="l2").fit(x_std)

    pen = _default_pen_value(int(x_std.size), scale=float(pen_scale))
    try:
        cps = algo.predict(pen=pen)
    except Exception as e:
        warnings.warn(f"PELT failed (pen={pen}): {e}")
        return []

    return [int(cp) for cp in cps if int(cp) < int(x.size)]


def consolidate_changepoints(
    cps: List[int],
    min_distance: int = 3,
    values: Optional[np.ndarray] = None,
) -> List[int]:
    """
    Merge CPs within *min_distance*.  If *values* is provided, keep the CP
    with the largest absolute jump.
    """
    if len(cps) <= 1:
        return cps
    cps_sorted = sorted(cps)
    clusters: List[List[int]] = [[cps_sorted[0]]]
    for cp in cps_sorted[1:]:
        if cp - clusters[-1][-1] <= min_distance:
            clusters[-1].append(cp)
        else:
            clusters.append([cp])

    result = []
    for cluster in clusters:
        if values is not None and len(cluster) > 1:
            best_cp = cluster[0]
            best_jump = 0.0
            for c in cluster:
                if 0 < c < len(values):
                    jump = abs(float(values[c]) - float(values[c - 1]))
                    if jump > best_jump:
                        best_jump = jump
                        best_cp = c
            result.append(best_cp)
        else:
            result.append(int(np.median(cluster)))
    return result
