# ============================================================
# pipeline/rolling.py — Rolling window statistics
# ============================================================
"""Compute rolling mean / median over case-indexed arrays."""

from typing import Optional

import numpy as np
import pandas as pd


def window_stat_series(
    values: np.ndarray,
    times: Optional[np.ndarray],
    w: int,
    step: Optional[int] = None,
    stat: str = "mean",
    cases: Optional[np.ndarray] = None,
    orig_cases: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Rolling mean/median over a case-indexed array with NaNs preserved.
    Only keeps windows with >=25 % valid points.

    Returns a DataFrame with one row per valid window.
    """
    stat = stat.lower().strip()
    if stat not in ("mean", "median"):
        raise ValueError("stat must be 'mean' or 'median'")

    v = np.asarray(values, dtype=float)
    t = np.asarray(times) if times is not None else None
    cs = np.asarray(cases, dtype=object) if cases is not None else None
    ocs = np.asarray(orig_cases, dtype=object) if orig_cases is not None else None

    n = int(v.size)
    w = int(w)
    if n < w:
        return pd.DataFrame(columns=["win_id", "idx_start", "idx_end",
                                      "t_end", "case_end", "orig_case_end",
                                      "case_mid", "orig_case_mid", "value"])

    if step is None:
        step = max(1, w // 10)
    step = int(step)

    min_valid = max(2, int(w * 0.25))
    starts = list(range(0, n - w + 1, step))

    rows = []
    win_counter = 0
    for s in starts:
        e = s + w - 1
        win = v[s:s + w]
        nv = int(np.sum(np.isfinite(win)))
        if nv < min_valid:
            continue

        win_counter += 1
        val = float(np.nanmean(win) if stat == "mean" else np.nanmedian(win))
        t_end = t[e] if t is not None and len(t) > e else pd.NaT
        case_end = cs[e] if cs is not None and len(cs) > e else pd.NA
        orig_case_end = ocs[e] if ocs is not None and len(ocs) > e else pd.NA

        # Midpoint of the window — better represents where the drift is
        m = s + w // 2
        case_mid = cs[m] if cs is not None and len(cs) > m else pd.NA
        orig_case_mid = ocs[m] if ocs is not None and len(ocs) > m else pd.NA

        rows.append({
            "win_id": win_counter,
            "idx_start": s + 1,
            "idx_end": e + 1,
            "t_end": t_end,
            "case_end": case_end,
            "orig_case_end": orig_case_end,
            "case_mid": case_mid,
            "orig_case_mid": orig_case_mid,
            "value": val,
        })

    return pd.DataFrame(rows)
