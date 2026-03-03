# ============================================================
# pipeline/window_selection.py — CV-based rolling-window selection
# ============================================================
"""Choose the best rolling-window size using the CV + knee method."""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass
class WindowSelectionResult:
    """Outcome of the window-size stability search."""
    status: str
    chosen_window: Optional[int] = None
    chosen_cv: Optional[float] = None
    all_results: Optional[pd.DataFrame] = None
    reason: Optional[str] = None
    n: Optional[int] = None


def choose_window_size_stability(
    x: np.ndarray,
    candidate_windows: List[int],
    stat: str = "mean",
    min_num_windows: int = 2,
    fail_mode: str = "return",
    knee_policy: str = "before",
) -> WindowSelectionResult:
    """
    Compute CV of rolling mean/median for different windows (case-indexed series
    with NaNs allowed), then use a knee method on the CV curve to define a
    threshold and pick the smallest window under that threshold.
    """
    stat = stat.lower().strip()
    if stat not in ("mean", "median"):
        raise ValueError("stat must be 'mean' or 'median'")
    fail_mode = fail_mode.lower().strip()
    if fail_mode not in ("return", "stop"):
        raise ValueError("fail_mode must be 'return' or 'stop'")
    knee_policy = knee_policy.lower().strip()
    if knee_policy not in ("before", "after"):
        raise ValueError("knee_policy must be 'before' or 'after'")

    v = np.asarray(x, dtype=float)
    n = int(v.size)

    max_w = n - min_num_windows + 1
    cands = [int(w) for w in candidate_windows if int(w) <= max_w]
    if len(cands) == 0:
        msg = f"insufficient_data: n={n} cannot produce {min_num_windows} rolling windows for any candidate window."
        if fail_mode == "stop":
            raise ValueError(msg)
        return WindowSelectionResult(status="insufficient_data", reason=msg, n=n)

    rows = []
    for w in cands:
        vals = []
        min_valid_w = max(2, w // 4)
        for i in range(w, n + 1):
            win = v[(i - w):i]
            nv = int(np.sum(np.isfinite(win)))
            if nv < min_valid_w:
                continue
            vals.append(np.nanmean(win) if stat == "mean" else np.nanmedian(win))
        vals = np.asarray(vals, dtype=float)
        if vals.size < min_num_windows:
            continue
        mu = float(np.mean(vals))
        sigma = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
        cv = sigma / (abs(mu) + 1e-12)
        if np.isfinite(cv):
            rows.append({"window": w, "cv": cv, "n_windows": int(vals.size)})

    if not rows:
        msg = f"insufficient_data: all candidate windows produced NA/Inf CV (n={n})."
        if fail_mode == "stop":
            raise ValueError(msg)
        return WindowSelectionResult(status="insufficient_data", reason=msg, n=n)

    res = pd.DataFrame(rows).sort_values("window").reset_index(drop=True)

    # If ALL CVs are below 0.1 the series is essentially flat — no drift
    _LOW_CV_THRESHOLD = 0.05
    if float(res["cv"].max()) < _LOW_CV_THRESHOLD:
        msg = (f"no_drift: all CV values < {_LOW_CV_THRESHOLD} "
               f"(max CV = {res['cv'].max():.4f}). "
               f"Series is too stable for change-point detection.")
        if fail_mode == "stop":
            raise ValueError(msg)
        return WindowSelectionResult(
            status="no_drift", reason=msg, all_results=res, n=n)

    # Knee on (window, cv) after normalisation
    xw = res["window"].to_numpy(dtype=float)
    yw = res["cv"].to_numpy(dtype=float)
    xw_n = (xw - xw.min()) / (xw.max() - xw.min() + 1e-12)
    yw_n = (yw - yw.min()) / (yw.max() - yw.min() + 1e-12)

    line = np.array([xw_n[-1] - xw_n[0], yw_n[-1] - yw_n[0]])
    d = np.abs(line[0] * (yw_n - yw_n[0]) - line[1] * (xw_n - xw_n[0])) / (np.linalg.norm(line) + 1e-12)
    knee_idx = int(np.argmax(d))
    if knee_policy == "after":
        knee_idx = min(knee_idx + 1, len(res) - 1)

    cv_threshold = float(res.loc[knee_idx, "cv"])
    feasible = res.loc[res["cv"] <= cv_threshold].copy()
    chosen = feasible.iloc[0] if len(feasible) > 0 else res.iloc[int(res["cv"].values.argmin())]

    return WindowSelectionResult(
        status="ok",
        chosen_window=int(chosen["window"]),
        chosen_cv=float(chosen["cv"]),
        all_results=res,
        n=n
    )
