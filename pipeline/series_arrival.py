# ============================================================
# pipeline/series_arrival.py — Per-case inter-arrival time series
# ============================================================
"""Build a case-indexed inter-arrival time series (same-day only)."""

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


def series_arrival_case_indexed(
    elog_seq: pd.DataFrame,
    n_cases: int,
    max_gap_hours: float = 4.0,
    same_day_only: bool = True,
    case_to_orig: Optional[Dict[str, str]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    For each case *i* (i >= 1):
      arrival_time[i] = start_time[i] − start_time[i−1]  (seconds).

    Filters:
      - same_day_only: only if both cases started on the same calendar day.
      - max_gap_hours: arrivals exceeding this are set to NaN (overnight / closure).

    Case 0 always has NaN (no previous case).

    Returns (values, times, cases, orig_cases) — all length *n_cases*.
    """
    n_cases = int(n_cases)
    values = np.full(n_cases, np.nan, dtype=float)
    times = np.full(n_cases, np.datetime64("NaT"), dtype="datetime64[ns]")
    cases = np.array([str(i) for i in range(n_cases)], dtype=object)
    orig_cases = cases.copy()

    if len(elog_seq) == 0:
        return values, times, cases, orig_cases

    # Get first event timestamp per case (case start time)
    df = elog_seq.copy()
    df["_case_num"] = pd.to_numeric(df[".case"], errors="coerce")
    df = df.dropna(subset=["_case_num", ".start"]).copy()
    df["_case_num"] = df["_case_num"].astype(int)

    case_starts = df.groupby("_case_num")[".start"].min().sort_index()

    # Orig-case mapping
    if ".orig_case" in df.columns:
        oc_map = df.groupby("_case_num")[".orig_case"].first()
        for ci, oc_val in oc_map.items():
            if 0 <= int(ci) < n_cases:
                orig_cases[int(ci)] = str(oc_val)
    elif case_to_orig is not None:
        for i in range(n_cases):
            orig_cases[i] = case_to_orig.get(str(i), str(i))

    case_indices = case_starts.index.to_numpy()
    start_times = case_starts.values

    # Store start times
    for ci, ts in zip(case_indices, start_times):
        if 0 <= int(ci) < n_cases:
            try:
                times[int(ci)] = ts
            except Exception:
                pass

    # Compute inter-arrival times
    max_gap_sec = float(max_gap_hours) * 3600.0

    for i in range(1, len(case_indices)):
        ci = int(case_indices[i])
        ci_prev = int(case_indices[i - 1])

        if ci_prev != ci - 1:
            continue

        ts_curr = start_times[i]
        ts_prev = start_times[i - 1]

        if pd.isna(ts_curr) or pd.isna(ts_prev):
            continue

        ts_curr = pd.Timestamp(ts_curr)
        ts_prev = pd.Timestamp(ts_prev)

        if same_day_only:
            try:
                if ts_curr.date() != ts_prev.date():
                    continue
            except Exception:
                continue

        delta_sec = (ts_curr - ts_prev).total_seconds()

        if delta_sec < 0:
            continue
        if delta_sec > max_gap_sec:
            continue

        if 0 <= ci < n_cases:
            values[ci] = delta_sec

    return values, times, cases, orig_cases
