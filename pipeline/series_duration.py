# ============================================================
# pipeline/series_duration.py — Per-case activity duration series
# ============================================================
"""Build a case-indexed duration series for a given activity."""

from typing import Tuple

import numpy as np
import pandas as pd


def series_duration_case_indexed(
    elog_dur: pd.DataFrame,
    activity: str,
    n_cases: int,
    how: str = "median",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    For each case *i*:
      d[i] = duration statistic for *activity* in that case (median by default),
      else NaN.

    Returns (values, times, cases, orig_cases) — all length *n_cases*.
    """
    how = how.lower().strip()
    if how not in ("first", "last", "mean", "median"):
        raise ValueError("how must be one of: first, last, mean, median")

    df = elog_dur.loc[elog_dur[".act"] == activity].copy()
    df["_case_num"] = pd.to_numeric(df[".case"], errors="coerce")
    df = df.dropna(subset=["_case_num"]).copy()
    df["_case_num"] = df["_case_num"].astype(int)

    values = np.full(int(n_cases), np.nan, dtype=float)
    times = np.full(int(n_cases), np.datetime64("NaT"), dtype="datetime64[ns]")

    if len(df) == 0:
        cases = np.array([str(i) for i in range(int(n_cases))], dtype=object)
        orig_cases = cases.copy()
        return values, times, cases, orig_cases

    df = df.sort_values(["_case_num", ".start"], kind="mergesort")

    if how == "first":
        per_case = df.groupby("_case_num")[".dur_sec"].first()
        per_time = df.groupby("_case_num")[".end"].first()
    elif how == "last":
        per_case = df.groupby("_case_num")[".dur_sec"].last()
        per_time = df.groupby("_case_num")[".end"].last()
    elif how == "mean":
        per_case = df.groupby("_case_num")[".dur_sec"].mean()
        per_time = df.groupby("_case_num")[".end"].last()
    else:  # median
        per_case = df.groupby("_case_num")[".dur_sec"].median()
        per_time = df.groupby("_case_num")[".end"].last()

    for ci, v in per_case.items():
        if 0 <= int(ci) < int(n_cases):
            values[int(ci)] = float(v)

    for ci, t in per_time.items():
        if 0 <= int(ci) < int(n_cases):
            try:
                times[int(ci)] = t
            except Exception:
                pass

    cases = np.array([str(i) for i in range(int(n_cases))], dtype=object)
    orig_cases = cases.copy()

    return values, times, cases, orig_cases
