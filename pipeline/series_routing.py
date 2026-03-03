# ============================================================
# pipeline/series_routing.py — Per-case routing probability series
# ============================================================
"""Build case-indexed routing probability series for activity pairs."""

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


def add_next_act(elog_seq: pd.DataFrame) -> pd.DataFrame:
    """Add a 'next_act' column (next activity within the same case)."""
    df = elog_seq.copy()
    df["_idx"] = np.arange(len(df))
    if "Event Index" in df.columns:
        df = df.sort_values([".case", "Event Index", "_idx"], kind="mergesort")
    else:
        df = df.sort_values([".case", ".start", "_idx"], kind="mergesort")
    df["next_act"] = df.groupby(".case")[".act"].shift(-1)
    return df.drop(columns=["_idx"])


def build_routing_pairs_from_elog(seq_with_next: pd.DataFrame, min_count: Optional[int] = None) -> pd.DataFrame:
    """Return a DataFrame of (from, to, n) activity pairs sorted by frequency."""
    tmp = seq_with_next.copy()
    tmp = tmp[(tmp[".act"].notna()) & (tmp["next_act"].notna()) & (tmp["next_act"].astype(str).str.strip() != "")]
    if len(tmp) == 0:
        return pd.DataFrame(columns=["from", "to", "n"])
    rp = (
        tmp.groupby([".act", "next_act"]).size()
        .reset_index(name="n").rename(columns={".act": "from", "next_act": "to"})
        .sort_values("n", ascending=False)
        .reset_index(drop=True)
    )
    if min_count is not None:
        rp = rp.loc[rp["n"] >= int(min_count)].copy().reset_index(drop=True)
    return rp


def series_routing_case_indexed(
    seq_with_next: pd.DataFrame,
    from_act: str,
    to_act: str,
    n_cases: int,
    case_to_orig: Optional[Dict[str, str]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    For each case *i*:
      p[i] = (# of from_act→to_act) / (# of from_act occurrences) if from_act appears,
      else NaN.

    Returns (values, times, cases, orig_cases) — all length *n_cases*.
    """
    n_cases = int(n_cases)
    df2 = seq_with_next.loc[seq_with_next[".act"] == from_act].copy()
    df2["ind"] = np.where(df2["next_act"] == to_act, 1.0, 0.0)
    df2["_case_num"] = pd.to_numeric(df2[".case"], errors="coerce")
    df2 = df2.dropna(subset=["_case_num"]).copy()
    df2["_case_num"] = df2["_case_num"].astype(int)

    case_prob = df2.groupby("_case_num")["ind"].mean()

    values = np.full(n_cases, np.nan, dtype=float)
    for ci, p in case_prob.items():
        if 0 <= int(ci) < n_cases:
            values[int(ci)] = float(p)

    cases = np.array([str(i) for i in range(n_cases)], dtype=object)

    orig_cases = np.array([str(i) for i in range(n_cases)], dtype=object)
    if case_to_orig is not None:
        for i in range(n_cases):
            orig_cases[i] = case_to_orig.get(str(i), str(i))
    if ".orig_case" in df2.columns:
        oc_map = df2.groupby("_case_num")[".orig_case"].first()
        for ci, oc_val in oc_map.items():
            if 0 <= int(ci) < n_cases:
                orig_cases[int(ci)] = oc_val

    times = np.empty(n_cases, dtype="datetime64[ns]")
    times[:] = np.datetime64("NaT")
    if ".start" in df2.columns:
        last_ts = df2.groupby("_case_num")[".start"].last()
        for ci, ts_val in last_ts.items():
            if 0 <= int(ci) < n_cases:
                try:
                    times[int(ci)] = ts_val
                except Exception:
                    pass

    return values, times, cases, orig_cases
