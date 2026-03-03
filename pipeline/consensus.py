# ============================================================
# pipeline/consensus.py — Unified consensus voting for drift clusters
# ============================================================
"""
Unified consensus logic for both duration and routing drifts.

Algorithm (shared for both types):
  1. Filter drifts by type prefix ("duration::" or "routing::")
  2. Proximity clustering: merge CPs within tolerance = max(5, n_cases // 20)
  3. Summarise each cluster (votes, unique base_params, representative case)
  4. Type-specific filtering:
       Duration — prefer clusters backed by raw-PELT CPs (precise, no window lag)
       Routing  — require >= 2 unique transition pairs per cluster (corroboration)
"""

from typing import Dict

import numpy as np
import pandas as pd


# ================================================================
# Core unified consensus
# ================================================================

def _compute_consensus_core(
    drifts: pd.DataFrame,
    n_cases: int,
    drift_prefix: str,          # "routing" or "duration"
) -> pd.DataFrame:
    """
    Shared consensus algorithm for duration and routing drifts.

    Parameters
    ----------
    drifts : DataFrame
        All detected changepoints (columns: param, base_param, cp_case, …).
    n_cases : int
        Total number of cases in the event log.
    drift_prefix : str
        "routing" or "duration" — determines filtering and column names.

    Returns
    -------
    DataFrame with one row per consensus changepoint.
    """
    bp_col = "base_param" if "base_param" in drifts.columns else "param"

    # ── 1. Filter drifts by type ────────────────────────────────────────
    if drift_prefix == "duration":
        mask = (
            drifts[bp_col].str.startswith("duration::")
            if len(drifts) > 0
            else pd.Series(dtype=bool)
        )
    else:
        mask = (
            drifts["param"].str.startswith(f"{drift_prefix}::")
            if len(drifts) > 0
            else pd.Series(dtype=bool)
        )

    type_drifts = drifts.loc[mask].copy() if mask.any() else pd.DataFrame()

    if len(type_drifts) == 0 or "cp_case" not in type_drifts.columns:
        return pd.DataFrame()

    type_drifts["_case_num"] = pd.to_numeric(type_drifts["cp_case"], errors="coerce")
    type_drifts = type_drifts.dropna(subset=["_case_num"]).copy()

    if len(type_drifts) == 0:
        return pd.DataFrame()

    # ── 2. Proximity clustering ─────────────────────────────────────────
    sorted_df = type_drifts.sort_values("_case_num").reset_index(drop=True)
    case_vals = sorted_df["_case_num"].to_numpy(dtype=float)

    tolerance = max(5, n_cases // 20)

    clusters: list = []
    current: list = [0]
    for i in range(1, len(sorted_df)):
        if (case_vals[i] - case_vals[current[-1]]) <= tolerance:
            current.append(i)
        else:
            clusters.append(current)
            current = [i]
    clusters.append(current)

    # ── 3. Summarise each cluster ───────────────────────────────────────
    entity_label = "activities" if drift_prefix == "duration" else "pairs"
    rows = []

    for cluster_idxs in clusters:
        cluster_df = sorted_df.iloc[cluster_idxs]
        n_votes = len(cluster_df)
        n_unique = cluster_df[bp_col].nunique()

        # Representative case index
        if drift_prefix == "duration":
            raw_mask = cluster_df["param"].str.startswith("duration_raw::")
            has_raw = bool(raw_mask.any())
            if has_raw:
                rep_case = int(cluster_df.loc[raw_mask, "_case_num"].median())
            else:
                rep_case = int(cluster_df["_case_num"].median())
        else:
            rep_case = int(cluster_df["_case_num"].median())
            has_raw = False

        closest_idx = (abs(cluster_df["_case_num"] - rep_case)).idxmin()
        rep = cluster_df.loc[closest_idx]

        row_data = {
            "consensus_case": str(rep_case),
            "consensus_orig_case": rep["cp_orig_case"],
            "consensus_time": rep["cp_time"] if "cp_time" in rep.index else pd.NaT,
            "n_votes": n_votes,
            f"n_unique_{entity_label}": n_unique,
            f"supporting_{entity_label}": ", ".join(sorted(cluster_df[bp_col].unique())),
        }
        if drift_prefix == "duration":
            row_data["has_raw_support"] = has_raw

        rows.append(row_data)

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows).sort_values("n_votes", ascending=False).reset_index(drop=True)

    # ── 4. Type-specific filtering ──────────────────────────────────────
    if drift_prefix == "duration":
        # Prefer raw-PELT supported clusters (precise, no window lag).
        # If any cluster has raw support, drop rolling-only clusters.
        # Fallback: if no raw CPs exist at all, keep all clusters.
        if "has_raw_support" in result.columns and result["has_raw_support"].any():
            result = result.loc[result["has_raw_support"]].reset_index(drop=True)
    else:
        # Routing: require >= 2 unique transition pairs per cluster.
        # A single noisy pair can fire anywhere; corroboration from a
        # second independent pair is the minimal evidence of a real drift.
        unique_col = f"n_unique_{entity_label}"
        result = result.loc[result[unique_col] >= 2].reset_index(drop=True)

    return result


# ================================================================
# Public API — backward-compatible signatures
# ================================================================

def compute_routing_consensus(
    drifts: pd.DataFrame,
    ts_df: pd.DataFrame,
    rp_map: Dict[str, int],
    n_cases: int,
) -> pd.DataFrame:
    """
    Routing consensus via unified proximity clustering + min 2 unique pairs.

    ``ts_df`` and ``rp_map`` are accepted for API compatibility with
    runner.py but are not used by the unified algorithm.
    """
    return _compute_consensus_core(drifts, n_cases, drift_prefix="routing")


def compute_duration_consensus(
    drifts: pd.DataFrame,
    da_map: Dict[str, int],
    n_cases: int,
) -> pd.DataFrame:
    """
    Duration consensus via unified proximity clustering + raw-PELT preference.

    ``da_map`` is accepted for API compatibility with runner.py but is
    not used by the unified algorithm.
    """
    return _compute_consensus_core(drifts, n_cases, drift_prefix="duration")
