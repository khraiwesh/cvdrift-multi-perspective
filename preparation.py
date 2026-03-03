# ============================================================
# preparation.py - Build case-indexed time series per drift type
#                   + CV-based window selection (separate step)
# ============================================================
"""
Prepare event-log time series for a specific drift type
(duration, routing, or arrival), then select the best window.

This is a structural wrapper around the existing pipeline modules -
the core CV-based window-selection algorithm is unchanged.

Usage::

    from preparation import preparation, preprocess
    from main import select_window

    preprocessed = preprocess(df)                        # parse once
    for drift_type in ["duration", "routing", "arrival"]:
        prep   = preparation(df, drift_type, preprocessed=preprocessed)
        winsel = select_window(prep)                     # CV-based window selection
        # prep["series_bundle"]  - raw case-indexed time series
        # prep["config"]         - detection parameters
        # winsel                 - window selection result (ready for detection)
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from pipeline.preprocessing import prepare_event_log_dual, prepare_seq_log
from pipeline.series_duration import series_duration_case_indexed
from pipeline.series_routing import (
    add_next_act,
    build_routing_pairs_from_elog,
    series_routing_case_indexed,
)
from pipeline.series_arrival import series_arrival_case_indexed


# ------------------------------------------------------------------
# Default pipeline parameters
# ------------------------------------------------------------------

DEFAULT_PARAMS = dict(
    CASE_COL="Case ID",
    ACT_COL="Activity",
    START_COL="Start Timestamp",
    END_COL="Complete Timestamp",
    RES_COL="Resource",
    tz="UTC",
    candidate_windows=[15, 20, 30, 50, 100, 200, 300, 400, 500, 600,
                       1000, 1500, 2000, 3000, 5000],
    duration_stat="median",
    duration_per_case="median",
    routing_stat="mean",
    routing_min_count=None,
    arrival_stat="median",
    arrival_max_gap_hours=4.0,
    arrival_same_day_only=True,
    knee_policy="before",
    window_strategy="cv_perpair",   # "cv_perpair" or "mode_window"
    pen_scale=3.0,
    cpd_model="l2",
    min_cp_distance=10,
    min_effect_size=0.15,
    min_n_points=10,
)


def _merge_params(overrides=None):
    """Return DEFAULT_PARAMS with optional overrides applied."""
    p = dict(DEFAULT_PARAMS)
    if overrides:
        p.update(overrides)
    return p


# ==================================================================
# Step 0 - Shared preprocessing  (call once, reuse across types)
# ==================================================================

def preprocess(df, params=None):
    """
    Parse the event log once, returning objects shared by all drift types.

    Returns
    -------
    dict with keys: logs, elog_seq, seq_with_next, n_cases, case_to_orig.
    """
    p = _merge_params(params)
    logs = prepare_event_log_dual(
        df, p["CASE_COL"], p["ACT_COL"], p["START_COL"], p["END_COL"],
        p["RES_COL"], tz=p["tz"],
    )
    elog_seq = prepare_seq_log(
        df, p["CASE_COL"], p["ACT_COL"], p["START_COL"], tz=p["tz"],
    )
    seq_with_next = add_next_act(elog_seq)
    n_cases = int(elog_seq[".case"].nunique())
    case_to_orig = (
        elog_seq.groupby(".case")[".orig_case"].first().to_dict()
        if ".orig_case" in elog_seq.columns else {}
    )
    return {
        "logs": logs,
        "elog_seq": elog_seq,
        "seq_with_next": seq_with_next,
        "n_cases": n_cases,
        "case_to_orig": case_to_orig,
    }


# ==================================================================
# Step 1 - PREPARATION  (build time series + config only)
# ==================================================================

def preparation(df, drift_type, params=None, preprocessed=None):
    """
    Build the initial case-indexed time series and assign the
    drift-specific configuration for **one** drift type.

    This function does NOT select a window - call select_window()
    afterwards with the result of this function.

    Returns
    -------
    dict
        drift_type    - echo of the requested type.
        series_bundle - list of dicts, each with the raw case-indexed
                        arrays (values, times, cases, orig_cases)
                        and metadata (label, n_valid, ...).
        config        - dict of detection-stage parameters.
        params        - merged parameter dict (for window selection).
        n_cases       - number of cases in the log.
    """
    drift_type = drift_type.lower().strip()
    valid = ("duration", "routing", "arrival")
    if drift_type not in valid:
        raise ValueError(
            f"drift_type must be one of {valid}, got '{drift_type}'"
        )

    p = _merge_params(params)
    if preprocessed is None:
        preprocessed = preprocess(df, params)

    # Build the raw time series (NO window selection here)
    builders = {
        "duration": _build_series_duration,
        "routing": _build_series_routing,
        "arrival": _build_series_arrival,
    }
    series_bundle, routing_meta = builders[drift_type](preprocessed, p)

    config = {
        "pen_scale": p["pen_scale"],
        "cpd_model": p["cpd_model"],
        "min_cp_distance": p["min_cp_distance"],
        "min_effect_size": p["min_effect_size"],
        "min_n_points": p["min_n_points"],
        "duration_per_case": p["duration_per_case"],
        "duration_roll_stat": p["duration_stat"],
        "routing_roll_stat": p["routing_stat"],
        "arrival_roll_stat": p["arrival_stat"],
    }

    result = {
        "drift_type": drift_type,
        "series_bundle": series_bundle,
        "config": config,
        "params": p,
        "n_cases": preprocessed["n_cases"],
    }
    if routing_meta is not None:
        result["routing_meta"] = routing_meta
    return result


# ==================================================================
# Series builders  (build only, no window selection)
# ==================================================================

def _build_series_duration(pre, p):
    """Build duration time series for every activity. No window selection."""
    elog_dur = pre["logs"].elog_dur
    n_cases = pre["n_cases"]

    activities = (
        elog_dur[".act"].dropna().astype(str).unique().tolist()
        if len(elog_dur) > 0 else []
    )

    bundle = []
    for a in activities:
        x, t, cases, orig_cases = series_duration_case_indexed(
            elog_dur, a, n_cases, how=p["duration_per_case"],
        )
        n_valid = int(np.sum(np.isfinite(x)))

        bundle.append({
            "label": f"duration::{a}",
            "activity": str(a),
            "values": x,
            "times": t,
            "cases": cases,
            "orig_cases": orig_cases,
            "n_valid": n_valid,
        })

    return bundle, None  # no routing meta


def _build_series_routing(pre, p):
    """Build routing probability series for every pair. No window selection."""
    seq_with_next = pre["seq_with_next"]
    n_cases = pre["n_cases"]
    case_to_orig = pre["case_to_orig"]

    routing_min_count = p["routing_min_count"]
    if routing_min_count is None:
        routing_min_count = max(10, int(np.sqrt(n_cases)))

    routing_pairs = build_routing_pairs_from_elog(
        seq_with_next, min_count=routing_min_count,
    )

    # Compute routing_min_mean_p (aligned with Routing19)
    routing_min_mean_p = None
    if len(routing_pairs) > 0:
        mean_ps = []
        for _, row in routing_pairs.iterrows():
            x_tmp, _, _, _ = series_routing_case_indexed(
                seq_with_next, str(row["from"]), str(row["to"]),
                n_cases, case_to_orig,
            )
            n_v = int(np.sum(np.isfinite(x_tmp)))
            if n_v > 0:
                mean_ps.append(float(np.nanmean(x_tmp)))
        if mean_ps:
            non_det = [mp for mp in mean_ps if mp < 0.95]
            if non_det:
                routing_min_mean_p = max(
                    0.01, min(0.30, float(np.percentile(non_det, 5)))
                )
            else:
                routing_min_mean_p = 0.01

    routing_meta = {
        "routing_min_count": int(routing_min_count),
        "routing_min_mean_p": routing_min_mean_p,
        "routing_pairs": routing_pairs,
    }

    bundle = []
    for _, row in routing_pairs.iterrows():
        from_act = str(row["from"])
        to_act = str(row["to"])

        x, t, cases, orig_cases = series_routing_case_indexed(
            seq_with_next, from_act, to_act, n_cases, case_to_orig,
        )
        n_valid = int(np.sum(np.isfinite(x)))
        mean_p = float(np.nanmean(x)) if n_valid > 0 else 0.0

        bundle.append({
            "label": f"routing::{from_act}->{to_act}",
            "from": from_act,
            "to": to_act,
            "values": x,
            "times": t,
            "cases": cases,
            "orig_cases": orig_cases,
            "n_valid": n_valid,
            "mean_p": mean_p,
        })

    return bundle, routing_meta


def _build_series_arrival(pre, p):
    """Build inter-arrival time series. No window selection."""
    elog_seq = pre["elog_seq"]
    n_cases = pre["n_cases"]
    case_to_orig = pre["case_to_orig"]

    x, t, cases, orig_cases = series_arrival_case_indexed(
        elog_seq, n_cases,
        max_gap_hours=p["arrival_max_gap_hours"],
        same_day_only=p["arrival_same_day_only"],
        case_to_orig=case_to_orig,
    )
    n_valid = int(np.sum(np.isfinite(x)))

    bundle = [{
        "label": "arrival::inter_arrival",
        "values": x,
        "times": t,
        "cases": cases,
        "orig_cases": orig_cases,
        "n_valid": n_valid,
    }]

    return bundle, None
