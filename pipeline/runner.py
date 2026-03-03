# ============================================================
# pipeline/runner.py — Orchestrates window selection + drift detection
# ============================================================
"""High-level functions that wire all pipeline stages together."""

import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pipeline.preprocessing import prepare_event_log_dual, prepare_seq_log
from pipeline.window_selection import choose_window_size_stability
from pipeline.series_duration import series_duration_case_indexed
from pipeline.series_routing import (
    add_next_act,
    build_routing_pairs_from_elog,
    series_routing_case_indexed,
)
from pipeline.series_arrival import series_arrival_case_indexed
from pipeline.rolling import window_stat_series
from pipeline.drift_detection import detect_drift_pelt, consolidate_changepoints
from pipeline.consensus import compute_routing_consensus, compute_duration_consensus


# ============================================================
# A) Window selection for duration + routing + arrival time
# ============================================================

def select_windows_duration_and_routing(
    df: pd.DataFrame,
    case_col: str,
    act_col: str,
    start_col: str,
    end_col: str,
    res_col: Optional[str] = None,
    tz: str = "UTC",
    candidate_windows: Optional[List[int]] = None,
    duration_stat: str = "median",
    duration_per_case: str = "median",
    activities_for_duration: Optional[List[str]] = None,
    routing_stat: str = "mean",
    routing_min_count: Optional[int] = None,
    routing_pairs: Optional[pd.DataFrame] = None,
    knee_policy: str = "before",
    arrival_stat: str = "median",
    arrival_max_gap_hours: float = 4.0,
    arrival_same_day_only: bool = True,
    include_arrival: bool = True,
) -> Dict[str, Any]:
    """Select the best rolling-window size for every duration activity, routing pair, and arrival time."""
    if candidate_windows is None:
        candidate_windows = [15, 20, 30, 50, 100, 200, 300, 400, 500, 600, 1000, 1500, 2000, 3000, 5000]

    logs = prepare_event_log_dual(df, case_col, act_col, start_col, end_col, res_col, tz=tz)
    elog_seq = prepare_seq_log(df, case_col, act_col, start_col, tz=tz)
    seq_with_next = add_next_act(elog_seq)

    n_cases = int(elog_seq[".case"].nunique())
    case_to_orig = elog_seq.groupby(".case")[".orig_case"].first().to_dict() if ".orig_case" in elog_seq.columns else {}

    out: Dict[str, Any] = {
        "activity_duration": [],
        "routing_probability": [],
        "arrival_time": [],
        "meta": {
            "n_cases": n_cases,
            "candidate_windows": list(candidate_windows),
            "knee_policy": knee_policy,
            "arrival_max_gap_hours": arrival_max_gap_hours,
            "arrival_same_day_only": arrival_same_day_only,
        }
    }

    # ----- Duration window selection (per activity) -----
    elog_dur = logs.elog_dur
    if activities_for_duration is None:
        activities_for_duration = (
            elog_dur[".act"].dropna().astype(str).unique().tolist()
            if len(elog_dur) > 0 else []
        )

    for a in activities_for_duration:
        x, _t, _cases, _oc = series_duration_case_indexed(elog_dur, a, n_cases, how=duration_per_case)
        n_valid = int(np.sum(np.isfinite(x)))
        if n_valid == 0:
            out["activity_duration"].append({
                "activity": str(a),
                "note": "no_data",
                "reason": "Activity not present in any case (or durations missing).",
                "n_cases_with_data": 0
            })
            continue

        wres = choose_window_size_stability(
            x=x,
            candidate_windows=list(candidate_windows),
            stat=duration_stat,
            min_num_windows=2,
            fail_mode="return",
            knee_policy=knee_policy,
        )

        if wres.status != "ok":
            out["activity_duration"].append({
                "activity": str(a),
                "note": wres.status,
                "reason": wres.reason,
                "n_cases_with_data": n_valid
            })
            continue

        out["activity_duration"].append({
            "activity": str(a),
            "n_cases_with_data": n_valid,
            "chosen_window": int(wres.chosen_window),
            "chosen_cv": float(wres.chosen_cv),
            "details": wres.all_results.copy(),
        })

    # ----- Routing window selection (per pair, with min_mean_p filter) -----
    if routing_min_count is None:
        routing_min_count = max(10, int(np.sqrt(n_cases)))

    if routing_pairs is None:
        routing_pairs = build_routing_pairs_from_elog(seq_with_next, min_count=routing_min_count)
    else:
        routing_pairs = routing_pairs.loc[routing_pairs["n"] >= int(routing_min_count)].copy().reset_index(drop=True)

    # Compute routing_min_mean_p (aligned with Routing19)
    routing_min_mean_p = None
    if len(routing_pairs) > 0:
        mean_ps = []
        for _, row in routing_pairs.iterrows():
            x_tmp, _, _, _ = series_routing_case_indexed(seq_with_next, str(row["from"]), str(row["to"]), n_cases, case_to_orig)
            n_v = int(np.sum(np.isfinite(x_tmp)))
            if n_v > 0:
                mean_ps.append(float(np.nanmean(x_tmp)))
        if mean_ps:
            non_det = [p for p in mean_ps if p < 0.95]
            if non_det:
                routing_min_mean_p = max(0.01, min(0.30, float(np.percentile(non_det, 5))))
            else:
                routing_min_mean_p = 0.01

    out["meta"]["routing_min_count"] = int(routing_min_count)
    out["meta"]["routing_min_mean_p"] = routing_min_mean_p
    out["meta"]["routing_pairs"] = routing_pairs

    for _, row in routing_pairs.iterrows():
        from_act = str(row["from"])
        to_act = str(row["to"])

        x, _t, _cases, _oc = series_routing_case_indexed(seq_with_next, from_act, to_act, n_cases, case_to_orig)
        n_valid = int(np.sum(np.isfinite(x)))
        if n_valid == 0:
            out["routing_probability"].append({
                "from": from_act, "to": to_act,
                "note": "no_data",
                "reason": "No occurrences of from_act in any case.",
                "n_cases_with_data": 0
            })
            continue

        mean_p = float(np.nanmean(x))

        # Filter rare routes (aligned with Routing19)
        if routing_min_mean_p is not None and np.isfinite(mean_p) and mean_p < float(routing_min_mean_p):
            out["routing_probability"].append({
                "from": from_act, "to": to_act,
                "note": "filtered_rare_route",
                "reason": f"mean_p={mean_p:.6f} < routing_min_mean_p={routing_min_mean_p}",
                "n_cases_with_data": n_valid
            })
            continue

        wres = choose_window_size_stability(
            x=x,
            candidate_windows=list(candidate_windows),
            stat=routing_stat,
            min_num_windows=2,
            fail_mode="return",
            knee_policy=knee_policy,
        )

        if wres.status != "ok":
            out["routing_probability"].append({
                "from": from_act, "to": to_act,
                "note": wres.status,
                "reason": wres.reason,
                "n_cases_with_data": n_valid
            })
            continue

        out["routing_probability"].append({
            "from": from_act, "to": to_act,
            "mean_p": float(np.nanmean(x)),
            "n_cases_with_data": n_valid,
            "chosen_window": int(wres.chosen_window),
            "chosen_cv": float(wres.chosen_cv),
            "details": wres.all_results.copy(),
        })

    # ----- Arrival Time window selection -----
    if include_arrival:
        x_arr, t_arr, cases_arr, orig_arr = series_arrival_case_indexed(
            elog_seq, n_cases,
            max_gap_hours=arrival_max_gap_hours,
            same_day_only=arrival_same_day_only,
            case_to_orig=case_to_orig
        )
        n_valid_arr = int(np.sum(np.isfinite(x_arr)))

        if n_valid_arr == 0:
            out["arrival_time"].append({
                "note": "no_data",
                "reason": f"No valid arrivals (same_day_only={arrival_same_day_only}, max_gap_hours={arrival_max_gap_hours})",
                "n_cases_with_data": 0
            })
        else:
            wres_arr = choose_window_size_stability(
                x=x_arr,
                candidate_windows=list(candidate_windows),
                stat=arrival_stat,
                min_num_windows=2,
                fail_mode="return",
                knee_policy=knee_policy,
            )

            if wres_arr.status != "ok":
                out["arrival_time"].append({
                    "note": wres_arr.status,
                    "reason": wres_arr.reason,
                    "n_cases_with_data": n_valid_arr
                })
            else:
                out["arrival_time"].append({
                    "n_cases_with_data": n_valid_arr,
                    "mean_arrival_sec": float(np.nanmean(x_arr)),
                    "median_arrival_sec": float(np.nanmedian(x_arr)),
                    "chosen_window": int(wres_arr.chosen_window),
                    "chosen_cv": float(wres_arr.chosen_cv),
                    "details": wres_arr.all_results.copy(),
                })

    return out


# ============================================================
# B) Drift detection using selected windows
# ============================================================

def detect_drifts_duration_and_routing(
    df: pd.DataFrame,
    case_col: str,
    act_col: str,
    start_col: str,
    end_col: str,
    res_col: Optional[str] = None,
    tz: str = "UTC",
    window_selection: Optional[Dict[str, Any]] = None,
    duration_per_case: str = "median",
    duration_roll_stat: str = "median",
    routing_roll_stat: str = "mean",
    arrival_roll_stat: str = "median",
    step: Optional[int] = None,
    pen_scale: float = 5.0,
    cpd_model: str = "l2",
    min_cp_distance: int = 10,
    min_effect_size: float = 0.15,
    min_n_points: int = 10,
    plot: bool = False,
) -> Dict[str, Any]:
    """Run PELT drift detection on all rolling series built from the window selection.

    """
    if window_selection is None:
        raise ValueError("window_selection is required. Run select_windows_duration_and_routing() first.")

    logs = prepare_event_log_dual(df, case_col, act_col, start_col, end_col, res_col, tz=tz)
    elog_seq = prepare_seq_log(df, case_col, act_col, start_col, tz=tz)
    seq_with_next = add_next_act(elog_seq)

    n_cases = int(elog_seq[".case"].nunique())
    case_to_orig = elog_seq.groupby(".case")[".orig_case"].first().to_dict() if ".orig_case" in elog_seq.columns else {}

    series_list: List[pd.DataFrame] = []

    # ----- Duration series (MULTI-SCALE) -----
    da_map: Dict[str, int] = {}
    for it in window_selection.get("activity_duration", []):
        if not isinstance(it, dict) or "chosen_window" not in it:
            continue
        a = str(it["activity"])
        w = int(it["chosen_window"])
        x, t, cases, orig_cases = series_duration_case_indexed(logs.elog_dur, a, n_cases, how=duration_per_case)
        n_valid = int(np.sum(np.isfinite(x)))

        max_w = max(10, n_valid // 10)
        w = min(w, max_w)

        da_map[a] = w

        scales = [w]
        w_half = w // 2
        if w_half >= 5 and n_valid >= w_half * 2:
            scales.append(w_half)
        w_quarter = w // 4
        if w_quarter >= 5 and n_valid >= w_quarter * 2:
            scales.append(w_quarter)

        for sw in scales:
            auto_step = max(1, sw // 10) if step is None else step
            ts = window_stat_series(x, t, w=sw, step=auto_step, stat=duration_roll_stat, cases=cases, orig_cases=orig_cases)
            if len(ts) > 0:
                scale_tag = "" if sw == w else f"[w{sw}]"
                ts["param"] = f"duration::{a}{scale_tag}"
                ts["base_param"] = f"duration::{a}"
                series_list.append(ts)

    # ----- Routing series (MULTI-SCALE) -----
    rp_map: Dict[str, int] = {}
    for it in window_selection.get("routing_probability", []):
        if not isinstance(it, dict) or "chosen_window" not in it:
            continue
        from_act = str(it["from"])
        to_act = str(it["to"])
        w = int(it["chosen_window"])
        key = f"{from_act}->{to_act}"
        rp_map[key] = w

        x, t, cases, orig_cases = series_routing_case_indexed(seq_with_next, from_act, to_act, n_cases, case_to_orig)
        n_raw = int(np.sum(np.isfinite(x)))

        scales = [w]
        w_half = w // 2
        if w_half >= 10 and n_raw >= w_half * 2:
            scales.append(w_half)

        for sw in scales:
            auto_step = max(1, sw // 10) if step is None else step
            ts = window_stat_series(x, t, w=sw, step=auto_step, stat=routing_roll_stat, cases=cases, orig_cases=orig_cases)
            if len(ts) > 0:
                scale_tag = "" if sw == w else f"[w{sw}]"
                ts["param"] = f"routing::{key}{scale_tag}"
                ts["base_param"] = f"routing::{key}"
                series_list.append(ts)

    # ----- Arrival Time series (MULTI-SCALE) -----
    arrival_w_map: Dict[str, int] = {}
    for it in window_selection.get("arrival_time", []):
        if not isinstance(it, dict) or "chosen_window" not in it:
            continue
        w = int(it["chosen_window"])
        arrival_w_map["arrival"] = w

        meta = window_selection.get("meta", {})
        arr_max_gap = float(meta.get("arrival_max_gap_hours", 4.0))
        arr_same_day = bool(meta.get("arrival_same_day_only", True))

        x, t, cases, orig_cases = series_arrival_case_indexed(
            elog_seq, n_cases,
            max_gap_hours=arr_max_gap,
            same_day_only=arr_same_day,
            case_to_orig=case_to_orig
        )
        n_raw = int(np.sum(np.isfinite(x)))

        if n_raw < 10:
            continue

        max_w = max(10, n_raw // 10)
        w = min(w, max_w)
        arrival_w_map["arrival"] = w

        scales = [w]
        w_half = w // 2
        if w_half >= 5 and n_raw >= w_half * 2:
            scales.append(w_half)
        w_quarter = w // 4
        if w_quarter >= 5 and n_raw >= w_quarter * 2:
            scales.append(w_quarter)

        for sw in scales:
            auto_step = max(1, sw // 10) if step is None else step
            ts = window_stat_series(x, t, w=sw, step=auto_step, stat=arrival_roll_stat, cases=cases, orig_cases=orig_cases)
            if len(ts) > 0:
                scale_tag = "" if sw == w else f"[w{sw}]"
                ts["param"] = f"arrival::inter_arrival{scale_tag}"
                ts["base_param"] = "arrival::inter_arrival"
                series_list.append(ts)

    if not series_list:
        raise ValueError("No time series could be built. Check window_selection and data.")

    ts_df = pd.concat(series_list, ignore_index=True)
    ts_df["value"] = pd.to_numeric(ts_df["value"], errors="coerce")
    ts_df = ts_df[np.isfinite(ts_df["value"])].copy()
    ts_df = ts_df.sort_values(["param", "win_id"], kind="mergesort").reset_index(drop=True)

    drift_summary_rows = []
    drift_rows = []

    for param, g in ts_df.groupby("param", sort=False):
        values = g["value"].to_numpy(dtype=float)

        if len(values) < int(min_n_points):
            drift_summary_rows.append({
                "param": param,
                "cps": [],
                "n_cps": 0,
                "n_points": int(values.size),
                "note": "skipped_too_short"
            })
            continue

        cps = detect_drift_pelt(values, pen_scale=pen_scale, model=cpd_model)
        cps = consolidate_changepoints(cps, min_distance=int(min_cp_distance), values=values)

        # Effect-size filter (raw mean difference)
        if min_effect_size > 0 and cps:
            kept = []
            for cp in cps:
                seg_before = values[:cp]
                seg_after = values[cp:]
                if len(seg_before) >= 2 and len(seg_after) >= 2:
                    mean_diff = abs(float(np.mean(seg_before)) - float(np.mean(seg_after)))
                    if mean_diff >= float(min_effect_size):
                        kept.append(cp)
                else:
                    kept.append(cp)
            cps = kept

        drift_summary_rows.append({
            "param": param,
            "cps": cps,
            "n_cps": int(len(cps)),
            "n_points": int(values.size),
            "note": ""
        })

        if cps:
            g_idx = g.reset_index(drop=True)
            for cp in cps:
                j = max(1, min(int(cp), len(g_idx))) - 1
                row = g_idx.iloc[j]

                base_param_val = g_idx["base_param"].iloc[0] if "base_param" in g_idx.columns else param
                drift_rows.append({
                    "param": param,
                    "base_param": base_param_val,
                    "cp_series_index": int(cp),
                    "cp_win_id": int(row["win_id"]),
                    "cp_time": row["t_end"],
                    "cp_value": float(row["value"]),
                    "cp_case": row.get("case_mid", row["case_end"]),
                    "cp_orig_case": row.get("orig_case_mid", row["orig_case_end"]),
                })

    # ---- Raw PELT on per-case duration values (no rolling window → precise CPs) ----
    for it in window_selection.get("activity_duration", []):
        if not isinstance(it, dict) or "chosen_window" not in it:
            continue
        a = str(it["activity"])
        x_raw, t_raw, cases_raw, orig_raw = series_duration_case_indexed(
            logs.elog_dur, a, n_cases, how=duration_per_case
        )
        n_valid_raw = int(np.sum(np.isfinite(x_raw)))
        if n_valid_raw < 20:
            continue

        x_finite = x_raw[np.isfinite(x_raw)]
        valid_idx = np.where(np.isfinite(x_raw))[0]

        raw_cps = detect_drift_pelt(x_finite, pen_scale=pen_scale, model=cpd_model)
        raw_cps = consolidate_changepoints(
            raw_cps,
            min_distance=max(3, n_cases // 50),
            values=x_finite,
        )

        # Effect-size filter (Cohen's d >= 0.3)
        if raw_cps:
            overall_std = float(np.std(x_finite, ddof=1)) if len(x_finite) > 1 else 0.0
            if overall_std > 1e-12:
                kept = []
                for rcp in raw_cps:
                    seg_b = x_finite[:rcp]
                    seg_a = x_finite[rcp:]
                    if len(seg_b) >= 2 and len(seg_a) >= 2:
                        cohen_d = abs(float(np.mean(seg_b)) - float(np.mean(seg_a))) / overall_std
                        if cohen_d >= 0.3:
                            kept.append(rcp)
                    else:
                        kept.append(rcp)
                raw_cps = kept

        drift_summary_rows.append({
            "param": f"duration_raw::{a}",
            "cps": list(raw_cps),
            "n_cps": len(raw_cps),
            "n_points": n_valid_raw,
            "note": "raw_pelt",
        })

        for rcp in raw_cps:
            if rcp >= len(valid_idx):
                continue
            case_idx = int(valid_idx[rcp])
            drift_rows.append({
                "param": f"duration_raw::{a}",
                "base_param": f"duration::{a}",
                "cp_series_index": int(rcp),
                "cp_win_id": 0,
                "cp_time": t_raw[case_idx] if t_raw is not None and case_idx < len(t_raw) else pd.NaT,
                "cp_value": float(x_raw[case_idx]) if case_idx < len(x_raw) else 0.0,
                "cp_case": str(case_idx),
                "cp_orig_case": str(orig_raw[case_idx]) if orig_raw is not None and case_idx < len(orig_raw) else str(case_idx),
            })

    drifts_summary = pd.DataFrame(drift_summary_rows)
    if drift_rows:
        drifts = pd.DataFrame(drift_rows).sort_values(["param", "cp_win_id"], kind="mergesort").reset_index(drop=True)
    else:
        drifts = pd.DataFrame(columns=["param", "base_param", "cp_series_index", "cp_win_id", "cp_time", "cp_value", "cp_case", "cp_orig_case"])

    # Consensus voting
    consensus_drifts = compute_routing_consensus(drifts, ts_df, rp_map, n_cases)

    # Duration consensus — cluster nearby CPs, prefer raw-PELT positions
    duration_consensus = compute_duration_consensus(drifts, da_map, n_cases)

    # Arrival time drifts (no consensus — single series)
    arrival_drifts = drifts[drifts["param"].str.startswith("arrival::")].copy() if len(drifts) > 0 else pd.DataFrame()

    if plot:
        _plot_all_params_with_cps(ts_df, drifts)

    return {
        "series": ts_df,
        "drifts_summary": drifts_summary,
        "drifts": drifts,
        "consensus_drifts": consensus_drifts,
        "duration_consensus": duration_consensus,
        "arrival_drifts": arrival_drifts,
    }


# ------------------------------------------------------------------
# Plotting helper
# ------------------------------------------------------------------

def _plot_all_params_with_cps(series_df: pd.DataFrame, drifts_df: pd.DataFrame):
    """Simple plot per param: rolling series + vertical lines at detected CPs."""
    params = series_df["param"].unique().tolist()
    for p in params:
        g = series_df.loc[series_df["param"] == p].copy()
        g = g.sort_values("win_id")
        plt.figure()
        plt.plot(g["t_end"], g["value"])
        if len(drifts_df) > 0:
            d = drifts_df.loc[drifts_df["param"] == p]
            for _, r in d.iterrows():
                try:
                    plt.axvline(r["cp_time"], linestyle="--", alpha=0.3)
                except Exception:
                    pass
        plt.title(p)
        plt.xlabel("Time")
        plt.ylabel("Rolling value")
        plt.tight_layout()
        plt.show()
