# ============================================================
# main.py — Single-file entry point for CV-based drift detection
# ============================================================
"""
Run concept-drift detection on **one** CSV / XES event log.

The user chooses which drift type(s) to detect:

    python main.py --file "path/to/log.csv" --drift duration
    python main.py --file "path/to/log.xes" --drift duration routing arrival
    python main.py                                  # interactive file picker, all types

For each selected drift type the pipeline:
  1. Calls ``preparation(df, drift_type)`` to build the case-indexed
     time series (no window selection yet).
  2. Calls ``select_window(prep)`` for CV-based window selection.
  3. Calls ``detect_drifts_duration_and_routing()`` (unchanged core)
     for PELT change-point detection + consensus.
"""

import argparse
import sys

import numpy as np
import pandas as pd

from pipeline.io import get_event_log, read_xes_to_dataframe
from pipeline.runner import detect_drifts_duration_and_routing
from pipeline.window_selection import choose_window_size_stability
from preparation import preparation, preprocess, DEFAULT_PARAMS


# ------------------------------------------------------------------
# Log loading
# ------------------------------------------------------------------

def _load_log(path: str, sep: str = ",") -> pd.DataFrame:
    """Load a single log file into a DataFrame."""
    lower = path.lower()
    if lower.endswith((".xes", ".xes.gz", ".xml")):
        return read_xes_to_dataframe(path, include_resource=True)
    return pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False)


# ------------------------------------------------------------------
# Pretty-printing helpers
# ------------------------------------------------------------------

def _print_window_selection(dtype: str, ws: dict):
    """Pretty-print the window selection result."""

    if dtype == "duration":
        print("\n=== Window selection: duration ===")
        for item in ws["activity_duration"]:
            if "chosen_window" in item:
                print(f"  {item['activity']}: "
                      f"window={item['chosen_window']}, "
                      f"CV={item['chosen_cv']:.4f}")
            else:
                print(f"  {item.get('activity', '?')}: "
                      f"{item.get('note', '?')} — "
                      f"{item.get('reason', '')}")

    elif dtype == "routing":
        print("\n=== Window selection: routing (first 10) ===")
        print(pd.DataFrame(ws["routing_probability"]).head(10))

    elif dtype == "arrival":
        print("\n=== Window selection: arrival time ===")
        for item in ws["arrival_time"]:
            if "chosen_window" in item:
                print(f"  Inter-arrival: "
                      f"window={item['chosen_window']}, "
                      f"CV={item['chosen_cv']:.4f}")
                print(f"    mean={item.get('mean_arrival_sec', 0) / 60:.1f} min, "
                      f"median={item.get('median_arrival_sec', 0) / 60:.1f} min")
            else:
                print(f"  Inter-arrival: "
                      f"{item.get('note', '?')} — "
                      f"{item.get('reason', '')}")


# ------------------------------------------------------------------
# Window selection  (CV + knee — moved here for visibility)
# ------------------------------------------------------------------

def select_window(prep_result):
    """
    Select the best rolling-window size for each series in the bundle
    using the CV + knee method.

    Supports two strategies (controlled by ``params["window_strategy"]``):

    * ``"cv_perpair"`` (default)  — each series / pair gets its own
      optimal window chosen via CV + knee.
    * ``"mode_window"`` — CV + knee runs per series as usual, but then
      ALL chosen windows are replaced with the **mode** (most frequent)
      window across all series of that drift type.  This gives a single
      uniform window, which can improve consensus corroboration.

    Parameters
    ----------
    prep_result : dict
        Output of preparation().

    Returns
    -------
    dict  (window_selection)
        Same structure as runner.select_windows_duration_and_routing()
        — ready to be passed into detect_drifts_duration_and_routing().
        Keys: activity_duration, routing_probability,
        arrival_time, meta.
    """
    drift_type = prep_result["drift_type"]
    bundle = prep_result["series_bundle"]
    p = prep_result["params"]
    n_cases = prep_result["n_cases"]
    strategy = p.get("window_strategy", "cv_perpair").lower().strip()

    # Skeleton (same shape as runner.py output)
    win_sel = {
        "activity_duration": [],
        "routing_probability": [],
        "arrival_time": [],
        "meta": {
            "n_cases": n_cases,
            "candidate_windows": list(p["candidate_windows"]),
            "knee_policy": p["knee_policy"],
            "arrival_max_gap_hours": p["arrival_max_gap_hours"],
            "arrival_same_day_only": p["arrival_same_day_only"],
            "window_strategy": strategy,
        },
    }

    # Copy routing meta if present
    if "routing_meta" in prep_result:
        rm = prep_result["routing_meta"]
        win_sel["meta"]["routing_min_count"] = rm["routing_min_count"]
        win_sel["meta"]["routing_min_mean_p"] = rm["routing_min_mean_p"]
        win_sel["meta"]["routing_pairs"] = rm["routing_pairs"]

    selectors = {
        "duration": _select_window_duration,
        "routing": _select_window_routing,
        "arrival": _select_window_arrival,
    }
    selectors[drift_type](bundle, p, win_sel, prep_result)

    # ── Mode-window override ──────────────────────────────────────
    if strategy == "mode_window":
        _apply_mode_window(win_sel, drift_type)

    return win_sel


def _apply_mode_window(win_sel, drift_type):
    """
    Replace every per-series chosen_window with the **mode** (most
    frequently selected) window across all series of that drift type.

    If two windows tie, the smaller one wins (favour sensitivity).
    """
    from collections import Counter

    key_map = {
        "duration": "activity_duration",
        "routing":  "routing_probability",
        "arrival":  "arrival_time",
    }
    ws_key = key_map[drift_type]
    items = win_sel[ws_key]

    # Collect all chosen windows
    windows = [it["chosen_window"] for it in items if "chosen_window" in it]
    if not windows:
        return  # nothing to override

    # Find the mode (ties broken by smallest window)
    counts = Counter(windows)
    max_count = max(counts.values())
    candidates = [w for w, c in counts.items() if c == max_count]
    mode_w = min(candidates)

    print(f"  [mode_window] {drift_type}: window distribution = {dict(counts)}")
    print(f"  [mode_window] {drift_type}: chosen mode window = {mode_w}")

    # Override every item that has a chosen_window
    for it in items:
        if "chosen_window" in it:
            it["chosen_window"] = mode_w

    win_sel["meta"]["mode_window"] = mode_w


def _select_window_duration(bundle, p, win_sel, prep_result):
    """Run CV-based window selection for each duration series."""
    for item in bundle:
        a = item["activity"]
        x = item["values"]
        n_valid = item["n_valid"]

        if n_valid == 0:
            win_sel["activity_duration"].append({
                "activity": str(a),
                "note": "no_data",
                "reason": "Activity not present in any case (or durations missing).",
                "n_cases_with_data": 0,
            })
            item["window_status"] = "no_data"
            continue

        wres = choose_window_size_stability(
            x=x,
            candidate_windows=list(p["candidate_windows"]),
            stat=p["duration_stat"],
            min_num_windows=2,
            fail_mode="return",
            knee_policy=p["knee_policy"],
        )

        if wres.status != "ok":
            win_sel["activity_duration"].append({
                "activity": str(a),
                "note": wres.status,
                "reason": wres.reason,
                "n_cases_with_data": n_valid,
            })
            item["window_status"] = wres.status
            continue

        win_sel["activity_duration"].append({
            "activity": str(a),
            "n_cases_with_data": n_valid,
            "chosen_window": int(wres.chosen_window),
            "chosen_cv": float(wres.chosen_cv),
            "details": wres.all_results.copy(),
        })
        item["window_status"] = "ok"
        item["chosen_window"] = int(wres.chosen_window)
        item["chosen_cv"] = float(wres.chosen_cv)


def _select_window_routing(bundle, p, win_sel, prep_result):
    """Run CV-based window selection for each routing pair."""
    routing_min_mean_p = None
    if "routing_meta" in prep_result:
        routing_min_mean_p = prep_result["routing_meta"]["routing_min_mean_p"]

    for item in bundle:
        from_act = item["from"]
        to_act = item["to"]
        x = item["values"]
        n_valid = item["n_valid"]
        mean_p = item.get("mean_p", 0.0)

        if n_valid == 0:
            win_sel["routing_probability"].append({
                "from": from_act, "to": to_act,
                "note": "no_data",
                "reason": "No occurrences of from_act in any case.",
                "n_cases_with_data": 0,
            })
            item["window_status"] = "no_data"
            continue

        # Filter rare routes (aligned with Routing19)
        if (routing_min_mean_p is not None
                and np.isfinite(mean_p)
                and mean_p < float(routing_min_mean_p)):
            win_sel["routing_probability"].append({
                "from": from_act, "to": to_act,
                "note": "filtered_rare_route",
                "reason": (f"mean_p={mean_p:.6f} < "
                           f"routing_min_mean_p={routing_min_mean_p}"),
                "n_cases_with_data": n_valid,
            })
            item["window_status"] = "filtered_rare_route"
            continue

        wres = choose_window_size_stability(
            x=x,
            candidate_windows=list(p["candidate_windows"]),
            stat=p["routing_stat"],
            min_num_windows=2,
            fail_mode="return",
            knee_policy=p["knee_policy"],
        )

        if wres.status != "ok":
            win_sel["routing_probability"].append({
                "from": from_act, "to": to_act,
                "note": wres.status,
                "reason": wres.reason,
                "n_cases_with_data": n_valid,
            })
            item["window_status"] = wres.status
            continue

        win_sel["routing_probability"].append({
            "from": from_act, "to": to_act,
            "mean_p": float(np.nanmean(x)),
            "n_cases_with_data": n_valid,
            "chosen_window": int(wres.chosen_window),
            "chosen_cv": float(wres.chosen_cv),
            "details": wres.all_results.copy(),
        })
        item["window_status"] = "ok"
        item["chosen_window"] = int(wres.chosen_window)
        item["chosen_cv"] = float(wres.chosen_cv)
        item["mean_p"] = mean_p


def _select_window_arrival(bundle, p, win_sel, prep_result):
    """Run CV-based window selection for the arrival series."""
    item = bundle[0]
    x = item["values"]
    n_valid = item["n_valid"]

    if n_valid == 0:
        win_sel["arrival_time"].append({
            "note": "no_data",
            "reason": (f"No valid arrivals "
                       f"(same_day_only={p['arrival_same_day_only']}, "
                       f"max_gap_hours={p['arrival_max_gap_hours']})"),
            "n_cases_with_data": 0,
        })
        item["window_status"] = "no_data"
        return

    wres = choose_window_size_stability(
        x=x,
        candidate_windows=list(p["candidate_windows"]),
        stat=p["arrival_stat"],
        min_num_windows=2,
        fail_mode="return",
        knee_policy=p["knee_policy"],
    )

    if wres.status != "ok":
        win_sel["arrival_time"].append({
            "note": wres.status,
            "reason": wres.reason,
            "n_cases_with_data": n_valid,
        })
        item["window_status"] = wres.status
        return

    win_sel["arrival_time"].append({
        "n_cases_with_data": n_valid,
        "mean_arrival_sec": float(np.nanmean(x)),
        "median_arrival_sec": float(np.nanmedian(x)),
        "chosen_window": int(wres.chosen_window),
        "chosen_cv": float(wres.chosen_cv),
        "details": wres.all_results.copy(),
    })
    item["window_status"] = "ok"
    item["chosen_window"] = int(wres.chosen_window)
    item["chosen_cv"] = float(wres.chosen_cv)
    item["mean_arrival_sec"] = float(np.nanmean(x))
    item["median_arrival_sec"] = float(np.nanmedian(x))


def _print_drift_results(drift_res: dict, drift_type: str):
    """Print drift detection results for one drift type."""
    with pd.option_context("display.max_rows", 50,
                           "display.max_columns", 10,
                           "display.width", 200):
        print(f"\n=== Drift summary ({drift_type}) ===")
        print(drift_res["drifts_summary"])

        print(f"\n=== Drift table ({drift_type}, first 50) ===")
        print(drift_res["drifts"].head(50))

    if drift_type == "routing":
        print(f"\n=== Consensus drifts (routing) ===")
        cd = drift_res.get("consensus_drifts")
        if cd is not None and len(cd) > 0:
            print(cd)
        else:
            print("No consensus routing drifts detected.")

    elif drift_type == "duration":
        print(f"\n=== Consensus drifts (duration) ===")
        dc = drift_res.get("duration_consensus")
        if dc is not None and len(dc) > 0:
            print(dc)
        else:
            print("No consensus duration drifts detected.")

    elif drift_type == "arrival":
        print(f"\n=== Arrival time drifts ===")
        ad = drift_res.get("arrival_drifts")
        if ad is not None and len(ad) > 0:
            print(ad)
        else:
            print("No arrival time drifts detected.")


# ------------------------------------------------------------------
# Pipeline runner  (called by main and by evaluation.py)
# ------------------------------------------------------------------

def run_pipeline_single(
    df: pd.DataFrame,
    drift_types: list,
    params: dict = None,
) -> dict:
    """
    Run the full pipeline on one log for the selected drift types.

    Returns a dict keyed by drift_type, each holding
    ``{"preparation": …, "detection": …}``.
    """
    p = dict(DEFAULT_PARAMS)
    if params:
        p.update(params)

    # Preprocess the log once (shared across drift types)
    preprocessed = preprocess(df, p)

    all_results: dict = {}
    for dtype in drift_types:
        print(f"\n{'=' * 60}")
        print(f"  Preparing: {dtype}")
        print(f"{'=' * 60}")

        # --- Step 1: preparation (build time series) -----
        prep = preparation(df, dtype, params=p, preprocessed=preprocessed)
        print(f"  Series built: {len(prep['series_bundle'])} series")

        # --- Step 2: window selection (CV + knee) --------
        win_sel = select_window(prep)
        _print_window_selection(dtype, win_sel)

        # --- Step 3: drift detection (PELT + consensus) --
        try:
            drift_res = detect_drifts_duration_and_routing(
                df=df,
                case_col=p["CASE_COL"],
                act_col=p["ACT_COL"],
                start_col=p["START_COL"],
                end_col=p["END_COL"],
                res_col=p["RES_COL"],
                tz=p.get("tz", "UTC"),
                window_selection=win_sel,
                duration_per_case=prep["config"]["duration_per_case"],
                duration_roll_stat=prep["config"]["duration_roll_stat"],
                routing_roll_stat=prep["config"]["routing_roll_stat"],
                arrival_roll_stat=prep["config"]["arrival_roll_stat"],
                step=None,
                pen_scale=prep["config"]["pen_scale"],
                cpd_model=prep["config"]["cpd_model"],
                min_cp_distance=prep["config"]["min_cp_distance"],
                min_effect_size=prep["config"]["min_effect_size"],
                min_n_points=prep["config"]["min_n_points"],
                plot=False,
            )
        except ValueError as exc:
            # Happens when no time series could be built (e.g. no valid data)
            print(f"  [SKIP] {dtype}: {exc}")
            all_results[dtype] = {"preparation": prep, "window_selection": win_sel, "detection": None}
            continue

        _print_drift_results(drift_res, dtype)
        all_results[dtype] = {
            "preparation": prep,
            "window_selection": win_sel,
            "detection": drift_res,
        }

    return all_results


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="CV-based drift detection on a single event log",
    )
    parser.add_argument(
        "--file", default=None,
        help="Path to event-log file (CSV / XES).  "
             "Omit for interactive file picker.",
    )
    parser.add_argument(
        "--drift", nargs="+",
        choices=["duration", "routing", "arrival"],
        default=["duration", "routing", "arrival"],
        help="Drift type(s) to detect (default: all three).",
    )
    parser.add_argument(
        "--window-strategy",
        choices=["cv_perpair", "mode_window"],
        default="cv_perpair",
        help="Window selection strategy: 'cv_perpair' (per-series optimal "
             "window via CV+knee, default) or 'mode_window' (use the mode "
             "of all per-series windows as a single uniform window).",
    )
    args = parser.parse_args()

    # Load log
    if args.file:
        df = _load_log(args.file)
        print(f"Loaded log: {args.file}  ({len(df)} events)")
    else:
        df = get_event_log(sep=",")
        print(f"Loaded log ({len(df)} events)")

    # Build params with chosen window strategy
    params = dict(DEFAULT_PARAMS)
    params["window_strategy"] = args.window_strategy

    # Run pipeline for each selected drift type
    run_pipeline_single(df, args.drift, params=params)


if __name__ == "__main__":
    main()

# %%
