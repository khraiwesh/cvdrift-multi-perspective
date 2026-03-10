# ============================================================
# run_combined.py — CV-based drift detection on one or many
#                   event logs (single file, multi-file, folder)
# ============================================================
"""
Run CV-based concept-drift detection on a single event log,
a set of explicitly listed files, or every log inside a folder.

Usage examples:
---------------------------------------------------------------------------
# Single file (equivalent to main.py):
    python run_combined.py --file "path/to/log.xes" --drift duration

# Multiple files (space-separated):
    python run_combined.py --files "log1.xes" "log2.csv" --drift duration routing

# Entire folder (equivalent to evaluation.py):
    python run_combined.py --dir "path/to/folder" --drift duration --out results.csv

# Custom window strategy:
    python run_combined.py --file "log.xes" --window-strategy mode_window
---------------------------------------------------------------------------
"""

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

import numpy as np
import pandas as pd

from pipeline.io import get_event_log, read_xes_to_dataframe
from pipeline.runner import detect_drifts_duration_and_routing
from pipeline.window_selection import choose_window_size_stability
from preparation import preparation, preprocess, DEFAULT_PARAMS
from evaluate_from_csv import (
    GT_MODES,
    run_evaluation,
)
from eval_routing import run_routing_evaluation


ALGORITHM_NAME = "CVDriftPipeline"


# ==================================================================
# Log loading
# ==================================================================

def _load_log(path: str, sep: str = ",") -> pd.DataFrame:
    """Load a single event-log file into a DataFrame (XES / CSV)."""
    lower = path.lower()
    if lower.endswith((".xes", ".xes.gz", ".xml")):
        return read_xes_to_dataframe(path, include_resource=True)
    return pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False)


# ==================================================================
# Pretty-printing helpers
# ==================================================================

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
        print("\n=== Consensus drifts (routing) ===")
        cd = drift_res.get("consensus_drifts")
        if cd is not None and len(cd) > 0:
            print(cd)
        else:
            print("No consensus routing drifts detected.")

    elif drift_type == "duration":
        print("\n=== Consensus drifts (duration) ===")
        dc = drift_res.get("duration_consensus")
        if dc is not None and len(dc) > 0:
            print(dc)
        else:
            print("No consensus duration drifts detected.")

    elif drift_type == "arrival":
        print("\n=== Arrival time drifts ===")
        ad = drift_res.get("arrival_drifts")
        if ad is not None and len(ad) > 0:
            print(ad)
        else:
            print("No arrival time drifts detected.")


# ==================================================================
# Window selection — internal helpers
# ==================================================================

def _apply_mode_window(win_sel: dict, drift_type: str):
    """
    Replace every per-series chosen_window with the mode (most
    frequently selected) window across all series of that drift type.
    Ties are broken by choosing the smallest window.
    """
    key_map = {
        "duration": "activity_duration",
        "routing":  "routing_probability",
        "arrival":  "arrival_time",
    }
    ws_key = key_map[drift_type]
    items = win_sel[ws_key]

    windows = [it["chosen_window"] for it in items if "chosen_window" in it]
    if not windows:
        return

    counts = Counter(windows)
    max_count = max(counts.values())
    candidates = [w for w, c in counts.items() if c == max_count]
    mode_w = min(candidates)

    print(f"  [mode_window] {drift_type}: distribution = {dict(counts)}")
    print(f"  [mode_window] {drift_type}: mode window = {mode_w}")

    for it in items:
        if "chosen_window" in it:
            it["chosen_window"] = mode_w

    win_sel["meta"]["mode_window"] = mode_w


def _select_window_duration(bundle, p, win_sel, prep_result):
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


def select_window(prep_result: dict) -> dict:
    """
    Select the best rolling-window size for each series using CV + knee.

    Two strategies are supported via ``window_strategy``:
    * ``"cv_perpair"``  — each series gets its own optimal window (default).
    * ``"mode_window"`` — individual windows are computed first, then all
      are replaced with the mode window across all series.
    """
    drift_type = prep_result["drift_type"]
    bundle = prep_result["series_bundle"]
    p = prep_result["params"]
    n_cases = prep_result["n_cases"]
    strategy = p.get("window_strategy", "cv_perpair").lower().strip()

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

    if "routing_meta" in prep_result:
        rm = prep_result["routing_meta"]
        win_sel["meta"]["routing_min_count"] = rm["routing_min_count"]
        win_sel["meta"]["routing_min_mean_p"] = rm["routing_min_mean_p"]
        win_sel["meta"]["routing_pairs"] = rm["routing_pairs"]

    selectors = {
        "duration": _select_window_duration,
        "routing":  _select_window_routing,
        "arrival":  _select_window_arrival,
    }
    selectors[drift_type](bundle, p, win_sel, prep_result)

    if strategy == "mode_window":
        _apply_mode_window(win_sel, drift_type)

    return win_sel


# ==================================================================
# Core pipeline (single log)
# ==================================================================

def run_pipeline_single(
    df: pd.DataFrame,
    drift_types: list,
    params: dict = None,
) -> dict:
    """
    Run the full pipeline on one log for the selected drift types.

    Returns
    -------
    dict
        Keyed by drift_type, each value holds
        ``{"preparation": …, "window_selection": …, "detection": …}``.
    """
    p = dict(DEFAULT_PARAMS)
    if params:
        p.update(params)

    preprocessed = preprocess(df, p)

    all_results: dict = {}
    for dtype in drift_types:
        print(f"\n{'=' * 60}")
        print(f"  Preparing: {dtype}")
        print(f"{'=' * 60}")

        prep = preparation(df, dtype, params=p, preprocessed=preprocessed)
        print(f"  Series built: {len(prep['series_bundle'])} series")

        win_sel = select_window(prep)
        _print_window_selection(dtype, win_sel)

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
            print(f"  [SKIP] {dtype}: {exc}")
            all_results[dtype] = {
                "preparation": prep,
                "window_selection": win_sel,
                "detection": None,
            }
            continue

        _print_drift_results(drift_res, dtype)
        all_results[dtype] = {
            "preparation": prep,
            "window_selection": win_sel,
            "detection": drift_res,
        }

    return all_results


# ==================================================================
# Batch evaluation helpers
# ==================================================================

def _collect_log_files(folder: str) -> list:
    """Recursively collect all supported event-log paths, sorted."""
    extensions = (".xes", ".xes.gz", ".csv", ".mxml")
    files = []
    for root, _dirs, entries in os.walk(folder):
        for entry in entries:
            if any(entry.lower().endswith(ext) for ext in extensions):
                files.append(os.path.join(root, entry))
    files.sort()
    return files


def _fmt_duration(seconds: float) -> str:
    """Return *seconds* formatted as hh:mm:ss."""
    total = int(seconds)
    h, remainder = divmod(total, 3600)
    m, s = divmod(remainder, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _extract_detected_cps(results: dict) -> list:
    """Pull consensus changepoint case indices from the pipeline results."""
    cps = []
    for _dtype, res in results.items():
        det = res.get("detection")
        if det is None:
            continue
        dc = det.get("duration_consensus")
        if dc is not None and len(dc) > 0 and "consensus_case" in dc.columns:
            cps.extend(int(c) for c in dc["consensus_case"])
        rc = det.get("consensus_drifts")
        if rc is not None and len(rc) > 0 and "consensus_case" in rc.columns:
            cps.extend(int(c) for c in rc["consensus_case"])
        ad = det.get("arrival_drifts")
        if ad is not None and len(ad) > 0 and "cp_case" in ad.columns:
            cps.extend(int(c) for c in ad["cp_case"])
    return sorted(set(cps))


def _append_row(csv_path: str, row_dict: dict):
    """Append a single result row to *csv_path*, creating the header if needed."""
    fieldnames = [
        "Algorithm",
        "Log Source",
        "Log",
        "Drift Types",
        "Detected Changepoints",
        "Actual Changepoints for Log",
        "Duration (Seconds)",
        "Duration (hh:mm:ss)",
        "Parameter Settings",
    ]
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row_dict)


# ==================================================================
# Batch evaluation loop
# ==================================================================

def batch_evaluate(
    log_paths: list,
    out_path: str,
    drift_types: list,
    actual_cps: list = None,
    params: dict = None,
):
    """
    Run the pipeline over all given log files and write results to
    *out_path* as a CSV.

    Parameters
    ----------
    log_paths   : list of file paths to process
    out_path    : output CSV path
    drift_types : drift types to detect
    actual_cps  : known ground-truth changepoints (optional)
    params      : dict overriding DEFAULT_PARAMS
    """
    csv_path = os.path.abspath(out_path)
    if not log_paths:
        print("No supported log files found.")
        return

    # Always start clean for batch runs
    if os.path.exists(csv_path):
        os.remove(csv_path)
        print(f"[INFO] Cleared existing file: {csv_path}")

    print(f"Found {len(log_paths)} log file(s) to process.")
    print(f"Drift types : {drift_types}")
    print(f"Output file : {csv_path}\n")

    p = dict(DEFAULT_PARAMS)
    if params:
        p.update(params)
    params_json = json.dumps(p, default=str)

    for idx, log_path in enumerate(log_paths, 1):
        log_name = os.path.basename(log_path)
        print("=" * 60)
        print(f"[{idx}/{len(log_paths)}] Processing: {log_name}")
        print("=" * 60)

        t0 = time.time()
        try:
            df = _load_log(log_path)
            results = run_pipeline_single(df, drift_types, params=p)
            detected = _extract_detected_cps(results)
            elapsed = time.time() - t0

            row = {
                "Algorithm": ALGORITHM_NAME,
                "Log Source": ", ".join(drift_types),
                "Log": log_name,
                "Drift Types": ", ".join(drift_types),
                "Detected Changepoints": str(detected),
                "Actual Changepoints for Log": str(actual_cps or []),
                "Duration (Seconds)": round(elapsed, 3),
                "Duration (hh:mm:ss)": _fmt_duration(elapsed),
                "Parameter Settings": params_json,
            }
        except Exception as exc:
            elapsed = time.time() - t0
            print(f"  ERROR on {log_name}: {exc}")
            error_params = dict(p, error=str(exc))
            row = {
                "Algorithm": ALGORITHM_NAME,
                "Log Source": ", ".join(drift_types),
                "Log": log_name,
                "Drift Types": ", ".join(drift_types),
                "Detected Changepoints": "ERROR",
                "Actual Changepoints for Log": str(actual_cps or []),
                "Duration (Seconds)": round(elapsed, 3),
                "Duration (hh:mm:ss)": _fmt_duration(elapsed),
                "Parameter Settings": json.dumps(error_params, default=str),
            }

        _append_row(csv_path, row)
        print(f"  -> Saved. Duration: {row['Duration (hh:mm:ss)']}\n")

    print("=" * 60)
    print(f"Batch complete. Results written to: {csv_path}")


# ==================================================================
# CLI entry point
# ==================================================================

def main():
    parser = argparse.ArgumentParser(
        description="CV-based drift detection — single file, multiple files, or a folder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file (output to stdout only)
  python run_combined.py --file log.xes --drift duration

  # Multiple explicit files
  python run_combined.py --files log1.xes log2.csv --drift duration routing --out results.csv

  # All logs inside a folder
  python run_combined.py --dir ./Datasets --drift duration routing arrival --out results.csv
        """,
    )

    # --- Input source (mutually exclusive) ---
    src = parser.add_mutually_exclusive_group()
    src.add_argument(
        "--file",
        metavar="FILE",
        help="Path to a single event-log file (CSV / XES). "
             "Omit for interactive file picker.",
    )
    src.add_argument(
        "--files",
        nargs="+",
        metavar="FILE",
        help="Paths to multiple event-log files (space-separated).",
    )
    src.add_argument(
        "--dir",
        metavar="FOLDER",
        help="Folder containing event-log files (scanned recursively).",
    )

    # --- Common options ---
    parser.add_argument(
        "--drift", nargs="+",
        choices=["duration", "routing", "arrival"],
        default=["duration", "routing", "arrival"],
        help="Drift type(s) to detect (default: all three).",
    )
    parser.add_argument(
        "--out", default="results.csv",
        help="Output CSV path for batch mode (default: results.csv).",
    )
    parser.add_argument(
        "--actual-cps", default=None,
        help='Ground-truth changepoints as a JSON list (single-file mode), e.g. "[37, 75]". '
             'If omitted, GT is auto-derived from the filename using --gt-mode.',
    )
    parser.add_argument(
        "--gt-mode",
        choices=list(GT_MODES.keys()),
        default="samira",
        help="How to derive ground-truth CPs from the filename (default: samira). "
             "Used when --actual-cps is not given.",
    )
    parser.add_argument(
        "--tol", type=int, default=None,
        help="Tolerance (in cases) for TP matching. Default: auto = max(10%% of n_cases, 10).",
    )
    parser.add_argument(
        "--window-strategy",
        choices=["cv_perpair", "mode_window"],
        default="cv_perpair",
        help="Window selection strategy (default: cv_perpair).",
    )

    args = parser.parse_args()

    # Build parameter dict
    params = dict(DEFAULT_PARAMS)
    params["window_strategy"] = args.window_strategy

    actual = json.loads(args.actual_cps) if args.actual_cps else []

    # Resolve output path — single file and batch use separate default files
    out_explicitly_set = args.out != "results.csv"
    _ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if out_explicitly_set:
        out_path = args.out
    elif args.file:
        out_path = os.path.join(OUTPUT_DIR, f"results_{_ts}.csv")
    else:
        out_path = os.path.join(OUTPUT_DIR, f"batch_results_{_ts}.csv")

    # ----------------------------------------------------------
    # Single-file mode
    # ----------------------------------------------------------
    if args.file:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

        t0 = time.time()
        df = _load_log(args.file)
        print(f"Loaded log: {args.file}  ({len(df)} events)")
        results = run_pipeline_single(df, args.drift, params=params)
        elapsed = time.time() - t0

        detected = _extract_detected_cps(results)

        row = {
            "Algorithm": ALGORITHM_NAME,
            "Log Source": os.path.basename(args.file),
            "Log": os.path.basename(args.file),
            "Drift Types": ", ".join(args.drift),
            "Detected Changepoints": str(detected),
            "Actual Changepoints for Log": str(actual),
            "Duration (Seconds)": round(elapsed, 3),
            "Duration (hh:mm:ss)": _fmt_duration(elapsed),
            "Parameter Settings": json.dumps(params, default=str),
        }
        _append_row(out_path, row)
        print(f"\n  -> Results saved to: {os.path.abspath(out_path)}")

    # ----------------------------------------------------------
    # Multi-file mode
    # ----------------------------------------------------------
    elif args.files:
        missing = [f for f in args.files if not os.path.isfile(f)]
        if missing:
            print(f"ERROR — file(s) not found: {missing}", file=sys.stderr)
            sys.exit(1)
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        batch_evaluate(args.files, out_path, args.drift, actual, params)

        # Auto-evaluate after batch
        eval_out = os.path.join(OUTPUT_DIR, "eval_" + os.path.splitext(os.path.basename(out_path))[0] + ".csv")
        if args.drift == ["routing"]:
            run_routing_evaluation(out_path, eval_out=eval_out)
        else:
            run_evaluation(out_path, gt_mode=args.gt_mode, tol=args.tol, eval_out=eval_out)

    # ----------------------------------------------------------
    # Folder mode
    # ----------------------------------------------------------
    elif args.dir:
        if not os.path.isdir(args.dir):
            print(f"ERROR — folder not found: {args.dir}", file=sys.stderr)
            sys.exit(1)
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        log_paths = _collect_log_files(args.dir)
        batch_evaluate(log_paths, out_path, args.drift, actual, params)

        # Auto-evaluate after batch
        eval_out = os.path.join(OUTPUT_DIR, "eval_" + os.path.splitext(os.path.basename(out_path))[0] + ".csv")
        if args.drift == ["routing"]:
            run_routing_evaluation(out_path, eval_out=eval_out)
        else:
            run_evaluation(out_path, gt_mode=args.gt_mode, tol=args.tol, eval_out=eval_out)

    # ----------------------------------------------------------
    # Interactive mode (no arguments supplied)
    # ----------------------------------------------------------
    else:
        df = get_event_log(sep=",")
        print(f"Loaded log ({len(df)} events)")
        run_pipeline_single(df, args.drift, params=params)


if __name__ == "__main__":
    main()
