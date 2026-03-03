"""
run_unified.py — Unified CV-based drift detection for duration AND routing
==========================================================================

Detection only — runs the algorithm and saves detected changepoints to Excel/CSV.
Evaluation (TP/FP/FN/F1) is done separately in a dedicated evaluation script.

Combines:
  • Duration detection  (pen_scale=3.0)  — same as main.py --drift duration
  • Routing  detection  (pen_scale=5.0)  — same as run_ostovar_cv_perpair.py

Both modes use the refactored CVDriftPipeline_v2 pipeline modules
(preparation → CV window selection → PELT → consensus).

Usage examples
--------------
  # Single file — detect both types
  python run_unified.py --file "log.xes" --drift duration routing

  # Single file — duration only
  python run_unified.py --file "log.xes" --drift duration

  # Batch — duration only, save to Excel
  python run_unified.py --dir "path/to/folder" --drift duration --out results.xlsx

  # Batch — routing only
  python run_unified.py --dir "path/to/folder" --drift routing --out results_routing.xlsx

  # Batch — both types in one run
  python run_unified.py --dir "path/to/folder" --drift duration routing --out results_both.xlsx

  # Override pen_scale if needed
  python run_unified.py --dir "path/to/folder" --drift routing --pen-scale-routing 4.0
"""

import argparse
import os
import json
import time
import sys
import io
import warnings

import pandas as pd

from main import run_pipeline_single, _load_log
from preparation import DEFAULT_PARAMS

warnings.filterwarnings("ignore")

# ── Algorithm label ──
ALGORITHM_NAME = "CVDrift_Unified"

# ── Type-specific penalty defaults ──
# Duration: pen_scale=3.0  (CVDriftPipeline_v2 default — more sensitive)
# Routing:  pen_scale=5.0  (Routing19 / Ostovar default — more conservative)
DEFAULT_PEN_SCALE_DURATION = 3.0
DEFAULT_PEN_SCALE_ROUTING  = 7.0


# ====================================================================
# CP extraction from pipeline result dict
# ====================================================================

def extract_detected_cps(results):
    """Pull consensus changepoint case indices from pipeline results."""
    cps = []
    for _dtype, res in results.items():
        det = res.get("detection")
        if det is None:
            continue
        # Duration consensus
        dc = det.get("duration_consensus")
        if dc is not None and len(dc) > 0 and "consensus_case" in dc.columns:
            cps.extend(int(c) for c in dc["consensus_case"])
        # Routing consensus
        rc = det.get("consensus_drifts")
        if rc is not None and len(rc) > 0 and "consensus_case" in rc.columns:
            cps.extend(int(c) for c in rc["consensus_case"])
    return sorted(set(cps))


# ====================================================================
# Core detection — runs each drift type with its own pen_scale
# ====================================================================

def run_detection(df, drift_types, pen_scale_duration=None, pen_scale_routing=None,
                  window_strategy=None):
    """
    Run drift detection with type-specific parameters.

    Parameters
    ----------
    window_strategy : str or None
        "cv_perpair" (default) — per-series optimal window via CV+knee.
        "mode_window" — compute mode of all per-series windows, apply uniformly.

    Returns
    -------
    combined_cps : list[int]
        All detected consensus CPs (union of both types).
    per_type_cps : dict
        {drift_type: [cp1, cp2, ...]}
    """
    if pen_scale_duration is None:
        pen_scale_duration = DEFAULT_PEN_SCALE_DURATION
    if pen_scale_routing is None:
        pen_scale_routing = DEFAULT_PEN_SCALE_ROUTING

    per_type_cps = {}

    for dtype in drift_types:
        params = dict(DEFAULT_PARAMS)
        if dtype == "duration":
            params["pen_scale"] = pen_scale_duration
        elif dtype == "routing":
            params["pen_scale"] = pen_scale_routing
        if window_strategy is not None:
            params["window_strategy"] = window_strategy

        results = run_pipeline_single(df, [dtype], params=params)
        cps = extract_detected_cps(results)
        per_type_cps[dtype] = cps

    combined_cps = sorted(set(
        cp for type_cps in per_type_cps.values() for cp in type_cps
    ))

    return combined_cps, per_type_cps


# ====================================================================
# Formatting helpers
# ====================================================================

def seconds_to_hhmmss(s):
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = int(s % 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"


# ====================================================================
# Single-file runner (console output)
# ====================================================================

def run_single(file_path, drift_types, pen_scale_duration=None,
               pen_scale_routing=None, window_strategy=None):
    """Run detection on a single file and print results."""
    df = _load_log(file_path)
    filename = os.path.basename(file_path)
    ws_label = window_strategy or "cv_perpair"
    print(f"Loaded: {filename}  ({len(df)} events)  window_strategy={ws_label}")

    t0 = time.time()
    combined_cps, per_type_cps = run_detection(
        df, drift_types, pen_scale_duration, pen_scale_routing,
        window_strategy=window_strategy,
    )
    elapsed = time.time() - t0

    print(f"\n{'='*60}")
    print(f"  RESULTS: {filename}")
    print(f"{'='*60}")
    for dtype in drift_types:
        cps = per_type_cps.get(dtype, [])
        psc = pen_scale_duration if dtype == "duration" else pen_scale_routing
        if psc is None:
            psc = DEFAULT_PEN_SCALE_DURATION if dtype == "duration" else DEFAULT_PEN_SCALE_ROUTING
        print(f"  {dtype:10s} CPs: {cps}  (pen_scale={psc})")
    print(f"  {'combined':10s} CPs: {combined_cps}")
    print(f"  Time: {elapsed:.1f}s ({seconds_to_hhmmss(elapsed)})")
    print(f"{'='*60}\n")

    return combined_cps, per_type_cps


# ====================================================================
# Batch runner (detection only — no evaluation)
# ====================================================================

def collect_log_files(folder):
    """Recursively collect supported event-log files, sorted."""
    extensions = (".xes", ".xes.gz", ".csv", ".mxml")
    files = []
    for root, _dirs, entries in os.walk(folder):
        for entry in entries:
            if any(entry.lower().endswith(ext) for ext in extensions):
                files.append(os.path.join(root, entry))
    files.sort()
    return files


def batch_detect(
    folder,
    out_path,
    drift_types,
    pen_scale_duration=None,
    pen_scale_routing=None,
    quiet=False,
    window_strategy=None,
):
    """Run detection on every event log in *folder* and save results."""
    folder = os.path.abspath(folder)
    out_path = os.path.abspath(out_path)
    log_files = collect_log_files(folder)

    if not log_files:
        print(f"No supported log files found in {folder}")
        return

    if pen_scale_duration is None:
        pen_scale_duration = DEFAULT_PEN_SCALE_DURATION
    if pen_scale_routing is None:
        pen_scale_routing = DEFAULT_PEN_SCALE_ROUTING

    ws_label = window_strategy or "cv_perpair"
    print(f"Found {len(log_files)} log file(s) in {folder}")
    print(f"Drift types: {drift_types}")
    print(f"pen_scale — duration: {pen_scale_duration}, routing: {pen_scale_routing}")
    print(f"window_strategy: {ws_label}")
    print(f"Output: {out_path}")
    print("=" * 100)

    rows = []

    for idx, log_path in enumerate(log_files, 1):
        fname = os.path.basename(log_path)
        print(f"\n[{idx:3d}/{len(log_files)}] {fname}")

        t0 = time.time()
        try:
            df = _load_log(log_path)

            # Suppress verbose pipeline output in batch mode
            if quiet:
                old_stdout, old_stderr = sys.stdout, sys.stderr
                sys.stdout = io.StringIO()
                sys.stderr = io.StringIO()

            combined_cps, per_type_cps = run_detection(
                df, drift_types, pen_scale_duration, pen_scale_routing,
                window_strategy=window_strategy,
            )

            if quiet:
                sys.stdout, sys.stderr = old_stdout, old_stderr

            elapsed = time.time() - t0

            row = {
                "Algorithm": ALGORITHM_NAME,
                "Log": fname,
                "Drift Types": ", ".join(drift_types),
            }
            if "duration" in drift_types:
                row["Duration CPs"] = json.dumps(per_type_cps.get("duration", []))
                row["Duration pen_scale"] = pen_scale_duration
            if "routing" in drift_types:
                row["Routing CPs"] = json.dumps(per_type_cps.get("routing", []))
                row["Routing pen_scale"] = pen_scale_routing
            row["Combined CPs"] = json.dumps(combined_cps)
            row["Num Detected CPs"] = len(combined_cps)
            row["Runtime (s)"] = round(elapsed, 3)
            row["Runtime (hh:mm:ss)"] = seconds_to_hhmmss(elapsed)
            rows.append(row)

            print(f"  Detected: {combined_cps}  | {elapsed:.1f}s")

        except Exception as e:
            if quiet:
                try:
                    sys.stdout, sys.stderr = old_stdout, old_stderr
                except Exception:
                    pass
            elapsed = time.time() - t0
            print(f"  ERROR: {e} ({elapsed:.1f}s)")
            row = {
                "Algorithm": ALGORITHM_NAME,
                "Log": fname,
                "Drift Types": ", ".join(drift_types),
            }
            if "duration" in drift_types:
                row["Duration CPs"] = "ERROR"
                row["Duration pen_scale"] = pen_scale_duration
            if "routing" in drift_types:
                row["Routing CPs"] = "ERROR"
                row["Routing pen_scale"] = pen_scale_routing
            row["Combined CPs"] = "ERROR"
            row["Num Detected CPs"] = 0
            row["Runtime (s)"] = round(elapsed, 3)
            row["Runtime (hh:mm:ss)"] = seconds_to_hhmmss(elapsed)
            rows.append(row)

    # ── Summary row ──
    tot_time = sum(r["Runtime (s)"] for r in rows if isinstance(r.get("Runtime (s)"), (int, float)))
    n_ok = sum(1 for r in rows if r.get("Combined CPs") != "ERROR")
    summary = {
        "Algorithm": ALGORITHM_NAME,
        "Log": f"=== TOTAL ({len(log_files)} files, {n_ok} OK) ===",
        "Drift Types": ", ".join(drift_types),
        "Combined CPs": "",
        "Num Detected CPs": "",
        "Runtime (s)": round(tot_time, 2),
        "Runtime (hh:mm:ss)": seconds_to_hhmmss(tot_time),
    }
    rows.append(summary)

    # ── Write output ──
    df_out = pd.DataFrame(rows)

    if out_path.lower().endswith((".xlsx", ".xls")):
        df_out.to_excel(out_path, index=False, sheet_name="Detection Results")
    else:
        df_out.to_csv(out_path, index=False)

    print(f"\n{'='*100}")
    print(f"Done: {n_ok}/{len(log_files)} files processed successfully")
    print(f"Total time: {tot_time:.0f}s ({tot_time/60:.1f} min)")
    print(f"Saved to: {out_path}")


# ====================================================================
# CLI
# ====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Unified CV-based drift detection (duration + routing). "
                    "Detection only — saves detected changepoints to Excel/CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_unified.py --file log.xes --drift duration routing
  python run_unified.py --file log.xes --drift duration
  python run_unified.py --dir logs/ --drift duration --out results.xlsx
  python run_unified.py --dir logs/ --drift routing --out results_routing.xlsx
  python run_unified.py --dir logs/ --drift duration routing --out results_both.xlsx
  python run_unified.py --dir logs/ --drift routing --pen-scale-routing 4.0
        """,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", help="Path to a single event-log file (CSV / XES)")
    group.add_argument("--dir", help="Folder containing event-log files (batch mode)")

    parser.add_argument(
        "--drift", nargs="+",
        choices=["duration", "routing"],
        default=["duration", "routing"],
        help="Drift type(s) to detect (default: both)",
    )
    parser.add_argument(
        "--out", default="results_unified.xlsx",
        help="Output file path — .xlsx or .csv (default: results_unified.xlsx)",
    )
    parser.add_argument(
        "--pen-scale-duration", type=float, default=DEFAULT_PEN_SCALE_DURATION,
        help=f"PELT penalty scale for duration detection (default: {DEFAULT_PEN_SCALE_DURATION})",
    )
    parser.add_argument(
        "--pen-scale-routing", type=float, default=DEFAULT_PEN_SCALE_ROUTING,
        help=f"PELT penalty scale for routing detection (default: {DEFAULT_PEN_SCALE_ROUTING})",
    )
    parser.add_argument(
        "--window-strategy",
        choices=["cv_perpair", "mode_window"],
        default="cv_perpair",
        help="Window selection strategy: 'cv_perpair' (per-series optimal window "
             "via CV+knee, default) or 'mode_window' (use the mode of all "
             "per-series windows as a single uniform window)",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-file pipeline output in batch mode",
    )

    args = parser.parse_args()

    if args.file:
        run_single(
            args.file, args.drift,
            pen_scale_duration=args.pen_scale_duration,
            pen_scale_routing=args.pen_scale_routing,
            window_strategy=args.window_strategy,
        )
    else:
        batch_detect(
            args.dir, args.out, args.drift,
            pen_scale_duration=args.pen_scale_duration,
            pen_scale_routing=args.pen_scale_routing,
            quiet=args.quiet,
            window_strategy=args.window_strategy,
        )


if __name__ == "__main__":
    main()
