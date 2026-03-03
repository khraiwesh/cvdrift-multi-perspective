# ============================================================
# evaluation.py — Batch evaluation over a folder of event logs
# ============================================================
"""
Run the CV-based drift detection pipeline on every event log in a
folder and collect results into a single CSV file.

Usage:
    python evaluation.py --dir "path/to/folder" --drift durationcd "
    python evaluation.py --dir "path/to/folder" --drift duration routing --out results.csv
    python evaluation.py --dir "path/to/folder"                          # all drift types
"""

import os
import csv
import json
import time
import argparse

import pandas as pd

from main import run_pipeline_single, _load_log
from preparation import DEFAULT_PARAMS

ALGORITHM_NAME = "CVDriftPipeline"


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _collect_log_files(folder: str) -> list:
    """Recursively collect supported event-log paths, sorted."""
    extensions = (".xes", ".xes.gz", ".csv", ".mxml")
    files = []
    for root, _dirs, entries in os.walk(folder):
        for entry in entries:
            lower = entry.lower()
            if any(lower.endswith(ext) for ext in extensions):
                files.append(os.path.join(root, entry))
    files.sort()
    return files


def _fmt_duration(seconds: float) -> str:
    """Return *seconds* as hh:mm:ss."""
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
        # Duration consensus
        dc = det.get("duration_consensus")
        if dc is not None and len(dc) > 0 and "consensus_case" in dc.columns:
            cps.extend(int(c) for c in dc["consensus_case"])
        # Routing consensus
        rc = det.get("consensus_drifts")
        if rc is not None and len(rc) > 0 and "consensus_case" in rc.columns:
            cps.extend(int(c) for c in rc["consensus_case"])
        # Arrival drifts (no consensus — individual CPs)
        ad = det.get("arrival_drifts")
        if ad is not None and len(ad) > 0 and "cp_case" in ad.columns:
            cps.extend(int(c) for c in ad["cp_case"])
    return sorted(set(cps))


def _append_row(csv_path: str, row_dict: dict):
    """Append a single result row to *csv_path*, creating header if needed."""
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


# ------------------------------------------------------------------
# Batch loop
# ------------------------------------------------------------------

def batch_evaluate(
    folder: str,
    out_path: str,
    drift_types: list,
    actual_cps: list = None,
):
    """Process every event log in *folder* and write results to *out_path*."""
    csv_path = os.path.abspath(out_path)
    folder = os.path.abspath(folder)

    log_files = _collect_log_files(folder)
    if not log_files:
        print(f"No supported log files found in {folder}")
        return

    print(f"Found {len(log_files)} log file(s) in {folder}")
    print(f"Drift types: {drift_types}")
    print(f"Results will be appended to {csv_path}\n")

    params_json = json.dumps(DEFAULT_PARAMS, default=str)

    for idx, log_path in enumerate(log_files, 1):
        log_name = os.path.basename(log_path)
        print("=" * 60)
        print(f"[{idx}/{len(log_files)}] Processing: {log_name}")
        print("=" * 60)

        t0 = time.time()
        try:
            df = _load_log(log_path)
            results = run_pipeline_single(df, drift_types)
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
            error_params = dict(DEFAULT_PARAMS, error=str(exc))
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
    print(f"Batch complete. Results in {csv_path}")


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Batch-run CVDriftPipeline on all logs in a folder",
    )
    parser.add_argument(
        "--dir", required=True,
        help="Folder containing event-log files",
    )
    parser.add_argument(
        "--out", default="results.csv",
        help="Output CSV path (default: results.csv)",
    )
    parser.add_argument(
        "--drift", nargs="+",
        choices=["duration", "routing", "arrival"],
        default=["duration", "routing", "arrival"],
        help="Drift type(s) to detect (default: all three)",
    )
    parser.add_argument(
        "--actual-cps", default=None,
        help='Actual changepoints as JSON list, e.g. "[37, 75]"',
    )
    args = parser.parse_args()

    actual = json.loads(args.actual_cps) if args.actual_cps else []
    batch_evaluate(args.dir, args.out, args.drift, actual)


if __name__ == "__main__":
    main()
