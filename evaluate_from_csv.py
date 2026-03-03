"""
evaluate_from_csv.py — Compute P / R / F1 from a detection-results CSV
======================================================================

Reads a CSV produced by run_unified.py and evaluates detected CPs against
known ground truth.  GT is derived from the filename pattern:
    dataset_{N}cases_...   →  GT = [round(0.37*N), round(0.75*N)]

A detected CP is a True Positive if it falls within ±tolerance of a GT point.
Each GT point can match at most one detection (greedy closest-first).

Usage:
  python evaluate_from_csv.py --csv duration_verify.csv
  python evaluate_from_csv.py --csv duration_verify.csv --tol 50
  python evaluate_from_csv.py --csv duration_verify.csv --gt-mode ceravolo
  python evaluate_from_csv.py --csv Ceravolo_routing_results.csv --gt-mode ceravolo --cp-col "Routing CPs"
"""

import argparse
import json
import re
import sys

import numpy as np
import pandas as pd


# ── GT extraction helpers ──────────────────────────────────────────

def extract_n_cases_from_filename(fname: str) -> int | None:
    """Extract N from 'dataset_{N}cases...' pattern."""
    m = re.search(r"(\d+)\s*cases", fname, re.IGNORECASE)
    return int(m.group(1)) if m else None


def gt_samira(n_cases: int) -> list[int]:
    """Samira dataset: GT at 37% and 75% of n_cases."""
    return [round(0.37 * n_cases), round(0.75 * n_cases)]


def gt_ceravolo(n_cases: int) -> list[int]:
    """Ceravolo dataset: GT at case 499 (fixed)."""
    return [499]


def gt_ostovar(n_cases: int) -> list[int]:
    """Ostovar dataset: GT at case 999, 1999 (fixed)."""
    return [999, 1999]


GT_MODES = {
    "samira": gt_samira,
    "ceravolo": gt_ceravolo,
    "ostovar": gt_ostovar,
}


# ── Evaluation logic ───────────────────────────────────────────────

def evaluate_cps(detected: list[int], ground_truth: list[int], tolerance: int) -> dict:
    """
    Evaluate detected CPs against ground truth with a tolerance window.

    Greedy matching: for each GT point, find the closest unmatched detection
    within ±tolerance.

    Returns dict with TP, FP, FN, P, R, F1.
    """
    det = sorted(detected)
    gt = sorted(ground_truth)

    matched_det = set()
    tp = 0

    for g in gt:
        best_idx = None
        best_dist = float("inf")
        for i, d in enumerate(det):
            if i in matched_det:
                continue
            dist = abs(d - g)
            if dist <= tolerance and dist < best_dist:
                best_dist = dist
                best_idx = i
        if best_idx is not None:
            matched_det.add(best_idx)
            tp += 1

    fp = len(det) - tp
    fn = len(gt) - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1": round(f1, 4),
    }


# ── Main ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate detection results CSV against ground truth.",
    )
    parser.add_argument("--csv", required=True, help="Path to detection results CSV")
    parser.add_argument(
        "--gt-mode", choices=list(GT_MODES.keys()), default="samira",
        help="Ground-truth derivation mode (default: samira)",
    )
    parser.add_argument(
        "--tol", type=int, default=None,
        help="Tolerance window for matching (default: auto = max(round(0.05*n_cases), 10))",
    )
    parser.add_argument(
        "--cp-col", default=None,
        help='Column containing detected CPs (default: auto-detect from "Duration CPs", "Routing CPs", or "Combined CPs")',
    )
    parser.add_argument(
        "--out", default=None,
        help="Optional output CSV for per-file evaluation details",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    # Drop summary rows
    df = df[~df["Log"].astype(str).str.startswith("===")].copy()
    df = df.reset_index(drop=True)

    # Auto-detect CP column
    cp_col = args.cp_col
    if cp_col is None:
        for candidate in ["Duration CPs", "Routing CPs", "Combined CPs"]:
            if candidate in df.columns:
                cp_col = candidate
                break
    if cp_col is None or cp_col not in df.columns:
        print(f"ERROR: Cannot find CP column. Available: {df.columns.tolist()}")
        sys.exit(1)

    gt_func = GT_MODES[args.gt_mode]

    print(f"Evaluating: {args.csv}")
    print(f"CP column:  {cp_col}")
    print(f"GT mode:    {args.gt_mode}")
    print(f"Files:      {len(df)}")
    print("=" * 90)

    total_tp = 0
    total_fp = 0
    total_fn = 0
    detail_rows = []

    for _, row in df.iterrows():
        fname = str(row["Log"])

        # Parse detected CPs
        raw = row[cp_col]
        if pd.isna(raw) or str(raw).strip() in ("", "ERROR", "[]"):
            detected = []
        else:
            try:
                detected = json.loads(str(raw))
            except json.JSONDecodeError:
                detected = []

        # Get n_cases and GT
        n_cases = extract_n_cases_from_filename(fname)
        if n_cases is None:
            print(f"  SKIP {fname} — cannot extract n_cases from filename")
            continue

        gt = gt_func(n_cases)

        # Tolerance
        if args.tol is not None:
            tol = args.tol
        else:
            tol = max(round(0.10 * n_cases), 10)

        result = evaluate_cps(detected, gt, tol)
        total_tp += result["TP"]
        total_fp += result["FP"]
        total_fn += result["FN"]

        status = "OK" if result["FN"] == 0 and result["FP"] == 0 else ""
        if result["FP"] > 0:
            status += f"FP={result['FP']} "
        if result["FN"] > 0:
            status += f"FN={result['FN']} "

        detail_rows.append({
            "Log": fname,
            "n_cases": n_cases,
            "GT": str(gt),
            "Detected": str(detected),
            "Tol": tol,
            **result,
            "Status": status.strip(),
        })

        tag = "✓" if result["F1"] == 1.0 else "✗"
        print(f"  {tag} {fname[:55]:55s}  Det={str(detected):25s} GT={str(gt):15s}  "
              f"P={result['Precision']:.2f} R={result['Recall']:.2f} F1={result['F1']:.2f}  {status.strip()}")

    # ── Aggregate metrics ──
    print("\n" + "=" * 90)
    agg_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    agg_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    agg_f1 = 2 * agg_p * agg_r / (agg_p + agg_r) if (agg_p + agg_r) > 0 else 0.0

    print(f"  AGGREGATE over {len(detail_rows)} files:")
    print(f"    TP = {total_tp}   FP = {total_fp}   FN = {total_fn}")
    print(f"    Precision = {agg_p:.4f}")
    print(f"    Recall    = {agg_r:.4f}")
    print(f"    F1        = {agg_f1:.4f}")
    print("=" * 90)

    # ── Optional detail CSV ──
    if args.out:
        detail_df = pd.DataFrame(detail_rows)
        # Add aggregate row
        agg_row = {
            "Log": "=== AGGREGATE ===",
            "n_cases": "",
            "GT": "",
            "Detected": "",
            "Tol": "",
            "TP": total_tp,
            "FP": total_fp,
            "FN": total_fn,
            "Precision": round(agg_p, 4),
            "Recall": round(agg_r, 4),
            "F1": round(agg_f1, 4),
            "Status": "",
        }
        detail_df = pd.concat([detail_df, pd.DataFrame([agg_row])], ignore_index=True)
        detail_df.to_csv(args.out, index=False)
        print(f"  Detail saved to: {args.out}")


if __name__ == "__main__":
    main()
