"""Compute Precision, Recall, and F1-score from resultsFinal.cvs
GT at 37% and 75% of dataset size.  Tolerance is 10% of dataset size.
"""
import os
import re
import ast
import pandas as pd
from collections import defaultdict


def extract_dataset_size(log_name: str) -> int:
    """Extract number of cases from log filename (e.g., 'dataset_500cases_20min_ABD.xes' -> 500)."""
    match = re.search(r'(\d+)cases', log_name.lower())
    if match:
        return int(match.group(1))
    return None


def extract_noise_level(log_name: str) -> str:
    """Extract noise level from filename. Returns '0%' for clean logs."""
    match = re.search(r'noisy_(\d+)%', log_name)
    if match:
        return f"{match.group(1)}%"
    return "0%"


def compute_tp_fp_fn(detected: list, actual: list, tolerance: int) -> tuple:
    """
    Compute TP, FP, FN given detected and actual changepoints.
    
    Args:
        detected: List of detected changepoint indices
        actual: List of actual changepoint indices
        tolerance: Window around actual CP to consider as match
    
    Returns:
        (tp, fp, fn) tuple
    """
    if not actual:
        # No ground truth
        return 0, len(detected), 0
    
    if not detected:
        # No detections
        return 0, 0, len(actual)
    
    matched_actual = set()
    matched_detected = set()
    
    # For each detected CP, check if it matches any actual CP within tolerance
    for det in detected:
        for i, act in enumerate(actual):
            if abs(det - act) <= tolerance and i not in matched_actual:
                matched_actual.add(i)
                matched_detected.add(det)
                break
    
    tp = len(matched_actual)
    fp = len(detected) - len(matched_detected)
    fn = len(actual) - len(matched_actual)
    
    return tp, fp, fn


def compute_metrics(tp: int, fp: int, fn: int) -> tuple:
    """Compute Precision, Recall, F1 from TP, FP, FN."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1


def main():
    import sys
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "resultsFinal.cvs",
        )
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found!")
        return
    
    df = pd.read_csv(csv_path)
    
    # Parse detected and actual changepoints
    print("Processing results...\n")
    
    results_by_size = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "count": 0})
    results_by_noise = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "count": 0})
    results_by_size_noise = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "count": 0})
    per_file_rows = []
    total_tp, total_fp, total_fn = 0, 0, 0
    
    for idx, row in df.iterrows():
        log_name = row["Log"]
        dataset_size = extract_dataset_size(log_name)
        
        if dataset_size is None:
            print(f"Warning: Could not extract dataset size from '{log_name}', skipping...")
            continue
        
        # 10% tolerance
        tolerance = int(0.10 * dataset_size)
        
        # Parse detected changepoints
        detected_str = row["Detected Changepoints"]
        if detected_str == "ERROR" or pd.isna(detected_str):
            detected = []
        else:
            try:
                detected = ast.literal_eval(detected_str)
            except:
                detected = []
        
        # Compute GT from filename: 37% and 75% of dataset size
        actual = [int(round(0.37 * dataset_size)), int(round(0.75 * dataset_size))]
        
        noise = extract_noise_level(log_name)
        
        # Compute metrics
        tp, fp, fn = compute_tp_fp_fn(detected, actual, tolerance)
        p, r, f1 = compute_metrics(tp, fp, fn)
        
        per_file_rows.append({
            "Log": log_name, "Size": dataset_size, "Noise": noise,
            "Detected": detected, "GT": actual, "Tolerance": tolerance,
            "TP": tp, "FP": fp, "FN": fn,
            "Precision": round(p, 4), "Recall": round(r, 4), "F1": round(f1, 4),
        })
        
        # Accumulate by dataset size
        results_by_size[dataset_size]["tp"] += tp
        results_by_size[dataset_size]["fp"] += fp
        results_by_size[dataset_size]["fn"] += fn
        results_by_size[dataset_size]["count"] += 1
        
        # Accumulate by noise level
        results_by_noise[noise]["tp"] += tp
        results_by_noise[noise]["fp"] += fp
        results_by_noise[noise]["fn"] += fn
        results_by_noise[noise]["count"] += 1
        
        # Accumulate by (size, noise)
        key = (dataset_size, noise)
        results_by_size_noise[key]["tp"] += tp
        results_by_size_noise[key]["fp"] += fp
        results_by_size_noise[key]["fn"] += fn
        results_by_size_noise[key]["count"] += 1
        
        # Accumulate totals
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
        print(f"{log_name:50s} | Size={dataset_size:4d} | Noise={noise:3s} | Tol={tolerance:3d} | Detected={str(detected):30s} | GT={actual} | TP={tp} FP={fp} FN={fn}")
    
    # Display results by dataset size
    print("\n" + "=" * 80)
    print("RESULTS BY DATASET SIZE")
    print("=" * 80)
    
    for size in sorted(results_by_size.keys()):
        stats = results_by_size[size]
        p, r, f1 = compute_metrics(stats["tp"], stats["fp"], stats["fn"])
        
        print(f"\nDataset Size: {size} cases ({stats['count']} files)")
        print(f"  TP={stats['tp']}, FP={stats['fp']}, FN={stats['fn']}")
        print(f"  Precision = {p:.4f}")
        print(f"  Recall    = {r:.4f}")
        print(f"  F1-Score  = {f1:.4f}")
    
    # Display results by noise level
    print("\n" + "=" * 80)
    print("RESULTS BY NOISE LEVEL")
    print("=" * 80)
    
    for noise in sorted(results_by_noise.keys(), key=lambda x: int(x.replace('%', ''))):
        stats = results_by_noise[noise]
        p, r, f1 = compute_metrics(stats["tp"], stats["fp"], stats["fn"])
        label = "Clean" if noise == "0%" else f"Noisy {noise}"
        print(f"\n{label} ({stats['count']} files)")
        print(f"  TP={stats['tp']}, FP={stats['fp']}, FN={stats['fn']}")
        print(f"  Precision = {p:.4f}")
        print(f"  Recall    = {r:.4f}")
        print(f"  F1-Score  = {f1:.4f}")
    
    # Display results by (size, noise)
    print("\n" + "=" * 80)
    print("RESULTS BY DATASET SIZE x NOISE LEVEL")
    print("=" * 80)
    
    sizes = sorted(set(k[0] for k in results_by_size_noise.keys()))
    noises = sorted(set(k[1] for k in results_by_size_noise.keys()), key=lambda x: int(x.replace('%', '')))
    
    # Table header
    hdr = f"{'Size':>6s} | {'Noise':>6s} | {'Files':>5s} | {'TP':>3s} | {'FP':>3s} | {'FN':>3s} | {'Prec':>6s} | {'Rec':>6s} | {'F1':>6s}"
    print(f"\n{hdr}")
    print("-" * len(hdr))
    for size in sizes:
        for noise in noises:
            key = (size, noise)
            if key not in results_by_size_noise:
                continue
            stats = results_by_size_noise[key]
            p, r, f1 = compute_metrics(stats["tp"], stats["fp"], stats["fn"])
            label = "Clean" if noise == "0%" else noise
            print(f"{size:>6d} | {label:>6s} | {stats['count']:>5d} | {stats['tp']:>3d} | {stats['fp']:>3d} | {stats['fn']:>3d} | {p:>6.4f} | {r:>6.4f} | {f1:>6.4f}")
    
    # Display overall results
    print("\n" + "=" * 80)
    print("OVERALL RESULTS")
    print("=" * 80)
    
    overall_p, overall_r, overall_f1 = compute_metrics(total_tp, total_fp, total_fn)
    
    print(f"\nTotal Files: {len(df)}")
    print(f"TP={total_tp}, FP={total_fp}, FN={total_fn}")
    print(f"Precision = {overall_p:.4f}")
    print(f"Recall    = {overall_r:.4f}")
    print(f"F1-Score  = {overall_f1:.4f}")
    
    # Save detailed per-file CSV
    base = os.path.splitext(csv_path)[0]
    detail_path = base + "_detail.csv"
    pd.DataFrame(per_file_rows).to_csv(detail_path, index=False)
    print(f"\nPer-file details saved to: {detail_path}")
    
    # Save summary CSV (size x noise)
    summary_path = base + "_metrics.csv"
    summary_rows = []
    for size in sizes:
        for noise in noises:
            key = (size, noise)
            if key not in results_by_size_noise:
                continue
            stats = results_by_size_noise[key]
            p, r, f1 = compute_metrics(stats["tp"], stats["fp"], stats["fn"])
            summary_rows.append({
                "Dataset_Size": size,
                "Noise": noise,
                "Num_Files": stats["count"],
                "TP": stats["tp"],
                "FP": stats["fp"],
                "FN": stats["fn"],
                "Precision": round(p, 4),
                "Recall": round(r, 4),
                "F1_Score": round(f1, 4),
            })
    # Add per-size totals
    for size in sizes:
        stats = results_by_size[size]
        p, r, f1 = compute_metrics(stats["tp"], stats["fp"], stats["fn"])
        summary_rows.append({
            "Dataset_Size": size, "Noise": "ALL",
            "Num_Files": stats["count"],
            "TP": stats["tp"], "FP": stats["fp"], "FN": stats["fn"],
            "Precision": round(p, 4), "Recall": round(r, 4), "F1_Score": round(f1, 4),
        })
    # Add overall row
    summary_rows.append({
        "Dataset_Size": "OVERALL", "Noise": "ALL",
        "Num_Files": len(df),
        "TP": total_tp, "FP": total_fp, "FN": total_fn,
        "Precision": round(overall_p, 4), "Recall": round(overall_r, 4),
        "F1_Score": round(overall_f1, 4),
    })
    
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()