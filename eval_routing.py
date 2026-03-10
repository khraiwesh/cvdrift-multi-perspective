"""Quick P/R/F1 evaluation for routing detection results CSV"""
import pandas as pd, json, re, sys


def get_size(name):
    m = re.search(r"_(\d+)_[A-Za-z]+\.xes$", name)
    return int(m.group(1)) if m else None


def get_noise(name):
    m = re.search(r"noise(\d+)", name)
    return int(m.group(1)) if m else None


def parse_cps(val):
    try:
        return json.loads(val)
    except Exception:
        return []


def run_routing_evaluation(csv_path: str, eval_out: str = None) -> dict:
    """
    Evaluate routing detection results CSV.
    GT = size/2, tol = size*0.10  (Ceravolo convention).
    Can be called programmatically from run_combined.py.
    Returns aggregate dict with TP, FP, FN, Precision, Recall, F1.
    """
    df = pd.read_csv(csv_path)
    df = df[~df["Log"].astype(str).str.startswith("===")].copy()

    # Auto-detect CP column
    cp_col = None
    for candidate in ["Routing CPs", "Detected Changepoints", "Duration CPs", "Combined CPs"]:
        if candidate in df.columns:
            cp_col = candidate
            break
    if cp_col is None:
        print(f"[Eval] ERROR: cannot find CP column. Available: {df.columns.tolist()}")
        return {}

    df["size"] = df["Log"].apply(get_size)
    df = df.dropna(subset=["size"]).copy()
    df["size"] = df["size"].astype(int)
    df["gt"] = df["size"] / 2
    df["tol"] = df["size"] * 0.10
    df["detected"] = df[cp_col].apply(parse_cps)

    print(f"\n{'=' * 80}")
    print(f"[Routing Eval]  source : {csv_path}")
    print(f"[Routing Eval]  CP col : {cp_col}   GT = size/2   tol = 10%")
    print(f"{'=' * 80}")

    total_tp = total_fp = total_fn = 0
    per_file = []

    for _, row in df.iterrows():
        gt, tol, detected = row["gt"], row["tol"], row["detected"]
        tp_cps = [c for c in detected if abs(c - gt) <= tol]
        fp_cps = [c for c in detected if abs(c - gt) > tol]
        tp = min(1, len(tp_cps))
        fp = len(fp_cps) + max(0, len(tp_cps) - 1)
        fn = 1 - tp
        total_tp += tp; total_fp += fp; total_fn += fn
        tag = "\u2713" if fn == 0 else "\u2717"
        print(f"  {tag} {str(row['Log'])[:55]:55s}  Det={str(detected):25s} GT={int(gt)}  {'OK' if fn==0 else 'FN=1'}")
        per_file.append(dict(Log=row["Log"], Size=int(row["size"]),
                             GT=int(gt), Tol=int(tol), Detected=detected,
                             TP=tp, FP=fp, FN=fn))

    P = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0
    R = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0
    F = 2*P*R/(P+R) if (P+R) else 0

    print(f"\n{'=' * 80}")
    print(f"  AGGREGATE over {len(per_file)} files:")
    print(f"CSV: {csv_path}  |  Total files: {len(df)}")
    print(f"TP={total_tp}  FP={total_fp}  FN={total_fn}")
    print(f"Precision = {P:.4f}  ({P*100:.2f}%)")
    print(f"Recall    = {R:.4f}  ({R*100:.2f}%)")
    print(f"F1-score  = {F:.4f}  ({F*100:.2f}%)")
    print(f"{'=' * 80}")

    # --- Breakdown by dataset size ---
    print("\n=== Breakdown by dataset size ===")
    for sz in sorted(df["size"].unique()):
        sub = [r for r in per_file if r["Size"] == sz]
        tp_s = sum(r["TP"] for r in sub)
        fp_s = sum(r["FP"] for r in sub)
        fn_s = sum(r["FN"] for r in sub)
        p_s = tp_s/(tp_s+fp_s) if (tp_s+fp_s) else 0
        r_s = tp_s/(tp_s+fn_s) if (tp_s+fn_s) else 0
        f_s = 2*p_s*r_s/(p_s+r_s) if (p_s+r_s) else 0
        print(f"  Size={sz:5d} | Files={len(sub):3d} | TP={tp_s:3d} FP={fp_s:3d} FN={fn_s:3d} | P={p_s:.4f} R={r_s:.4f} F1={f_s:.4f}")

    # --- Breakdown by noise level ---
    print("\n=== Breakdown by noise level ===")
    noise_vals = sorted(set(get_noise(r["Log"]) for r in per_file if get_noise(r["Log"]) is not None))
    for nz in noise_vals:
        sub = [r for r in per_file if get_noise(r["Log"]) == nz]
        tp_s = sum(r["TP"] for r in sub)
        fp_s = sum(r["FP"] for r in sub)
        fn_s = sum(r["FN"] for r in sub)
        p_s = tp_s/(tp_s+fp_s) if (tp_s+fp_s) else 0
        r_s = tp_s/(tp_s+fn_s) if (tp_s+fn_s) else 0
        f_s = 2*p_s*r_s/(p_s+r_s) if (p_s+r_s) else 0
        print(f"  Noise={nz:3d}% | Files={len(sub):3d} | TP={tp_s:3d} FP={fp_s:3d} FN={fn_s:3d} | P={p_s:.4f} R={r_s:.4f} F1={f_s:.4f}")

    # --- Missed files ---
    missed = [r for r in per_file if r["FN"] == 1]
    print(f"\n=== Missed files (FN=1): {len(missed)} ===")
    for r in missed:
        print(f"  {r['Log']}  GT={r['GT']}  Detected={r['Detected']}")

    if eval_out:
        import os
        detail_df = pd.DataFrame(per_file)
        extra_rows = []

        # Aggregate row
        extra_rows.append({
            "Log": "=== AGGREGATE ===", "Size": f"{len(per_file)} files",
            "GT": "-", "Tol": "-", "Detected": "-",
            "TP": total_tp, "FP": total_fp, "FN": total_fn,
            "Precision": round(P, 4), "Recall": round(R, 4), "F1": round(F, 4),
        })

        # Separator
        extra_rows.append({"Log": "", "Size": "", "GT": "", "Tol": "", "Detected": "",
                           "TP": "", "FP": "", "FN": "", "Precision": "", "Recall": "", "F1": ""})

        # Breakdown by size
        extra_rows.append({"Log": "=== BREAKDOWN BY SIZE ===", "Size": "", "GT": "", "Tol": "",
                           "Detected": "", "TP": "", "FP": "", "FN": "", "Precision": "", "Recall": "", "F1": ""})
        for sz in sorted(df["size"].unique()):
            sub = [r for r in per_file if r["Size"] == sz]
            tp_s = sum(r["TP"] for r in sub)
            fp_s = sum(r["FP"] for r in sub)
            fn_s = sum(r["FN"] for r in sub)
            p_s = tp_s/(tp_s+fp_s) if (tp_s+fp_s) else 0
            r_s = tp_s/(tp_s+fn_s) if (tp_s+fn_s) else 0
            f_s = 2*p_s*r_s/(p_s+r_s) if (p_s+r_s) else 0
            extra_rows.append({"Log": f"Size={sz}", "Size": len(sub), "GT": "-", "Tol": "-",
                               "Detected": "-", "TP": tp_s, "FP": fp_s, "FN": fn_s,
                               "Precision": round(p_s, 4), "Recall": round(r_s, 4), "F1": round(f_s, 4)})

        # Separator
        extra_rows.append({"Log": "", "Size": "", "GT": "", "Tol": "", "Detected": "",
                           "TP": "", "FP": "", "FN": "", "Precision": "", "Recall": "", "F1": ""})

        # Breakdown by noise
        extra_rows.append({"Log": "=== BREAKDOWN BY NOISE ===", "Size": "", "GT": "", "Tol": "",
                           "Detected": "", "TP": "", "FP": "", "FN": "", "Precision": "", "Recall": "", "F1": ""})
        noise_vals2 = sorted(set(get_noise(r["Log"]) for r in per_file if get_noise(r["Log"]) is not None))
        for nz in noise_vals2:
            sub = [r for r in per_file if get_noise(r["Log"]) == nz]
            tp_s = sum(r["TP"] for r in sub)
            fp_s = sum(r["FP"] for r in sub)
            fn_s = sum(r["FN"] for r in sub)
            p_s = tp_s/(tp_s+fp_s) if (tp_s+fp_s) else 0
            r_s = tp_s/(tp_s+fn_s) if (tp_s+fn_s) else 0
            f_s = 2*p_s*r_s/(p_s+r_s) if (p_s+r_s) else 0
            extra_rows.append({"Log": f"Noise={nz}%", "Size": len(sub), "GT": "-", "Tol": "-",
                               "Detected": "-", "TP": tp_s, "FP": fp_s, "FN": fn_s,
                               "Precision": round(p_s, 4), "Recall": round(r_s, 4), "F1": round(f_s, 4)})

        pd.concat([detail_df, pd.DataFrame(extra_rows)], ignore_index=True).to_csv(eval_out, index=False)
        print(f"  Detail saved to: {eval_out}")

    return {
        "TP": total_tp, "FP": total_fp, "FN": total_fn,
        "Precision": round(P, 4), "Recall": round(R, 4), "F1": round(F, 4),
    }


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "Routingresults_routing.csv"
    run_routing_evaluation(csv_path)
