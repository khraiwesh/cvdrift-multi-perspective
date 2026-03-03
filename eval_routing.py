"""Quick P/R/F1 evaluation for Routingresults_routing.csv"""
import pandas as pd, json, re, sys

csv_path = sys.argv[1] if len(sys.argv) > 1 else "Routingresults_routing.csv"
df = pd.read_csv(csv_path)
df = df[~df["Log"].str.startswith("===")].copy()

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

df["size"] = df["Log"].apply(get_size)
df = df.dropna(subset=["size"])
df["size"] = df["size"].astype(int)
df["gt"] = df["size"] / 2
df["tol"] = df["size"] * 0.10
df["detected"] = df["Routing CPs"].apply(parse_cps)

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
    per_file.append(dict(Log=row["Log"], Size=int(row["size"]),
                         GT=int(gt), Tol=int(tol), Detected=detected,
                         TP=tp, FP=fp, FN=fn))

P = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0
R = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0
F = 2*P*R/(P+R) if (P+R) else 0

print(f"CSV: {csv_path}  |  Total files: {len(df)}")
print(f"TP={total_tp}  FP={total_fp}  FN={total_fn}")
print(f"Precision = {P:.4f}  ({P*100:.2f}%)")
print(f"Recall    = {R:.4f}  ({R*100:.2f}%)")
print(f"F1-score  = {F:.4f}  ({F*100:.2f}%)")

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
