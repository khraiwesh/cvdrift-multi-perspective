"""Compute P/R/F1 for MDD results.csv — GT=37%/75%, tol=10%."""
import pandas as pd
import re, ast

df = pd.read_csv("results.csv")

def get_size(n):
    m = re.search(r"(\d+)cases", n)
    return int(m.group(1)) if m else None

def get_noise(n):
    if "noisy" not in n.lower():
        return "clean"
    m = re.search(r"noisy[_ ]?(\d+)%", n)
    return f"{m.group(1)}%" if m else "noisy"

def parse_cps(v):
    if isinstance(v, str):
        if v.strip() == "ERROR":
            return []
        try:
            return ast.literal_eval(v)
        except Exception:
            return []
    return []

def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    return p, r, f1

def file_metrics(det, sz):
    gt = [int(round(0.37 * sz)), int(round(0.75 * sz))]
    tol = int(round(0.10 * sz))
    tp = 0
    matched = set()
    for g in gt:
        for i, d in enumerate(det):
            if abs(d - g) <= tol and i not in matched:
                tp += 1
                matched.add(i)
                break
    return tp, len(det) - len(matched), len(gt) - tp

df["log_name"] = df["Log"].str.strip()
df["size"] = df["log_name"].apply(get_size)
df["noise"] = df["log_name"].apply(get_noise)
df["det"] = df["Detected Changepoints"].apply(parse_cps)
df["category"] = df["noise"].apply(lambda x: "Clean" if x == "clean" else "Noisy")

tps, fps, fns = [], [], []
for _, r in df.iterrows():
    tp, fp, fn = file_metrics(r["det"], r["size"])
    tps.append(tp)
    fps.append(fp)
    fns.append(fn)
df["TP"] = tps
df["FP"] = fps
df["FN"] = fns

def agg(sub):
    tp = int(sub["TP"].sum())
    fp = int(sub["FP"].sum())
    fn = int(sub["FN"].sum())
    p, r, f1 = prf(tp, fp, fn)
    return tp, fp, fn, p, r, f1

errors = df[df["Detected Changepoints"].str.strip() == "ERROR"]
print(f"Total files: {len(df)}")
print(f"Errors: {len(errors)}")
print()

tp, fp, fn, p, r, f1 = agg(df)
print(f"OVERALL:  P={p:.3f}  R={r:.3f}  F1={f1:.3f}  (TP={tp} FP={fp} FN={fn})")
print()

for cat in ["Clean", "Noisy"]:
    s = df[df["category"] == cat]
    tp2, fp2, fn2, p2, r2, f12 = agg(s)
    print(f"{cat:6s} ({len(s):2d}): P={p2:.3f}  R={r2:.3f}  F1={f12:.3f}  (TP={tp2} FP={fp2} FN={fn2})")
print()

for sz in sorted(df["size"].unique()):
    s = df[df["size"] == sz]
    tp2, fp2, fn2, p2, r2, f12 = agg(s)
    print(f"Size {sz:>5d} ({len(s):2d}): P={p2:.3f}  R={r2:.3f}  F1={f12:.3f}")
print()

for sz in sorted(df["size"].unique()):
    for cat in ["Clean", "Noisy"]:
        s = df[(df["size"] == sz) & (df["category"] == cat)]
        if len(s) == 0:
            continue
        tp2, fp2, fn2, p2, r2, f12 = agg(s)
        print(f"{sz:>5d} {cat:6s} ({len(s):2d}): P={p2:.3f}  R={r2:.3f}  F1={f12:.3f}")
    print()

print("--- PER-FILE DETAIL ---")
for _, r in df.sort_values(["size", "noise", "log_name"]).iterrows():
    short = r["log_name"].replace("dataset_", "").replace("_ABD", "").replace(".xes", "")
    cp_str = str(r["det"]) if r["det"] else "[]"
    sz = r["size"]
    gt = [int(round(0.37 * sz)), int(round(0.75 * sz))]
    tp2, fp2, fn2 = r["TP"], r["FP"], r["FN"]
    p2, r2, f12 = prf(tp2, fp2, fn2)
    print(f"  {short:<42s} GT={gt}  det={cp_str:<30s} P={p2:.3f} R={r2:.3f} F1={f12:.3f}")
