"""Generate final tuning report from the 3 dataset CSV results."""
import pandas as pd

bose = pd.read_csv("tune_results_bose.csv")
cer  = pd.read_csv("tune_results_ceravolo1000.csv")
ost  = pd.read_csv("tune_results_ostovar.csv")

print("=" * 80)
print("  BOSE — Top 5")
print("=" * 80)
print(bose.head(5).to_string(index=False))

print()
print("=" * 80)
print("  CERAVOLO-1000 — Top 5")
print("=" * 80)
print(cer.head(5).to_string(index=False))

print()
print("=" * 80)
print("  OSTOVAR — Top 5")
print("=" * 80)
print(ost.head(5).to_string(index=False))

# ── Best universal config (same params for all 3 datasets) ──
rows = []
for _, rb in bose.iterrows():
    s, ps, mes = rb["strategy"], rb["pen_scale"], rb["min_effect_size"]
    rc = cer[(cer["strategy"] == s) & (cer["pen_scale"] == ps) & (cer["min_effect_size"] == mes)]
    ro = ost[(ost["strategy"] == s) & (ost["pen_scale"] == ps) & (ost["min_effect_size"] == mes)]
    if len(rc) == 0 or len(ro) == 0:
        continue
    tp = int(rb["total_TP"]) + int(rc.iloc[0]["total_TP"]) + int(ro.iloc[0]["total_TP"])
    fp = int(rb["total_FP"]) + int(rc.iloc[0]["total_FP"]) + int(ro.iloc[0]["total_FP"])
    fn = int(rb["total_FN"]) + int(rc.iloc[0]["total_FN"]) + int(ro.iloc[0]["total_FN"])
    P = tp / (tp + fp) if (tp + fp) else 0
    R = tp / (tp + fn) if (tp + fn) else 0
    F1 = 2 * P * R / (P + R) if (P + R) else 0
    rows.append({
        "strategy": s, "pen_scale": ps, "min_effect_size": mes,
        "TP": tp, "FP": fp, "FN": fn,
        "P": round(P, 4), "R": round(R, 4), "F1": round(F1, 4),
        "bose_F1": round(float(rb["micro_F1"]), 4),
        "cer_F1": round(float(rc.iloc[0]["micro_F1"]), 4),
        "ost_F1": round(float(ro.iloc[0]["micro_F1"]), 4),
    })

top_univ = pd.DataFrame(rows).sort_values("F1", ascending=False).reset_index(drop=True)
best = top_univ.iloc[0]

print()
print("=" * 80)
print("  BEST UNIVERSAL CONFIG (same params for all 3 datasets)")
print("=" * 80)
print(f"  Strategy:        {best['strategy']}")
print(f"  pen_scale:       {best['pen_scale']}")
print(f"  min_effect_size: {best['min_effect_size']}")
print(f"  Overall: TP={best['TP']}  FP={best['FP']}  FN={best['FN']}  "
      f"P={best['P']}  R={best['R']}  F1={best['F1']}")
print(f"  Per-dataset: Bose={best['bose_F1']}, Ceravolo={best['cer_F1']}, "
      f"Ostovar={best['ost_F1']}")

print()
print("  Top 10 universal configs:")
print("  " + "-" * 78)
print(top_univ.head(10).to_string(index=False))

# ── Comparison vs baseline ──
print()
print("=" * 80)
print("  COMPARISON: Baseline vs Best Tuned")
print("=" * 80)
print("  Baseline (pen_scale=5.0, min_effect_size=0.15, cv_perpair):")
print("    Ceravolo F1=0.7815, Ostovar F1=0.8591, Bose F1=0.7500")
print("    Overall: TP=190, FP=38, FN=39, P=0.8333, R=0.8297, F1=0.8315")

delta = round(best["F1"] - 0.8315, 4)
sign = "+" if delta >= 0 else ""
print()
print(f"  Best universal tuned ({best['strategy']}, ps={best['pen_scale']}, "
      f"mes={best['min_effect_size']}):")
print(f"    Ceravolo F1={best['cer_F1']}, Ostovar F1={best['ost_F1']}, "
      f"Bose F1={best['bose_F1']}")
print(f"    Overall: TP={best['TP']}, FP={best['FP']}, FN={best['FN']}, "
      f"P={best['P']}, R={best['R']}, F1={best['F1']}")
print(f"    Delta F1: {sign}{delta}")

# ── Best per-dataset (oracle) ──
b_best = bose.iloc[0]
c_best = cer.iloc[0]
o_best = ost.iloc[0]
tp_pd = int(b_best["total_TP"]) + int(c_best["total_TP"]) + int(o_best["total_TP"])
fp_pd = int(b_best["total_FP"]) + int(c_best["total_FP"]) + int(o_best["total_FP"])
fn_pd = int(b_best["total_FN"]) + int(c_best["total_FN"]) + int(o_best["total_FN"])
P_pd = tp_pd / (tp_pd + fp_pd) if (tp_pd + fp_pd) else 0
R_pd = tp_pd / (tp_pd + fn_pd) if (tp_pd + fn_pd) else 0
F1_pd = 2 * P_pd * R_pd / (P_pd + R_pd) if (P_pd + R_pd) else 0

print()
print("  Best PER-DATASET configs (oracle upper bound):")
print(f"    Bose:     {b_best['strategy']}, ps={b_best['pen_scale']}, "
      f"mes={b_best['min_effect_size']} -> F1={b_best['micro_F1']}")
print(f"    Ceravolo: {c_best['strategy']}, ps={c_best['pen_scale']}, "
      f"mes={c_best['min_effect_size']} -> F1={c_best['micro_F1']}")
print(f"    Ostovar:  {o_best['strategy']}, ps={o_best['pen_scale']}, "
      f"mes={o_best['min_effect_size']} -> F1={o_best['micro_F1']}")
print(f"    Combined: TP={tp_pd}, FP={fp_pd}, FN={fn_pd}, "
      f"P={P_pd:.4f}, R={R_pd:.4f}, F1={F1_pd:.4f}")
