"""Build summary table: overall micro-F1 per (strategy, pen_scale, min_effect_size)."""
import pandas as pd

df = pd.read_csv(
    r"C:\Users\samira\OneDrive - GJU\Desktop\PhD Progress -Submissions\Dougakn\Concept Drift\CVDriftPipeline_v2\tune_results_ostovar.csv",
    skipinitialspace=True,
)

# Strip column names (may have trailing spaces)
df.columns = [c.strip() for c in df.columns]

# Keep only data rows (non-empty Algorithm)
df = df[df["Algorithm"].notna() & (df["Algorithm"].str.strip() != "")].copy()
df["Algorithm"] = df["Algorithm"].str.strip()
df["strategy"] = df["strategy"].str.strip()
df["pen_scale"] = pd.to_numeric(df["pen_scale"], errors="coerce")
df["min_effect_size"] = pd.to_numeric(df["min_effect_size"], errors="coerce")
df["total_TP"] = pd.to_numeric(df["total_TP"], errors="coerce")
df["total_FP"] = pd.to_numeric(df["total_FP"], errors="coerce")
df["total_FN"] = pd.to_numeric(df["total_FN"], errors="coerce")
df["macro_F1"] = pd.to_numeric(df["macro_F1"], errors="coerce")

# Number of logs per dataset (for weighted macro F1)
N_LOGS = {"Bose": 1, "Ceravolo": 75, "Ostovar": 75}  # total = 151

# Group by (strategy, pen_scale, min_effect_size) and aggregate across datasets
grouped = df.groupby(["strategy", "pen_scale", "min_effect_size"]).agg(
    total_TP=("total_TP", "sum"),
    total_FP=("total_FP", "sum"),
    total_FN=("total_FN", "sum"),
).reset_index()

# Compute weighted macro F1: weight each dataset's macro_F1 by its number of logs
def weighted_macro_f1(sub):
    total_logs = 0
    weighted_sum = 0.0
    for _, row in sub.iterrows():
        algo = row["Algorithm"]
        n = N_LOGS.get(algo, 1)
        weighted_sum += n * row["macro_F1"]
        total_logs += n
    return weighted_sum / total_logs if total_logs > 0 else 0.0

wmf1 = df.groupby(["strategy", "pen_scale", "min_effect_size"]).apply(
    weighted_macro_f1, include_groups=False
).reset_index(name="weighted_macro_F1")

grouped = grouped.merge(wmf1, on=["strategy", "pen_scale", "min_effect_size"])

# Compute overall micro P, R, F1
grouped["micro_P"] = grouped["total_TP"] / (grouped["total_TP"] + grouped["total_FP"])
grouped["micro_R"] = grouped["total_TP"] / (grouped["total_TP"] + grouped["total_FN"])
grouped["micro_F1"] = 2 * grouped["micro_P"] * grouped["micro_R"] / (grouped["micro_P"] + grouped["micro_R"])
grouped["micro_F1"] = grouped["micro_F1"].fillna(0)

# Sort by micro_F1 descending
grouped = grouped.sort_values("micro_F1", ascending=False).reset_index(drop=True)

# Round for display
for c in ["micro_P", "micro_R", "micro_F1", "weighted_macro_F1"]:
    grouped[c] = grouped[c].round(4)

print("=" * 90)
print("  OVERALL SUMMARY — All 3 datasets combined (Bose + Ceravolo-1000 + Ostovar)")
print("=" * 90)
cols = ["strategy", "pen_scale", "min_effect_size",
        "total_TP", "total_FP", "total_FN",
        "micro_P", "micro_R", "micro_F1", "weighted_macro_F1"]
print(grouped[cols].to_string(index=False))

# Also save to CSV
out = r"C:\Users\samira\OneDrive - GJU\Desktop\PhD Progress -Submissions\Dougakn\Concept Drift\CVDriftPipeline_v2\tune_summary_overall.csv"
grouped[cols].to_csv(out, index=False)
print(f"\nSaved to: {out}")
