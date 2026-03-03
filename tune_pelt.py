"""
tune_pelt.py — Grid-search over pen_scale and min_effect_size
=============================================================

Caches the expensive preprocessing + window-selection + rolling-series
building *once* per (file, window_strategy).  Then sweeps PELT penalty
and effect-size threshold quickly on the cached series.

**Optimisations**
* Log is parsed only once per file (shared across strategies).
* CV window selection runs once; mode_window is derived via deep-copy.
* CachedSeries accepts pre-parsed logs — no redundant parsing.

Usage
-----
  python tune_pelt.py                       # run all 3 datasets
  python tune_pelt.py --dataset bose        # run only Bose
  python tune_pelt.py --dataset ceravolo    # run only Ceravolo
  python tune_pelt.py --dataset ostovar     # run only Ostovar
"""

import argparse
import copy
import glob
import os
import sys
import time
import warnings
from collections import Counter
from itertools import product
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── project imports ──
from pipeline.io import read_xes_to_dataframe
from pipeline.preprocessing import prepare_event_log_dual, prepare_seq_log
from pipeline.series_duration import series_duration_case_indexed
from pipeline.series_routing import (
    add_next_act,
    series_routing_case_indexed,
)
from pipeline.series_arrival import series_arrival_case_indexed
from pipeline.rolling import window_stat_series
from pipeline.drift_detection import detect_drift_pelt, consolidate_changepoints
from pipeline.consensus import compute_routing_consensus, compute_duration_consensus

from preparation import preparation, preprocess, DEFAULT_PARAMS
from main import select_window, _apply_mode_window, _load_log

warnings.filterwarnings("ignore")

# ====================================================================
# Configuration
# ====================================================================

# Parameter grid  (reduced for speed — 4×4×2 = 32 combos)
PEN_SCALES       = [2.0, 3.0, 5.0, 7.0]
MIN_EFFECT_SIZES = [0.0, 0.10, 0.15, 0.30]
STRATEGIES       = ["cv_perpair", "mode_window"]

# Dataset definitions
BASE_EVAL = (
    r"C:\Users\samira\OneDrive - GJU\Desktop\PhD Progress -Submissions"
    r"\Dougakn\Concept Drift\Experiment_Setup_Concept_Drift"
    r"\cdrift-evaluation\EvaluationLogs"
)

DATASETS = {
    "bose": {
        "files": [os.path.join(BASE_EVAL, "Bose", "bose_log.xes", "bose_log.xes")],
        "drift_type": "routing",
        "gt_func": lambda fname, n: [1199, 2399, 3599, 4799],
        "tol_func": lambda fname, n: 600,
        "label": "Bose (1 log, 6000 traces)",
    },
    "ceravolo": {
        "dir": os.path.join(BASE_EVAL, "Ceravolo", "All"),
        "drift_type": "routing",
        "gt_func": lambda fname, n: [n // 2],
        "tol_func": lambda fname, n: max(1, n // 10),
        "label": "Ceravolo (135 logs)",
        "filter_size": None,   # None = all files; set 1000 for fair comparison
    },
    "ceravolo1000": {
        "dir": os.path.join(BASE_EVAL, "Ceravolo", "All"),
        "drift_type": "routing",
        "gt_func": lambda fname, n: [n // 2],
        "tol_func": lambda fname, n: max(1, n // 10),
        "label": "Ceravolo size=1000 (75 logs)",
        "filter_size": 1000,
    },
    "ostovar": {
        "dir": os.path.join(BASE_EVAL, "Ostovar", "All", "Draft"),
        "drift_type": "routing",
        "gt_func": lambda fname, n: [999, 1999],
        "tol_func": lambda fname, n: 300,
        "label": "Ostovar (75 logs)",
    },
    # ── Duration drift datasets ───────────────────────────────
    "selected": {
        "dir": (r"C:\Users\samira\OneDrive - GJU\Desktop\PhD Progress "
                r"-Submissions\Dougakn\Concept Drift\Datasets\Logs\Selected"),
        "drift_type": "duration",
        "gt_func": lambda fname, n: [round(0.37 * n), round(0.75 * n)],
        "tol_func": lambda fname, n: max(1, n // 10),
        "label": "Selected — duration drift (68 logs, GT@37%+75%)",
    },
}


# ====================================================================
# Evaluation: greedy matching
# ====================================================================

def evaluate_cps(detected: List[int], ground_truth: List[int],
                 tolerance: int) -> Dict[str, Any]:
    """Greedy matching — each GT matched to at most one detected CP."""
    det = sorted(detected)
    gt  = sorted(ground_truth)
    matched_det = set()
    tp = 0

    for g in gt:
        best_idx, best_dist = -1, float("inf")
        for i, d in enumerate(det):
            if i in matched_det:
                continue
            dist = abs(d - g)
            if dist <= tolerance and dist < best_dist:
                best_idx, best_dist = i, dist
        if best_idx >= 0:
            matched_det.add(best_idx)
            tp += 1

    fp = len(det) - tp
    fn = len(gt)  - tp
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"TP": tp, "FP": fp, "FN": fn, "P": p, "R": r, "F1": f1}


# ====================================================================
# Cached series builder  (accepts pre-parsed logs — no redundant I/O)
# ====================================================================

class CachedSeries:
    """Precomputes rolling series + raw duration vectors once.

    ``run_pelt(pen_scale, min_effect_size)`` then sweeps only the
    cheap PELT + consensus step.
    """

    def __init__(self, win_sel, logs, elog_seq, seq_with_next, params):
        p = params
        self.n_cases = win_sel["meta"]["n_cases"]
        n_cases = self.n_cases

        case_to_orig = (
            elog_seq.groupby(".case")[".orig_case"].first().to_dict()
            if ".orig_case" in elog_seq.columns else {}
        )
        duration_per_case = p.get("duration_per_case", "median")
        duration_roll_stat = p.get("duration_stat", "median")
        routing_roll_stat  = p.get("routing_stat", "mean")
        arrival_roll_stat  = p.get("arrival_stat", "median")

        series_list = []

        # ── Duration series (multi-scale) ─────────────────────────
        self.da_map = {}
        for it in win_sel.get("activity_duration", []):
            if not isinstance(it, dict) or "chosen_window" not in it:
                continue
            a = str(it["activity"])
            w = int(it["chosen_window"])
            x, t, cases, orig_cases = series_duration_case_indexed(
                logs.elog_dur, a, n_cases, how=duration_per_case)
            n_valid = int(np.sum(np.isfinite(x)))
            max_w = max(10, n_valid // 10)
            w = min(w, max_w)
            self.da_map[a] = w

            scales = [w]
            w_half = w // 2
            if w_half >= 5 and n_valid >= w_half * 2:
                scales.append(w_half)
            w_quarter = w // 4
            if w_quarter >= 5 and n_valid >= w_quarter * 2:
                scales.append(w_quarter)

            for sw in scales:
                auto_step = max(1, sw // 10)
                ts = window_stat_series(
                    x, t, w=sw, step=auto_step,
                    stat=duration_roll_stat, cases=cases, orig_cases=orig_cases)
                if len(ts) > 0:
                    tag = "" if sw == w else f"[w{sw}]"
                    ts["param"] = f"duration::{a}{tag}"
                    ts["base_param"] = f"duration::{a}"
                    series_list.append(ts)

        # ── Routing series (multi-scale) ──────────────────────────
        self.rp_map = {}
        for it in win_sel.get("routing_probability", []):
            if not isinstance(it, dict) or "chosen_window" not in it:
                continue
            from_act = str(it["from"])
            to_act   = str(it["to"])
            w = int(it["chosen_window"])
            key = f"{from_act}->{to_act}"
            self.rp_map[key] = w

            x, t, cases, orig_cases = series_routing_case_indexed(
                seq_with_next, from_act, to_act, n_cases, case_to_orig)
            n_raw = int(np.sum(np.isfinite(x)))

            scales = [w]
            w_half = w // 2
            if w_half >= 10 and n_raw >= w_half * 2:
                scales.append(w_half)

            for sw in scales:
                auto_step = max(1, sw // 10)
                ts = window_stat_series(
                    x, t, w=sw, step=auto_step,
                    stat=routing_roll_stat, cases=cases, orig_cases=orig_cases)
                if len(ts) > 0:
                    tag = "" if sw == w else f"[w{sw}]"
                    ts["param"] = f"routing::{key}{tag}"
                    ts["base_param"] = f"routing::{key}"
                    series_list.append(ts)

        # ── Arrival series (multi-scale) ──────────────────────────
        self.arrival_w_map = {}
        for it in win_sel.get("arrival_time", []):
            if not isinstance(it, dict) or "chosen_window" not in it:
                continue
            w = int(it["chosen_window"])
            self.arrival_w_map["arrival"] = w
            meta = win_sel.get("meta", {})
            arr_max_gap = float(meta.get("arrival_max_gap_hours", 4.0))
            arr_same_day = bool(meta.get("arrival_same_day_only", True))

            x, t, cases, orig_cases = series_arrival_case_indexed(
                elog_seq, n_cases,
                max_gap_hours=arr_max_gap,
                same_day_only=arr_same_day,
                case_to_orig=case_to_orig,
            )
            n_raw = int(np.sum(np.isfinite(x)))
            if n_raw < 10:
                continue
            max_w = max(10, n_raw // 10)
            w = min(w, max_w)
            self.arrival_w_map["arrival"] = w

            scales = [w]
            w_half = w // 2
            if w_half >= 5 and n_raw >= w_half * 2:
                scales.append(w_half)
            w_quarter = w // 4
            if w_quarter >= 5 and n_raw >= w_quarter * 2:
                scales.append(w_quarter)

            for sw in scales:
                auto_step = max(1, sw // 10)
                ts = window_stat_series(
                    x, t, w=sw, step=auto_step,
                    stat=arrival_roll_stat, cases=cases, orig_cases=orig_cases)
                if len(ts) > 0:
                    tag = "" if sw == w else f"[w{sw}]"
                    ts["param"] = f"arrival::inter_arrival{tag}"
                    ts["base_param"] = "arrival::inter_arrival"
                    series_list.append(ts)

        if not series_list:
            self.ts_df = pd.DataFrame()
            self.has_series = False
            self.raw_duration_cache = []
            return

        self.ts_df = pd.concat(series_list, ignore_index=True)
        self.ts_df["value"] = pd.to_numeric(self.ts_df["value"], errors="coerce")
        self.ts_df = self.ts_df[np.isfinite(self.ts_df["value"])].copy()
        self.ts_df = self.ts_df.sort_values(
            ["param", "win_id"], kind="mergesort").reset_index(drop=True)
        self.has_series = True

        # ── Cache raw duration vectors for raw-PELT pass ─────────
        self.raw_duration_cache = []
        for it in win_sel.get("activity_duration", []):
            if not isinstance(it, dict) or "chosen_window" not in it:
                continue
            a = str(it["activity"])
            x_raw, t_raw, cases_raw, orig_raw = series_duration_case_indexed(
                logs.elog_dur, a, n_cases, how=duration_per_case)
            n_valid_raw = int(np.sum(np.isfinite(x_raw)))
            if n_valid_raw < 20:
                continue
            x_finite = x_raw[np.isfinite(x_raw)]
            valid_idx = np.where(np.isfinite(x_raw))[0]
            self.raw_duration_cache.append({
                "activity": a,
                "x_finite": x_finite,
                "valid_idx": valid_idx,
                "x_raw": x_raw,
                "t_raw": t_raw,
                "orig_raw": orig_raw,
                "n_valid_raw": n_valid_raw,
            })

    # ────────────────────────────────────────────────────────────
    def run_pelt(self, pen_scale: float, min_effect_size: float,
                 cpd_model: str = "l2",
                 min_cp_distance: int = 10,
                 min_n_points: int = 10) -> Dict[str, Any]:
        """Run PELT + effect-size filter + consensus on cached series."""

        if not self.has_series:
            return {"consensus_cps": [], "duration_consensus_cps": [],
                    "all_cps": []}

        n_cases = self.n_cases
        ts_df = self.ts_df

        drift_rows = []

        for param, g in ts_df.groupby("param", sort=False):
            values = g["value"].to_numpy(dtype=float)
            if len(values) < int(min_n_points):
                continue

            cps = detect_drift_pelt(values, pen_scale=pen_scale, model=cpd_model)
            cps = consolidate_changepoints(
                cps, min_distance=int(min_cp_distance), values=values)

            # Effect-size filter (raw mean difference)
            if min_effect_size > 0 and cps:
                kept = []
                for cp in cps:
                    seg_b = values[:cp]
                    seg_a = values[cp:]
                    if len(seg_b) >= 2 and len(seg_a) >= 2:
                        md = abs(float(np.mean(seg_b)) - float(np.mean(seg_a)))
                        if md >= float(min_effect_size):
                            kept.append(cp)
                    else:
                        kept.append(cp)
                cps = kept

            if cps:
                g_idx = g.reset_index(drop=True)
                base_param_val = (g_idx["base_param"].iloc[0]
                                  if "base_param" in g_idx.columns else param)
                for cp in cps:
                    j = max(1, min(int(cp), len(g_idx))) - 1
                    row = g_idx.iloc[j]
                    drift_rows.append({
                        "param": param,
                        "base_param": base_param_val,
                        "cp_series_index": int(cp),
                        "cp_win_id": int(row["win_id"]),
                        "cp_time": row["t_end"],
                        "cp_value": float(row["value"]),
                        "cp_case": row.get("case_mid", row["case_end"]),
                        "cp_orig_case": row.get("orig_case_mid",
                                                row["orig_case_end"]),
                    })

        # ── Raw PELT on per-case duration values ──────────────────
        for item in self.raw_duration_cache:
            a = item["activity"]
            x_finite = item["x_finite"]
            valid_idx = item["valid_idx"]
            t_raw = item["t_raw"]
            orig_raw = item["orig_raw"]
            x_raw = item["x_raw"]

            raw_cps = detect_drift_pelt(
                x_finite, pen_scale=pen_scale, model=cpd_model)
            raw_cps = consolidate_changepoints(
                raw_cps,
                min_distance=max(3, n_cases // 50),
                values=x_finite,
            )

            # Cohen's d >= 0.3
            if raw_cps:
                overall_std = (float(np.std(x_finite, ddof=1))
                               if len(x_finite) > 1 else 0.0)
                if overall_std > 1e-12:
                    kept = []
                    for rcp in raw_cps:
                        seg_b = x_finite[:rcp]
                        seg_a = x_finite[rcp:]
                        if len(seg_b) >= 2 and len(seg_a) >= 2:
                            d = (abs(float(np.mean(seg_b)) -
                                     float(np.mean(seg_a))) / overall_std)
                            if d >= 0.3:
                                kept.append(rcp)
                        else:
                            kept.append(rcp)
                    raw_cps = kept

            for rcp in raw_cps:
                if rcp >= len(valid_idx):
                    continue
                case_idx = int(valid_idx[rcp])
                drift_rows.append({
                    "param": f"duration_raw::{a}",
                    "base_param": f"duration::{a}",
                    "cp_series_index": int(rcp),
                    "cp_win_id": 0,
                    "cp_time": (t_raw[case_idx]
                                if t_raw is not None and case_idx < len(t_raw)
                                else pd.NaT),
                    "cp_value": (float(x_raw[case_idx])
                                 if case_idx < len(x_raw) else 0.0),
                    "cp_case": str(case_idx),
                    "cp_orig_case": (str(orig_raw[case_idx])
                                     if orig_raw is not None
                                     and case_idx < len(orig_raw)
                                     else str(case_idx)),
                })

        # Build drifts DataFrame
        if drift_rows:
            drifts = pd.DataFrame(drift_rows).sort_values(
                ["param", "cp_win_id"], kind="mergesort").reset_index(drop=True)
        else:
            drifts = pd.DataFrame(
                columns=["param", "base_param", "cp_series_index",
                          "cp_win_id", "cp_time", "cp_value",
                          "cp_case", "cp_orig_case"])

        # ── Consensus ─────────────────────────────────────────────
        routing_cps = []
        consensus = compute_routing_consensus(
            drifts, ts_df, self.rp_map, n_cases)
        if consensus is not None and len(consensus) > 0:
            if "consensus_case" in consensus.columns:
                routing_cps = sorted(int(c)
                                     for c in consensus["consensus_case"])

        duration_cps = []
        dur_cons = compute_duration_consensus(drifts, self.da_map, n_cases)
        if dur_cons is not None and len(dur_cons) > 0:
            if "consensus_case" in dur_cons.columns:
                duration_cps = sorted(int(c)
                                      for c in dur_cons["consensus_case"])

        return {
            "consensus_cps": routing_cps,
            "duration_consensus_cps": duration_cps,
            "all_cps": sorted(set(routing_cps + duration_cps)),
        }


# ====================================================================
# File loading + dataset size
# ====================================================================

def load_xes(path):
    """Load a .xes file → DataFrame, return (df, n_cases)."""
    df = read_xes_to_dataframe(path, include_resource=True)
    n = df["Case ID"].nunique()
    return df, n


# ====================================================================
# Build caches for BOTH strategies at once  (shared parsing!)
# ====================================================================

def build_cache_pair(df, drift_type, params_override=None):
    """Run preparation + CV window selection once, derive both strategies.

    Returns ``{strategy: CachedSeries}`` and ``n_cases``.
    """
    p = dict(DEFAULT_PARAMS)
    p["window_strategy"] = "cv_perpair"       # run CV first
    if params_override:
        p.update(params_override)

    # Step 1-2: preparation + CV window selection (done ONCE)
    preprocessed = preprocess(df, p)
    prep = preparation(df, drift_type, params=p, preprocessed=preprocessed)
    win_sel_cv = select_window(prep)            # cv_perpair
    n_cases = prep["n_cases"]

    # Derive mode_window from cv_perpair (instant — just override)
    win_sel_mode = copy.deepcopy(win_sel_cv)
    _apply_mode_window(win_sel_mode, drift_type)
    win_sel_mode["meta"]["window_strategy"] = "mode_window"

    # Parse the log ONCE (shared by both CachedSeries objects)
    logs = prepare_event_log_dual(
        df, p["CASE_COL"], p["ACT_COL"],
        p["START_COL"], p["END_COL"],
        p.get("RES_COL"), tz=p.get("tz", "UTC"),
    )
    elog_seq = prepare_seq_log(
        df, p["CASE_COL"], p["ACT_COL"],
        p["START_COL"], tz=p.get("tz", "UTC"),
    )
    seq_with_next = add_next_act(elog_seq)

    out = {}
    for strat, ws in [("cv_perpair", win_sel_cv),
                       ("mode_window", win_sel_mode)]:
        try:
            out[strat] = CachedSeries(ws, logs, elog_seq, seq_with_next, p)
        except Exception as exc:
            print(f"    [WARN] CachedSeries({strat}) failed: {exc}")
            out[strat] = None

    return out, n_cases


# ====================================================================
# Tune one dataset
# ====================================================================

def tune_dataset(dataset_key: str, pen_scales=None, effect_sizes=None,
                 strategies=None):
    """
    Run the full grid search for one dataset.

    Returns a DataFrame with columns:
        strategy, pen_scale, min_effect_size,
        total_TP, total_FP, total_FN, micro_P, micro_R, micro_F1, macro_F1
    """
    if pen_scales is None:
        pen_scales = PEN_SCALES
    if effect_sizes is None:
        effect_sizes = MIN_EFFECT_SIZES
    if strategies is None:
        strategies = STRATEGIES

    ds = DATASETS[dataset_key]
    drift_type = ds["drift_type"]
    gt_func    = ds["gt_func"]
    tol_func   = ds["tol_func"]
    filter_size = ds.get("filter_size")

    # Gather file list
    if "files" in ds:
        files = ds["files"]
    else:
        d = ds["dir"]
        files = sorted(glob.glob(os.path.join(d, "*.xes")))
        if not files:
            files = sorted(glob.glob(os.path.join(d, "*.xes.gz")))

    print(f"\n{'='*70}")
    print(f"  TUNING: {ds['label']}  ({len(files)} files found)")
    print(f"{'='*70}")

    # ── Load + filter ─────────────────────────────────────────
    file_data = []
    for fpath in files:
        df, n = load_xes(fpath)
        if filter_size is not None and n != filter_size:
            continue
        file_data.append((fpath, df, n))

    print(f"  Files after filter: {len(file_data)}")

    # ── Build caches (expensive; done ONCE per file for BOTH strategies) ──
    caches = {}   # (file_idx, strategy) → CachedSeries
    print(f"\n  Building caches (both strategies at once) ...")
    t0 = time.time()
    for idx, (fpath, df, n) in enumerate(file_data):
        pair, _ = build_cache_pair(df, drift_type)
        for strat in strategies:
            caches[(idx, strat)] = pair.get(strat)
        if (idx + 1) % 10 == 0 or (idx + 1) == len(file_data):
            elapsed = time.time() - t0
            print(f"    cached {idx+1}/{len(file_data)}  ({elapsed:.0f}s)")

    # ── Grid search ──────────────────────────────────────────────
    combos = list(product(strategies, pen_scales, effect_sizes))
    n_combos = len(combos)
    print(f"\n  Grid: {len(strategies)} strategies × "
          f"{len(pen_scales)} pen_scales × "
          f"{len(effect_sizes)} effect_sizes = {n_combos} combos")

    results = []
    t_grid = time.time()

    for ci, (strat, ps, mes) in enumerate(combos):
        total_tp, total_fp, total_fn = 0, 0, 0
        f1_list = []

        for idx, (fpath, df, n) in enumerate(file_data):
            cache = caches.get((idx, strat))
            if cache is None or not cache.has_series:
                gt = gt_func(os.path.basename(fpath), n)
                total_fn += len(gt)
                f1_list.append(0.0)
                continue

            pelt_res = cache.run_pelt(pen_scale=ps, min_effect_size=mes)

            if drift_type == "routing":
                det_cps = pelt_res["consensus_cps"]
            elif drift_type == "duration":
                det_cps = pelt_res["duration_consensus_cps"]
            else:
                det_cps = pelt_res["all_cps"]

            gt  = gt_func(os.path.basename(fpath), n)
            tol = tol_func(os.path.basename(fpath), n)
            ev  = evaluate_cps(det_cps, gt, tol)
            total_tp += ev["TP"]
            total_fp += ev["FP"]
            total_fn += ev["FN"]
            f1_list.append(ev["F1"])

        p_agg = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
        r_agg = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
        f1_agg = 2 * p_agg * r_agg / (p_agg + r_agg) if (p_agg + r_agg) else 0.0
        macro_f1 = float(np.mean(f1_list)) if f1_list else 0.0

        results.append({
            "strategy": strat,
            "pen_scale": ps,
            "min_effect_size": mes,
            "total_TP": total_tp,
            "total_FP": total_fp,
            "total_FN": total_fn,
            "micro_P": round(p_agg, 4),
            "micro_R": round(r_agg, 4),
            "micro_F1": round(f1_agg, 4),
            "macro_F1": round(macro_f1, 4),
        })

        if (ci + 1) % 20 == 0 or (ci + 1) == n_combos:
            elapsed = time.time() - t_grid
            print(f"    combo {ci+1}/{n_combos}  ({elapsed:.0f}s)")

    res_df = pd.DataFrame(results)
    res_df = res_df.sort_values("micro_F1", ascending=False).reset_index(drop=True)

    # ── Save results ─────────────────────────────────────────────
    out_path = os.path.join(
        os.path.dirname(__file__),
        f"tune_results_{dataset_key}.csv",
    )
    res_df.to_csv(out_path, index=False)
    print(f"\n  Results saved -> {out_path}")

    # ── Print top 10 ─────────────────────────────────────────────
    print(f"\n  {'='*60}")
    print(f"  TOP 10 configs for {ds['label']}")
    print(f"  {'='*60}")
    print(res_df.head(10).to_string(index=False))

    return res_df


# ====================================================================
# Main
# ====================================================================

def main():
    parser = argparse.ArgumentParser(description="Tune PELT parameters")
    parser.add_argument("--dataset", nargs="*",
                        choices=list(DATASETS.keys()),
                        default=None,
                        help="Which dataset(s) to tune. Default: all.")
    args = parser.parse_args()

    datasets_to_run = args.dataset or ["bose", "ceravolo1000", "ostovar"]

    all_results = {}
    for dk in datasets_to_run:
        res = tune_dataset(dk)
        all_results[dk] = res

    # ── Final summary ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  FINAL SUMMARY — Best config per dataset")
    print("=" * 70)
    for dk, res in all_results.items():
        best = res.iloc[0]
        print(f"\n  {DATASETS[dk]['label']}:")
        print(f"    strategy       = {best['strategy']}")
        print(f"    pen_scale      = {best['pen_scale']}")
        print(f"    min_effect_size= {best['min_effect_size']}")
        print(f"    micro F1       = {best['micro_F1']}")
        print(f"    micro P        = {best['micro_P']}")
        print(f"    micro R        = {best['micro_R']}")
        print(f"    TP={best['total_TP']}  FP={best['total_FP']}  FN={best['total_FN']}")


if __name__ == "__main__":
    main()
