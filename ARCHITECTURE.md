# CVDriftPipeline_v2 — Architecture

## Overview

CV-based concept drift detection pipeline for process mining event logs.  
Detects three drift types: **duration**, **routing**, and **arrival time**.

---

## Entry Points

| Command | Description |
|---------|-------------|
| `python main.py --file "log.xes" --drift duration` | Run detection on a **single** log file |
| `python main.py --file "log.xes" --drift duration routing arrival` | Detect all three drift types |
| `python main.py --window-strategy mode_window --file "log.xes" --drift duration` | Use mode-window strategy |
| `python evaluation.py` | **Batch evaluate** all logs in a folder → results CSV |
| `python run_unified.py` | **Unified batch runner** for duration + routing |
| `python tune_pelt.py --dataset selected` | **Grid-search tuning** of PELT hyperparameters |
| `python tune_pelt.py --dataset bose` | Tune on Bose dataset |
| `python tune_pelt.py --dataset ceravolo1000` | Tune on Ceravolo dataset |
| `python tune_pelt.py --dataset ostovar` | Tune on Ostovar dataset |

---

## Directory Structure

```
CVDriftPipeline_v2/
│
├── main.py                  # Primary CLI entry point (single-file detection)
├── preparation.py           # Builds case-indexed time series per drift type
├── evaluation.py            # Batch evaluator (folder → CSV)
├── run_unified.py           # Unified batch runner (duration + routing)
├── tune_pelt.py             # Grid-search hyperparameter tuning
│
├── pipeline/                # Core pipeline package (leaf modules)
│   ├── __init__.py
│   ├── io.py                # Log loading (CSV / XES)
│   ├── preprocessing.py     # Timestamp parsing, dual event logs
│   ├── series_duration.py   # Case-indexed duration series
│   ├── series_routing.py    # Case-indexed routing probability series
│   ├── series_arrival.py    # Case-indexed inter-arrival time series
│   ├── rolling.py           # Rolling window statistics
│   ├── window_selection.py  # CV + knee window size selection
│   ├── drift_detection.py   # PELT change-point detection (ruptures)
│   ├── consensus.py         # Consensus voting (proximity clustering)
│   └── runner.py            # Internal orchestrator (wires all stages)
│
├── evaluate_from_csv.py     # Post-hoc: P/R/F1 from detection CSV
├── compute_metrics.py       # Post-hoc: metrics from resultsFinal
├── plot_tuning.py           # Post-hoc: tuning result plots
├── report_tuning.py         # Post-hoc: best universal config report
├── compare_all_methods.py   # Post-hoc: compare with published methods
├── build_summary.py         # Post-hoc: summary builder
│
├── Datasets/                # Event log datasets (XES files)
└── Others/                  # External method implementations
```

---

## Dependency Graph

```
                ┌──────────────────────────────────────────────────┐
                │           PIPELINE PACKAGE (Layer 1)             │
                │           No project imports — leaf modules      │
                │                                                  │
                │  io.py  preprocessing.py  window_selection.py    │
                │  series_duration.py  series_routing.py           │
                │  series_arrival.py   rolling.py                  │
                │  drift_detection.py  consensus.py                │
                └────────────────────┬─────────────────────────────┘
                                     │
                     ┌───────────────┼───────────────┐
                     ▼               ▼               │
              ┌─────────────┐  ┌──────────────┐      │
              │ runner.py   │  │preparation.py│      │
              │ (pipeline/) │  │  imports:    │      │
              │ imports all │  │  preproc,    │      │
              │ 8 pipeline  │  │  3 series    │      │
              │ modules     │  │  modules     │      │
              └──────┬──────┘  └──────┬───────┘      │
                     │                │              │
                     ▼                ▼              │
                ┌─────────────────────────┐          │
                │        main.py          │          │
                │  imports: pipeline.io,  │          │
                │   runner, window_sel,   │          │
                │   preparation           │          │
                └──┬───────┬──────────────┘          │
                   │       │                         │
          ┌────────┤       ├──────────┐              │
          ▼        ▼       ▼          ▼              │
   ┌──────────┐ ┌────────────┐ ┌──────────────┐     │
   │evaluation│ │run_unified │ │ tune_pelt.py │◄────┘
   │   .py    │ │   .py      │ │imports main +│
   │          │ │            │ │preparation + │
   │          │ │            │ │8 pipeline    │
   │          │ │            │ │modules direct│
   └──────────┘ └────────────┘ └──────────────┘

   ┌──────────────────────────────────────────────┐
   │      STANDALONE ANALYSIS (no project imports) │
   │  evaluate_from_csv.py   compute_metrics.py    │
   │  plot_tuning.py         report_tuning.py      │
   │  compare_all_methods.py build_summary.py      │
   └──────────────────────────────────────────────┘
```

---

## Pipeline Flow (per drift type)

```
  Event Log (XES/CSV)
        │
        ▼
  ┌─────────────────────────────────────────────┐
  │ Step 0: PREPROCESS  (preparation.py)        │
  │   prepare_event_log_dual() → elog_dur       │
  │   prepare_seq_log()        → elog_seq       │
  │   add_next_act()           → seq_with_next  │
  └─────────────────┬───────────────────────────┘
                    │
                    ▼
  ┌─────────────────────────────────────────────┐
  │ Step 1: PREPARATION  (preparation.py)       │
  │   Build case-indexed time series per type:  │
  │   • duration: one series per activity       │
  │   • routing:  one series per (from→to) pair │
  │   • arrival:  one inter-arrival series      │
  └─────────────────┬───────────────────────────┘
                    │
                    ▼
  ┌─────────────────────────────────────────────┐
  │ Step 2: WINDOW SELECTION  (main.py)         │
  │   select_window() → CV + knee method        │
  │   Strategy: cv_perpair | mode_window        │
  │   • cv_perpair: per-series optimal window   │
  │   • mode_window: uniform mode across series │
  └─────────────────┬───────────────────────────┘
                    │
                    ▼
  ┌─────────────────────────────────────────────┐
  │ Step 3: DETECTION  (pipeline/runner.py)     │
  │   detect_drifts_duration_and_routing()      │
  │                                             │
  │   3a. Multi-scale rolling series            │
  │       duration: w, w/2, w/4                 │
  │       routing:  w, w/2                      │
  │                                             │
  │   3b. PELT on each rolling series           │
  │       + consolidate nearby CPs              │
  │       + effect-size filter (mean diff)      │
  │                                             │
  │   3c. Raw PELT (duration ONLY)              │
  │       PELT on raw per-case values           │
  │       + Cohen's d ≥ 0.3 filter              │
  │                                             │
  │   3d. Consensus voting                      │
  │       duration: prefer raw-PELT clusters    │
  │       routing:  require ≥2 unique pairs     │
  └─────────────────────────────────────────────┘
```

---

## File-by-File Reference

### Core Modules

| File | Lines | Exports | Role |
|------|-------|---------|------|
| `main.py` | 523 | `run_pipeline_single()`, `select_window()`, `_apply_mode_window()`, `_load_log()` | Primary CLI + pipeline orchestrator |
| `preparation.py` | 305 | `preparation()`, `preprocess()`, `DEFAULT_PARAMS` | Time series construction + default config |
| `pipeline/runner.py` | 574 | `detect_drifts_duration_and_routing()`, `select_windows_duration_and_routing()` | Detection engine: rolling → PELT → consensus |

### Pipeline Modules (pipeline/)

| File | Lines | Exports | Role |
|------|-------|---------|------|
| `io.py` | 480 | `get_event_log()`, `read_xes_to_dataframe()` | Log ingestion (CSV, XES, pm4py fallback) |
| `preprocessing.py` | 154 | `prepare_event_log_dual()`, `prepare_seq_log()` | Timestamp parsing, dual event log creation |
| `series_duration.py` | 72 | `series_duration_case_indexed()` | Per-activity duration time series |
| `series_routing.py` | 93 | `add_next_act()`, `build_routing_pairs_from_elog()`, `series_routing_case_indexed()` | Routing probability time series |
| `series_arrival.py` | 106 | `series_arrival_case_indexed()` | Inter-arrival time series |
| `rolling.py` | 83 | `window_stat_series()` | Rolling window mean/median computation |
| `window_selection.py` | 119 | `choose_window_size_stability()`, `WindowSelectionResult` | CV + knee window selection |
| `drift_detection.py` | 94 | `detect_drift_pelt()`, `consolidate_changepoints()` | PELT CPD via ruptures library |
| `consensus.py` | 180 | `compute_routing_consensus()`, `compute_duration_consensus()` | Proximity clustering + type-specific filtering |

### Execution Scripts

| File | Lines | Role |
|------|-------|------|
| `evaluation.py` | 202 | Batch evaluate all logs in a folder → P/R/F1 CSV |
| `run_unified.py` | 398 | Unified batch runner for duration + routing detection |
| `tune_pelt.py` | 701 | Grid-search tuning with CachedSeries optimisation |

### Post-hoc Analysis (standalone, no project imports)

| File | Lines | Role |
|------|-------|------|
| `evaluate_from_csv.py` | 250 | Compute P/R/F1 from detection CSV vs filename-derived GT |
| `compute_metrics.py` | 279 | Compute metrics from resultsFinal with GT at 37%/75% |
| `plot_tuning.py` | 50 | 2×2 grid plots: micro-F1 vs effect-size by pen_scale |
| `report_tuning.py` | 109 | Join tuning CSVs, find best universal configuration |
| `compare_all_methods.py` | — | Compare pipeline vs published methods |
| `build_summary.py` | — | Summary builder |

---

## Key Parameters (DEFAULT_PARAMS)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `pen_scale` | 3.0 | PELT penalty multiplier: `pen = scale × ln(n)` |
| `min_effect_size` | 0.15 | Minimum raw mean difference to keep a CP |
| `cpd_model` | "l2" | PELT cost model |
| `min_cp_distance` | 10 | Minimum distance between consolidated CPs |
| `min_n_points` | 10 | Skip series shorter than this |
| `candidate_windows` | [15..5000] | Window sizes tried by CV+knee |
| `duration_stat` | "median" | Rolling statistic for duration |
| `duration_per_case` | "median" | Aggregation of per-case activity durations |
| `routing_stat` | "mean" | Rolling statistic for routing |
| `knee_policy` | "before" | Knee selection policy |
| `window_strategy` | "cv_perpair" | "cv_perpair" or "mode_window" |

---

## Duration vs Routing — Key Differences

| Aspect | Duration | Routing |
|--------|----------|---------|
| Series | One per **activity** | One per **(from→to) pair** |
| Values | Execution time (seconds) | Transition probability [0,1] |
| Rolling stat | `median` | `mean` |
| Pre-filtering | None | Rare pairs: `min_count` + `min_mean_p` |
| Multi-scale | 3 scales (w, w/2, w/4) | 2 scales (w, w/2) |
| Raw PELT pass | Yes (Cohen's d ≥ 0.3) | No |
| Consensus | Prefer raw-PELT clusters | Require ≥2 unique pairs |

---

## Tuning Results (Selected dataset, 68 logs)

Best configuration (tune_pelt.py --dataset selected):

| Parameter | Value |
|-----------|-------|
| Strategy | cv_perpair |
| pen_scale | **3.0** |
| min_effect_size | 0.30 (all values identical) |
| **Micro F1** | **0.8746** |
| Micro Precision | 0.8531 |
| Micro Recall | 0.8971 |
| TP / FP / FN | 122 / 21 / 14 |
