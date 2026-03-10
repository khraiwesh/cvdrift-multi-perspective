# CVDrift — Multi-Perspective Concept Drift Detection

A CV-based (Coefficient of Variation) concept drift detection pipeline for process mining event logs.  
Detects three types of drift:
- **Duration**: Changes in activity execution times
- **Routing**: Changes in control-flow / transition probabilities between activities
- **Arrival**: Changes in inter-arrival times between cases

---

## Requirements

- **Python 3.10+**

### Required Python Packages

| Package | Description | Used By |
|---------|-------------|---------|
| `numpy` | Numerical computations | All modules |
| `pandas` | DataFrame operations | All modules |
| `ruptures` | PELT change-point detection algorithm | `pipeline/drift_detection.py` |
| `matplotlib` | Plotting and visualization | `pipeline/runner.py`, `plot_tuning.py` |
| `python-dateutil` | Timestamp parsing | `pipeline/io.py` |
| `openpyxl` | Excel file reading/writing | `compare_all_methods.py`, `run_unified.py` |
| `pm4py` *(optional)* | Advanced XES file reading | `pipeline/io.py` — falls back to built-in XML parser if not installed |

---

## Installation

### 1. Clone or download the repository

```bash
git clone <repo-url>
cd cvdrift-multi-perspective
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows:
venv\Scripts\activate

# Linux/macOS:
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install numpy pandas ruptures matplotlib python-dateutil openpyxl

pip install -r requirements.txt 
```

**Optional** — for enhanced XES support via pm4py:
```bash
pip install pm4py
```

> **Note:** If `pm4py` is not installed, the project uses its own lightweight XML parser to read XES files.

---

## Usage

### Unified Combined Runner (`run_combined.py`) ⭐ Recommended

The primary recommended entry point. Replaces both `main.py` and `evaluation.py`. Supports single-file, multi-file, and folder-batch modes. Results are auto-saved to the `output/` folder with a timestamp in the filename (e.g. `output/batch_results_20260310_143045.csv`).

```bash
# Single file — duration drift only
python run_combined.py --file "Datasets/Evaluation/Evaluation Logs/Experiment #1 (Activity duration)/dataset_1000cases_10min_ABD.xes" --drift duration

# Single file — routing drift
python run_combined.py --file "log.xes" --drift routing

# Entire folder — duration drift, Samira datasets (auto-evaluates after detection)
python run_combined.py --dir "Datasets/Evaluation/Evaluation Logs/Experiment #1 (Activity duration)" --drift duration --gt-mode samira

# Entire folder — routing drift, Ceravolo datasets (auto-evaluates after detection)
python run_combined.py --dir "Datasets/Evaluation/Evaluation Logs/Experiment #2 (control flow )/Ceravolo/All" --drift routing

# Entire folder — routing drift, Ostovar datasets
python run_combined.py --dir "Datasets/Evaluation/Evaluation Logs/Experiment #2 (control flow )/Ostovar" --drift routing --gt-mode ostovar
```

**Parameters:**
| Parameter | Description | Default |
|-----------|-------------|---------|
| `--file` | Single event log file | Interactive picker if omitted |
| `--files` | Multiple event log files (space-separated) | — |
| `--dir` | Folder — all logs inside will be processed | — |
| `--drift` | Drift type(s): `duration`, `routing`, `arrival` | All three |
| `--out` | Output CSV path (overrides auto-timestamp name) | `output/batch_results_<timestamp>.csv` |
| `--gt-mode` | Ground-truth mode: `samira`, `ceravolo`, `ostovar` | `samira` |
| `--tol` | Tolerance window for TP matching (in cases) | Auto: `max(10% × n_cases, 10)` |
| `--window-strategy` | `cv_perpair` or `mode_window` | `cv_perpair` |
| `--actual-cps` | Override GT with explicit JSON list, e.g. `"[370, 750]"` | — |

**Output files (auto-created in `output/` folder):**
| File | Description |
|------|-------------|
| `output/results_<timestamp>.csv` | Single-file mode detection results |
| `output/batch_results_<timestamp>.csv` | Batch mode detection results |
| `output/eval_batch_results_<timestamp>.csv` | Batch mode evaluation (P/R/F1), auto-generated after detection |

> **Auto-evaluation (batch mode only):** After `--files` or `--dir` runs, P/R/F1 evaluation is automatically computed and saved:
> - `--drift routing` → uses `eval_routing.py` logic: GT = `size/2`, tol = `10% × size` (Ceravolo convention)
> - `--drift duration` or combined → uses `evaluate_from_csv.py` with the specified `--gt-mode` (`samira` by default)
>
> Single-file mode (`--file`) only saves detection results — no automatic evaluation.

---

### Hyperparameter Tuning (`tune_pelt.py`)

Grid-search optimization over PELT algorithm parameters.

```bash
# Run on all datasets
python tune_pelt.py

# Run on a specific dataset
python tune_pelt.py --dataset bose
python tune_pelt.py --dataset ceravolo1000
python tune_pelt.py --dataset ostovar
```

> **Note:** Update the `BASE_EVAL` path in `tune_pelt.py` to point to your local dataset directory.

---

### Post-hoc Analysis Scripts

These scripts analyze detection results after the pipeline has run:

| Script | Description | Usage |
|--------|-------------|-------|
| `evaluate_from_csv.py` | Compute P/R/F1 from a detection CSV | `python evaluate_from_csv.py --csv results.csv --tol 50` |
| `compute_metrics.py` | Metrics from resultsFinal.cvs | `python compute_metrics.py [csv_path]` |
| `eval_routing.py` | P/R/F1 for routing results | `python eval_routing.py [csv_path]` |
| `compare_all_methods.py` | Compare CVDrift, MDD, OC methods | `python compare_all_methods.py` |
| `report_tuning.py` | Best universal configuration report | `python report_tuning.py` |
| `plot_tuning.py` | Tuning result plots | `python plot_tuning.py` |
| `build_summary.py` | Summary table builder | `python build_summary.py` |

---

## Project Architecture

```
cvdrift-multi-perspective/
│
├── main.py                  # Single-file detection (legacy entry point)
├── preparation.py           # Builds case-indexed time series per drift type
├── evaluation.py            # Batch evaluator (folder → CSV, legacy)
├── run_combined.py          # ⭐ Unified entry point (single / multi / batch)
├── run_unified.py           # Unified batch runner (duration + routing)
├── tune_pelt.py             # Grid-search hyperparameter tuning
│
├── pipeline/                # Core pipeline package
│   ├── io.py                # Log reading (CSV / XES)
│   ├── preprocessing.py     # Timestamp parsing, dual event log creation
│   ├── series_duration.py   # Case-indexed duration time series
│   ├── series_routing.py    # Case-indexed routing probability series
│   ├── series_arrival.py    # Case-indexed inter-arrival time series
│   ├── rolling.py           # Rolling window statistics
│   ├── window_selection.py  # CV + knee window size selection
│   ├── drift_detection.py   # PELT change-point detection (ruptures)
│   ├── consensus.py         # Consensus voting (proximity clustering)
│   └── runner.py            # Internal orchestrator (wires all stages)
│
├── evaluate_from_csv.py     # Post-hoc: P/R/F1 from CSV
├── compute_metrics.py       # Post-hoc: metric computation
├── eval_routing.py          # Post-hoc: routing evaluation
├── plot_tuning.py           # Post-hoc: tuning plots
├── report_tuning.py         # Post-hoc: best configuration report
├── compare_all_methods.py   # Post-hoc: method comparison
├── build_summary.py         # Post-hoc: summary table
│
├── Datasets/                # Event log datasets
│   └── Evaluation/
│       ├── Evaluation Logs/
│       │   ├── Experiment #1 (Activity duration)/   # Duration drift experiments (XES)
│       │   └── Experiment #2 (control flow)/        # Control-flow drift experiments
│       │       ├── Bose/
│       │       ├── Ceravolo/
│       │       └── Ostovar/
│       └── DurationDrift Approaches/                # MDD etc. comparison
│
└── ARCHITECTURE.md          # Detailed architecture documentation
```

---

## Pipeline Flow

Drift detection proceeds in 4 stages:

```
  1. PREPROCESSING (preparation.py → preprocess)
     ↓  Event log → elog_dur, elog_seq, seq_with_next
  
  2. SERIES CONSTRUCTION (preparation.py → preparation)
     ↓  Build case-indexed time series per drift type
  
  3. WINDOW SELECTION (main.py → select_window)
     ↓  CV + knee method → optimal rolling window size
  
  4. DETECTION (pipeline/runner.py → detect_drifts)
     ↓  PELT algorithm + consensus voting → drift points
```

---

## Supported File Formats

| Format | Description |
|--------|-------------|
| `.xes` | IEEE XES event log standard |
| `.xes.gz` | Compressed XES |
| `.csv` | Comma-separated values |
| `.mxml` | MXML event log format |

### CSV File Format

Expected column names for CSV files (defaults):

| Column | Description |
|--------|-------------|
| `Case ID` | Case identifier |
| `Activity` | Activity name |
| `Start Timestamp` | Start timestamp |
| `Complete Timestamp` | Completion timestamp |
| `Resource` | Resource (worker) name |

> Column names can be customized via `DEFAULT_PARAMS` in `preparation.py`.

---

## Default Parameters

The main configuration is defined in the `DEFAULT_PARAMS` dictionary in `preparation.py`:

```python
DEFAULT_PARAMS = dict(
    candidate_windows=[15, 20, 30, 50, 100, 200, 300, 400, 500, 600,
                       1000, 1500, 2000, 3000, 5000],
    duration_stat="median",       # Rolling window statistic for duration
    routing_stat="mean",          # Rolling window statistic for routing
    arrival_stat="median",        # Rolling window statistic for arrival
    pen_scale=3.0,                # PELT penalty scale
    cpd_model="l2",               # PELT model
    min_cp_distance=10,           # Minimum distance between changepoints
    min_effect_size=0.15,         # Minimum effect size threshold
    window_strategy="cv_perpair", # Window selection strategy
)
```

---

## Quick Start — Example Run

```bash
# 1. Install dependencies
pip install numpy pandas ruptures matplotlib python-dateutil openpyxl

# 2a. Single file — duration drift detection
python run_combined.py --file "Datasets/Evaluation/Evaluation Logs/Experiment #1 (Activity duration)/dataset_1000cases_10min_ABD.xes" --drift duration

# 2b. Batch — all Ceravolo logs, routing drift (auto-evaluates after detection)
python run_combined.py --dir "Datasets/Evaluation/Evaluation Logs/Experiment #2 (control flow )/Ceravolo/All" --drift routing

# 3. Results are saved automatically to output/ folder:
#    output/batch_results_<timestamp>.csv      — detection results
#    output/eval_batch_results_<timestamp>.csv — P/R/F1 evaluation
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: No module named 'ruptures'` | `pip install ruptures` |
| `ModuleNotFoundError: No module named 'pipeline'` | Ensure you are running from the project root directory |
| Cannot read XES files | `pip install pm4py` or rely on the built-in parser |
| `openpyxl` error | `pip install openpyxl` (only needed for Excel outputs) |
| Path errors in `tune_pelt.py` | Update the `BASE_EVAL` variable to your local dataset path |
