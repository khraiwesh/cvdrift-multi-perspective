# ============================================================
# pipeline/preprocessing.py — Sequence log + duration log preparation
# ============================================================
"""Parse timestamps and build the dual (sequence / duration) event logs."""

import re
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


# ------------------------------------------------------------------
# Timestamp helper
# ------------------------------------------------------------------

def _parse_ts(series: pd.Series, tz: str = "UTC") -> pd.Series:
    """Robust timestamp parsing, returns tz-aware (UTC default)."""
    s = series.astype(str)

    ts = None
    try:
        ts = pd.to_datetime(s, errors="coerce", utc=True, format="ISO8601")
    except Exception:
        pass

    if ts is None or int(ts.notna().sum()) == 0:
        ts = pd.to_datetime(s, errors="coerce", utc=True)

    non_empty = int((s.str.strip() != "").sum())
    parsed = int(ts.notna().sum())
    if non_empty > 0 and parsed == 0:
        ts = s.apply(lambda x: pd.to_datetime(x, errors="coerce", utc=True))

    if tz and tz.upper() != "UTC":
        try:
            ts = ts.dt.tz_convert(tz)
        except Exception:
            pass
    return ts


# ------------------------------------------------------------------
# Dataclass
# ------------------------------------------------------------------

@dataclass
class DualLogs:
    """Container for the sequence log and the duration-enriched log."""
    elog_seq: pd.DataFrame
    elog_dur: pd.DataFrame


# ------------------------------------------------------------------
# Log preparation
# ------------------------------------------------------------------

def prepare_seq_log(df: pd.DataFrame, case_col: str, act_col: str, start_col: str, tz: str = "UTC") -> pd.DataFrame:
    """
    Build a sorted sequence log with normalised column names (.case, .act, .start).

    Keeps Case IDs exactly as in df[case_col].
    If 'Orig Case ID' exists, keeps it as '.orig_case'.
    If Start Timestamp missing, falls back to 'Complete Timestamp' if present.
    """
    df2 = df.copy()
    df2.columns = [c.strip() for c in df2.columns]

    cols = [case_col, act_col, start_col]
    if "Orig Case ID" in df2.columns:
        cols.append("Orig Case ID")
    if "Event Index" in df2.columns:
        cols.append("Event Index")
    if "Complete Timestamp" in df2.columns and "Complete Timestamp" not in cols:
        cols.append("Complete Timestamp")

    elog_seq = df2[cols].copy()
    elog_seq = elog_seq.rename(columns={case_col: ".case", act_col: ".act", start_col: ".start"})

    elog_seq[".case"] = elog_seq[".case"].astype(str).str.strip()
    elog_seq[".act"] = elog_seq[".act"].astype(str).apply(lambda x: re.sub(r"\s+", " ", x).strip())

    if "Orig Case ID" in elog_seq.columns:
        elog_seq[".orig_case"] = elog_seq["Orig Case ID"].astype(str).str.strip()
    else:
        elog_seq[".orig_case"] = pd.NA

    elog_seq = elog_seq[(elog_seq[".case"] != "") & (elog_seq[".act"] != "")].copy()
    elog_seq[".start"] = _parse_ts(elog_seq[".start"], tz=tz)

    if "Complete Timestamp" in elog_seq.columns:
        miss_start = int(elog_seq[".start"].isna().sum())
        if miss_start > 0:
            end_ts = _parse_ts(elog_seq["Complete Timestamp"], tz=tz)
            elog_seq.loc[elog_seq[".start"].isna(), ".start"] = end_ts.loc[elog_seq[".start"].isna()]

    if "Event Index" in elog_seq.columns:
        elog_seq["Event Index"] = pd.to_numeric(elog_seq["Event Index"], errors="coerce")
        elog_seq = elog_seq[elog_seq["Event Index"].notna()].copy()
        elog_seq["Event Index"] = elog_seq["Event Index"].astype(int)
        elog_seq = elog_seq.sort_values([".case", "Event Index"], kind="mergesort")
    else:
        elog_seq = elog_seq[elog_seq[".start"].notna()].copy()
        elog_seq = elog_seq.sort_values([".case", ".start"], kind="mergesort")

    return elog_seq


def prepare_event_log_dual(
    df: pd.DataFrame,
    case_col: str,
    act_col: str,
    start_col: str,
    end_col: str,
    res_col: Optional[str] = None,
    tz: str = "UTC",
) -> DualLogs:
    """Build both the sequence log and the duration log from the raw DataFrame."""
    required = [case_col, act_col, start_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing columns in df: " + ", ".join(missing) +
            "\nAvailable columns:\n- " + "\n- ".join(df.columns.tolist())
        )

    df2 = df.copy()
    if res_col is None or res_col not in df2.columns:
        df2[".res_tmp"] = ""
        res_use = ".res_tmp"
    else:
        res_use = res_col

    df2[".case"] = df2[case_col].astype(str)
    df2[".act"] = df2[act_col].astype(str)
    df2[".start"] = _parse_ts(df2[start_col], tz=tz)
    df2[".end"] = _parse_ts(df2[end_col], tz=tz) if end_col in df2.columns else pd.NaT
    df2[".res"] = df2[res_use].astype(str)

    base = df2.loc[df2[".case"].notna() & df2[".act"].notna() & df2[".start"].notna()].copy()
    base = base.sort_values([".case", ".start"], kind="mergesort")
    if len(base) == 0:
        raise ValueError("No valid rows after parsing .case/.act/.start.")

    elog_seq = base

    elog_dur = base.loc[base[".end"].notna()].copy()
    if len(elog_dur) > 0:
        elog_dur[".dur_sec"] = (elog_dur[".end"] - elog_dur[".start"]).dt.total_seconds()
        elog_dur = elog_dur.loc[np.isfinite(elog_dur[".dur_sec"]) & (elog_dur[".dur_sec"] >= 0)].copy()

    return DualLogs(elog_seq=elog_seq, elog_dur=elog_dur)
