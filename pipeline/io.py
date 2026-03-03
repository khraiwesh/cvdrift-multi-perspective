# ============================================================
# pipeline/io.py — Event-log reading (CSV / XES)
# ============================================================
"""Read event logs from CSV or XES files into a pandas DataFrame."""

from typing import List, Dict, Any
import xml.etree.ElementTree as ET
import pandas as pd
from statistics import median
from datetime import timedelta
import dateutil.parser as _dtparse

# Optional: pm4py for robust XES handling; fallback to lightweight XML parser
try:
    from pm4py.objects.log.importer.xes import factory as xes_importer  # pm4py older
    HAVE_PM4PY = True
except Exception:
    try:
        from pm4py.objects.log.importer.xes import importer as xes_importer  # pm4py newer
        HAVE_PM4PY = True
    except Exception:
        HAVE_PM4PY = False


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _find_local_children(elem, localname: str):
    """Return children with matching local-name (ignores XML namespaces)."""
    return [c for c in list(elem) if c.tag.endswith(localname)]


def _has_lifecycle(events_sample) -> bool:
    """Quick check: do any events carry a lifecycle:transition value?"""
    for ev in events_sample:
        lc = ev.get("lifecycle:transition") or ev.get("lifecycle") or ""
        if str(lc).strip():
            return True
    return False


def _pair_lifecycle_events(events, include_resource: bool) -> list:
    """
    Pair start/complete lifecycle events per activity within a single trace.

    Groups events by activity name.  For each activity the i-th start is
    paired with the i-th complete (positional pairing — works correctly
    when activities don't interleave).  If only completes exist (no
    starts), the complete timestamp is used for both start and end.

    Returns a list of dicts ready to be appended to the rows list.
    """
    activity_events: Dict[str, Dict[str, list]] = {}

    for ev in events:
        act = ev.get("concept:name") or ev.get("activity") or ""
        ts = ev.get("time:timestamp")
        ts_str = ts.isoformat() if hasattr(ts, "isoformat") else ("" if ts is None else str(ts))
        resource = (ev.get("org:resource") or ev.get("resource") or "") if include_resource else ""
        lifecycle = str(ev.get("lifecycle:transition") or ev.get("lifecycle") or "").strip().lower()

        act_key = str(act)
        if act_key not in activity_events:
            activity_events[act_key] = {"starts": [], "completes": []}

        if lifecycle.startswith("start"):
            activity_events[act_key]["starts"].append({"ts": ts_str, "resource": resource})
        elif lifecycle.startswith("complete"):
            activity_events[act_key]["completes"].append({"ts": ts_str, "resource": resource})

    paired: list = []
    for act, evts in activity_events.items():
        starts = evts["starts"]
        completes = evts["completes"]

        # Pair the i-th start with the i-th complete
        n_pairs = min(len(starts), len(completes))
        for i in range(n_pairs):
            paired.append({
                "Activity": str(act),
                "Start Timestamp": starts[i]["ts"],
                "Complete Timestamp": completes[i]["ts"],
                "Resource": completes[i]["resource"] or starts[i]["resource"],
            })

        # Completes without matching starts (e.g. Ostovar / Bose logs)
        if len(starts) == 0 and len(completes) > 0:
            for comp in completes:
                paired.append({
                    "Activity": str(act),
                    "Start Timestamp": comp["ts"],
                    "Complete Timestamp": comp["ts"],
                    "Resource": comp["resource"],
                })
    return paired


def _pair_lifecycle_events_xml(events, include_resource: bool) -> list:
    """XML-branch equivalent of _pair_lifecycle_events."""
    activity_events: Dict[str, Dict[str, list]] = {}

    for ev in events:
        act = ts_str = lifecycle = resource = ""
        for attr in list(ev):
            key = attr.attrib.get("key", "")
            val = attr.attrib.get("value", "")
            if attr.tag.endswith("string") and key in ("concept:name", "activity"):
                act = val
            if attr.tag.endswith("date") and key in ("time:timestamp", "timestamp"):
                ts_str = val
            if attr.tag.endswith("string") and key in ("lifecycle:transition", "lifecycle"):
                lifecycle = val.lower()
            if include_resource and attr.tag.endswith("string") and key in ("org:resource", "resource"):
                resource = val

        act_key = str(act)
        if act_key not in activity_events:
            activity_events[act_key] = {"starts": [], "completes": []}

        if lifecycle.startswith("start"):
            activity_events[act_key]["starts"].append({"ts": ts_str, "resource": resource})
        elif lifecycle.startswith("complete"):
            activity_events[act_key]["completes"].append({"ts": ts_str, "resource": resource})

    paired: list = []
    for act, evts in activity_events.items():
        starts = evts["starts"]
        completes = evts["completes"]

        n_pairs = min(len(starts), len(completes))
        for i in range(n_pairs):
            paired.append({
                "Activity": str(act),
                "Start Timestamp": starts[i]["ts"],
                "Complete Timestamp": completes[i]["ts"],
                "Resource": completes[i]["resource"] or starts[i]["resource"],
            })

        if len(starts) == 0 and len(completes) > 0:
            for comp in completes:
                paired.append({
                    "Activity": str(act),
                    "Start Timestamp": comp["ts"],
                    "Complete Timestamp": comp["ts"],
                    "Resource": comp["resource"],
                })
    return paired


# ------------------------------------------------------------------
# Median per-activity gap estimation helpers
# ------------------------------------------------------------------

def _safe_parse_ts(ts_str: str):
    """Parse an ISO timestamp string; return None on failure."""
    if not ts_str:
        return None
    try:
        return _dtparse.parse(ts_str)
    except Exception:
        return None


def _compute_median_activity_gaps_pm4py(log) -> Dict[str, float]:
    """
    Scan all traces (pm4py EventLog) and collect consecutive-event time
    gaps grouped by the *first* event's activity name.  Return the
    **median gap in seconds** for each activity.
    """
    from collections import defaultdict
    gaps: Dict[str, list] = defaultdict(list)

    for trace in log:
        events = list(trace)
        for i in range(len(events) - 1):
            act = events[i].get("concept:name") or events[i].get("activity") or ""
            ts0 = events[i].get("time:timestamp")
            ts1 = events[i + 1].get("time:timestamp")
            if ts0 is None or ts1 is None:
                continue
            # Convert to datetime if needed
            if hasattr(ts0, "timestamp"):
                d0 = ts0
            else:
                d0 = _safe_parse_ts(str(ts0))
            if hasattr(ts1, "timestamp"):
                d1 = ts1
            else:
                d1 = _safe_parse_ts(str(ts1))
            if d0 is None or d1 is None:
                continue
            delta = (d1 - d0).total_seconds()
            if delta >= 0:
                gaps[str(act)].append(delta)

    return {act: median(vals) for act, vals in gaps.items() if vals}


def _compute_median_activity_gaps_xml(traces, find_events_fn) -> Dict[str, float]:
    """
    XML-branch equivalent: scan traces, compute median consecutive-event
    gap per activity.  Returns median gap in seconds per activity.
    """
    from collections import defaultdict
    gaps: Dict[str, list] = defaultdict(list)

    for trace in traces:
        xml_events = find_events_fn(trace, "event")
        # Extract (activity, timestamp_str) pairs
        ev_list = []
        for ev in xml_events:
            act = ts_str = ""
            for attr in list(ev):
                key = attr.attrib.get("key", "")
                val = attr.attrib.get("value", "")
                if attr.tag.endswith("string") and key in ("concept:name", "activity"):
                    act = val
                if attr.tag.endswith("date") and key in ("time:timestamp", "timestamp"):
                    ts_str = val
            ev_list.append((act, ts_str))

        for i in range(len(ev_list) - 1):
            act = ev_list[i][0]
            d0 = _safe_parse_ts(ev_list[i][1])
            d1 = _safe_parse_ts(ev_list[i + 1][1])
            if d0 is None or d1 is None:
                continue
            delta = (d1 - d0).total_seconds()
            if delta >= 0:
                gaps[str(act)].append(delta)

    return {act: median(vals) for act, vals in gaps.items() if vals}


# ------------------------------------------------------------------
# XES -> DataFrame
# ------------------------------------------------------------------

def read_xes_to_dataframe(path: str, include_resource: bool = True) -> pd.DataFrame:
    """
    Parse XES into a DataFrame with columns:
        Case ID | Orig Case ID | Activity | Start Timestamp |
        Complete Timestamp | Resource | Event Index

    - Case ID:      unique per trace (trace index)
    - Orig Case ID: preserves the original concept:name
    - Lifecycle:    if start+complete pairs exist, they are paired per
                    activity to recover real durations.  If only completes
                    exist, start == complete (duration = 0).
    """
    rows: List[Dict[str, str]] = []

    if HAVE_PM4PY:
        log = xes_importer.apply(path)

        # Detect lifecycle once from first trace
        first_events = list(log[0]) if log else []
        lifecycle_present = _has_lifecycle(first_events)

        # Pre-compute median per-activity gaps (used only when lifecycle absent)
        if not lifecycle_present:
            median_gaps = _compute_median_activity_gaps_pm4py(log)
        else:
            median_gaps = {}

        for ti, trace in enumerate(log):
            orig_case_id = (
                getattr(trace, "attributes", {}).get("concept:name")
                or getattr(trace, "attributes", {}).get("case:concept:name")
                or str(ti)
            )
            case_id = str(ti)

            if lifecycle_present:
                paired = _pair_lifecycle_events(trace, include_resource)
                for ei, p in enumerate(paired):
                    rows.append({
                        "Case ID": str(case_id),
                        "Orig Case ID": str(orig_case_id),
                        "Activity": p["Activity"],
                        "Start Timestamp": p["Start Timestamp"],
                        "Complete Timestamp": p["Complete Timestamp"],
                        "Resource": p["Resource"],
                        "Event Index": str(ei),
                    })
            else:
                # Hybrid duration estimation:
                #  - non-last events: consecutive-event gap (preserves variance)
                #  - last event: median gap for that activity (avoids 0-duration)
                #  - outlier clipping: cap at 3× median to avoid inter-case contamination
                trace_events = list(trace)
                trace_dts = []
                for ev in trace_events:
                    ts = ev.get("time:timestamp")
                    if ts is not None and hasattr(ts, "timestamp"):
                        trace_dts.append(ts)
                    else:
                        trace_dts.append(_safe_parse_ts(
                            ts.isoformat() if hasattr(ts, "isoformat") else ("" if ts is None else str(ts))
                        ))

                for ei, ev in enumerate(trace_events):
                    act = ev.get("concept:name") or ev.get("activity") or ""
                    ts = ev.get("time:timestamp")
                    ts_str = ts.isoformat() if hasattr(ts, "isoformat") else ("" if ts is None else str(ts))
                    resource = (ev.get("org:resource") or ev.get("resource") or "") if include_resource else ""

                    start_dt = trace_dts[ei]
                    med_gap = median_gaps.get(str(act))

                    if ei + 1 < len(trace_dts) and start_dt and trace_dts[ei + 1]:
                        # Non-last event: use gap to next event
                        raw_gap = (trace_dts[ei + 1] - start_dt).total_seconds()
                        # Clip outliers at 3× median (if median available)
                        if med_gap and raw_gap > 3.0 * med_gap:
                            gap = med_gap
                        else:
                            gap = max(raw_gap, 0.0)
                    elif med_gap and start_dt:
                        # Last event or missing next: use median
                        gap = med_gap
                    else:
                        gap = 0.0

                    if start_dt and gap > 0:
                        end_dt = start_dt + timedelta(seconds=gap)
                        end_ts = end_dt.isoformat()
                    else:
                        end_ts = ts_str

                    rows.append({
                        "Case ID": str(case_id),
                        "Orig Case ID": str(orig_case_id),
                        "Activity": str(act),
                        "Start Timestamp": ts_str,
                        "Complete Timestamp": end_ts,
                        "Resource": str(resource),
                        "Event Index": str(ei),
                    })

    else:
        # ---- lightweight XML fallback (no pm4py) ----
        tree = ET.parse(path)
        root = tree.getroot()
        traces = _find_local_children(root, "trace")

        # Detect lifecycle from a small sample
        lifecycle_present = False
        for trace in traces[:50]:
            for ev in _find_local_children(trace, "event"):
                for attr in list(ev):
                    if attr.tag.endswith("string") and attr.attrib.get("key") in ("lifecycle:transition", "lifecycle"):
                        if str(attr.attrib.get("value", "")).strip():
                            lifecycle_present = True
                            break
                if lifecycle_present:
                    break
            if lifecycle_present:
                break

        # Pre-compute median per-activity gaps (used only when lifecycle absent)
        if not lifecycle_present:
            median_gaps = _compute_median_activity_gaps_xml(traces, _find_local_children)
        else:
            median_gaps = {}

        for ti, trace in enumerate(traces):
            orig_case_id = None
            for s in [c for c in list(trace) if c.tag.endswith("string")]:
                if s.attrib.get("key") in ("concept:name", "case:concept:name", "case:id"):
                    orig_case_id = s.attrib.get("value")
                    break
            orig_case_id = orig_case_id or str(ti)
            case_id = str(ti)

            xml_events = _find_local_children(trace, "event")

            if lifecycle_present:
                paired = _pair_lifecycle_events_xml(xml_events, include_resource)
                for ei, p in enumerate(paired):
                    rows.append({
                        "Case ID": str(case_id),
                        "Orig Case ID": str(orig_case_id),
                        "Activity": p["Activity"],
                        "Start Timestamp": p["Start Timestamp"],
                        "Complete Timestamp": p["Complete Timestamp"],
                        "Resource": p["Resource"],
                        "Event Index": str(ei),
                    })
            else:
                # Hybrid duration estimation (XML branch)
                parsed_events = []
                for ev in xml_events:
                    act = ts_str = resource = ""
                    for attr in list(ev):
                        key = attr.attrib.get("key", "")
                        val = attr.attrib.get("value", "")
                        if attr.tag.endswith("string") and key in ("concept:name", "activity"):
                            act = val
                        if attr.tag.endswith("date") and key in ("time:timestamp", "timestamp"):
                            ts_str = val
                        if include_resource and attr.tag.endswith("string") and key in ("org:resource", "resource"):
                            resource = val
                    parsed_events.append({"act": act, "ts": ts_str, "resource": resource,
                                          "dt": _safe_parse_ts(ts_str)})

                for ei, pe in enumerate(parsed_events):
                    start_ts = pe["ts"]
                    start_dt = pe["dt"]
                    med_gap = median_gaps.get(str(pe["act"]))

                    if ei + 1 < len(parsed_events) and start_dt and parsed_events[ei + 1]["dt"]:
                        raw_gap = (parsed_events[ei + 1]["dt"] - start_dt).total_seconds()
                        if med_gap and raw_gap > 3.0 * med_gap:
                            gap = med_gap
                        else:
                            gap = max(raw_gap, 0.0)
                    elif med_gap and start_dt:
                        gap = med_gap
                    else:
                        gap = 0.0

                    if start_dt and gap > 0:
                        end_dt = start_dt + timedelta(seconds=gap)
                        end_ts = end_dt.isoformat()
                    else:
                        end_ts = start_ts

                    rows.append({
                        "Case ID": str(case_id),
                        "Orig Case ID": str(orig_case_id),
                        "Activity": str(pe["act"]),
                        "Start Timestamp": start_ts,
                        "Complete Timestamp": end_ts,
                        "Resource": str(pe["resource"]),
                        "Event Index": str(ei),
                    })

    df = pd.DataFrame(
        rows,
        columns=["Case ID", "Orig Case ID", "Activity", "Start Timestamp",
                 "Complete Timestamp", "Resource", "Event Index"]
    )
    df = df.fillna("").astype(str)
    return df


# ------------------------------------------------------------------
# File-picker helper
# ------------------------------------------------------------------

def get_event_log(sep: str = ",") -> pd.DataFrame:
    """Open file picker (CSV or XES) and return a DataFrame suitable for the pipeline."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askopenfilename(
            title="Select event log (CSV or XES)",
            filetypes=[("XES files", "*.xes;*.xml"), ("CSV files", "*.csv"), ("All files", "*.*")]
        )
        root.destroy()
        if not path:
            raise RuntimeError("No file selected.")

        if path.lower().endswith((".xes", ".xml")):
            return read_xes_to_dataframe(path, include_resource=True)

        return pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False)

    except Exception:
        path = input("Paste full CSV/XES path: ").strip()
        if not path:
            raise RuntimeError("No file path provided.")
        if path.lower().endswith((".xes", ".xml")):
            return read_xes_to_dataframe(path, include_resource=True)
        return pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False)
