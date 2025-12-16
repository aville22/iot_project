from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

VITAL_COLUMNS = ["heart_rate", "spo2", "sbp", "dbp"]


@dataclass
class WindowConfig:
    """
    Windowing configuration for building information arrays.

    window_seconds:
        Time span of each window.

    stride_seconds:
        Step between window start times.

    label_strategy:
        How to label the window if multiple samples exist inside:
        - "mode": most frequent risk_level
        - "last": last sample risk_level
        - "max": maximum risk_level (more conservative)
    """
    window_seconds: int = 30
    stride_seconds: int = 10
    label_strategy: str = "mode"


def _label_window(risk_series: pd.Series, strategy: str) -> int:
    """
    Choose a single label for a window.
    """
    if risk_series.empty:
        return -1

    if strategy == "mode":
        return int(risk_series.mode().iloc[0])
    if strategy == "last":
        return int(risk_series.iloc[-1])
    if strategy == "max":
        return int(risk_series.max())

    raise ValueError(f"Unknown label_strategy: {strategy}")


def _semantic_tags(agg: dict) -> dict:
    """
    Create semantic tags from aggregated vitals (rule-based).
    These tags are your 'semantic representation' layer.
    """
    hr_mean = agg["hr_mean"]
    spo2_min = agg["spo2_min"]
    sbp_mean = agg["sbp_mean"]

    tag_tachycardia = int(hr_mean >= 120)
    tag_hypoxemia = int(spo2_min < 90)
    tag_hypertension = int(sbp_mean >= 140)
    tag_hypotension = int(sbp_mean < 90)

    # One combined alert tag (simple but useful baseline)
    tag_alert = int(tag_tachycardia or tag_hypoxemia or tag_hypotension)

    return {
        "tag_tachycardia": tag_tachycardia,
        "tag_hypoxemia": tag_hypoxemia,
        "tag_hypertension": tag_hypertension,
        "tag_hypotension": tag_hypotension,
        "tag_alert": tag_alert,
    }


def build_information_arrays(
    df: pd.DataFrame,
    config: Optional[WindowConfig] = None,
) -> pd.DataFrame:
    """
    Convert a preprocessed stream into Information Arrays:
    - Windowed aggregation (stats per time window)
    - Semantic tagging

    Output: one row per (patient_id, window_start, window_end)
    """
    if config is None:
        config = WindowConfig()

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values(["patient_id", "timestamp"]).reset_index(drop=True)

    rows = []

    for patient_id, g in df.groupby("patient_id", sort=False):
        g = g.sort_values("timestamp")
        t_min = g["timestamp"].iloc[0]
        t_max = g["timestamp"].iloc[-1]

        # Build sliding windows over continuous time
        start = t_min
        while start <= t_max:
            end = start + pd.Timedelta(seconds=config.window_seconds)

            w = g[(g["timestamp"] >= start) & (g["timestamp"] < end)]
            if len(w) == 0:
                start = start + pd.Timedelta(seconds=config.stride_seconds)
                continue

            # Aggregations form your "information array" representation
            agg = {
                "patient_id": int(patient_id),
                "window_start": start,
                "window_end": end,
                "n_samples": int(len(w)),
                # Heart rate
                "hr_mean": float(w["heart_rate"].mean()),
                "hr_std": float(w["heart_rate"].std(ddof=0)),
                "hr_min": float(w["heart_rate"].min()),
                "hr_max": float(w["heart_rate"].max()),
                # SpO2
                "spo2_mean": float(w["spo2"].mean()),
                "spo2_std": float(w["spo2"].std(ddof=0)),
                "spo2_min": float(w["spo2"].min()),
                "spo2_max": float(w["spo2"].max()),
                # Blood pressure
                "sbp_mean": float(w["sbp"].mean()),
                "sbp_std": float(w["sbp"].std(ddof=0)),
                "sbp_min": float(w["sbp"].min()),
                "sbp_max": float(w["sbp"].max()),
                "dbp_mean": float(w["dbp"].mean()),
                "dbp_std": float(w["dbp"].std(ddof=0)),
                "dbp_min": float(w["dbp"].min()),
                "dbp_max": float(w["dbp"].max()),
            }

            # Window label
            agg["risk_level"] = _label_window(w["risk_level"], strategy=config.label_strategy)

            # Semantic tags (rule-based)
            agg.update(_semantic_tags(agg))

            rows.append(agg)

            start = start + pd.Timedelta(seconds=config.stride_seconds)

    return pd.DataFrame(rows)
