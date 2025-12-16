from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import math


@dataclass
class ValidationConfig:
    """
    Validation rules for incoming IoT measurements.
    """
    max_clock_skew_sec: float = 5.0   # how old an event may be before considered "late"
    clamp_values: bool = True         # clamp to physiological ranges
    drop_on_invalid: bool = False     # if True -> drop invalid events, else sanitize


RANGES = {
    "heart_rate": (30.0, 220.0),
    "spo2": (50.0, 100.0),
    "sbp": (50.0, 250.0),
    "dbp": (30.0, 150.0),
}


def _is_nan(x) -> bool:
    try:
        return math.isnan(float(x))
    except Exception:
        return True


def validate_and_sanitize(event: Dict, now_ts: float, cfg: ValidationConfig) -> Tuple[Dict | None, Dict]:
    """
    Validate and sanitize an incoming event.

    Returns:
      (sanitized_event_or_none, meta)
    meta includes:
      - dropped: bool
      - late: bool
      - corrected_fields: int
      - reason: str
    """
    meta = {"dropped": False, "late": False, "corrected_fields": 0, "reason": ""}

    # Basic schema checks
    for k in ("patient_id", "timestamp", "heart_rate", "spo2", "sbp", "dbp"):
        if k not in event:
            meta["dropped"] = True
            meta["reason"] = f"missing_field:{k}"
            return None, meta

    ts = float(event["timestamp"])
    if now_ts - ts > cfg.max_clock_skew_sec:
        meta["late"] = True

    e = dict(event)

    # Validate numeric fields
    for field, (lo, hi) in RANGES.items():
        val = e.get(field, None)
        if val is None or _is_nan(val):
            if cfg.drop_on_invalid:
                meta["dropped"] = True
                meta["reason"] = f"nan:{field}"
                return None, meta
            # sanitize: replace with mid-point
            e[field] = (lo + hi) / 2.0
            meta["corrected_fields"] += 1
            continue

        v = float(val)

        # Handle negative / absurd values
        if v < lo or v > hi:
            if cfg.drop_on_invalid:
                meta["dropped"] = True
                meta["reason"] = f"out_of_range:{field}"
                return None, meta
            if cfg.clamp_values:
                e[field] = min(hi, max(lo, v))
            else:
                e[field] = v
            meta["corrected_fields"] += 1

    return e, meta
