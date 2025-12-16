# realtime/processor.py
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Optional, Tuple


def _safe_float(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        x = float(v)
        if x != x:  # NaN
            return None
        return x
    except Exception:
        return None


def _mean(xs):
    return sum(xs) / len(xs) if xs else None


def _std(xs):
    if not xs:
        return None
    mu = _mean(xs)
    var = sum((x - mu) ** 2 for x in xs) / len(xs)  # population std
    return var ** 0.5


def _min(xs):
    return min(xs) if xs else None


def _max(xs):
    return max(xs) if xs else None


@dataclass
class WindowConfig:
    window_sec: float = 30.0
    step_sec: float = 5.0
    min_samples: int = 10


class WindowProcessor:
    """
    Per-patient sliding window over event-time.

    Incoming event:
      {
        "patient_id": int|str,
        "timestamp": float,
        "vitals": {"hr": .., "spo2": .., "sbp": .., "dbp": ..}
      }

    Emits features that match data/metrics.json feature_names:
      n_samples,
      hr_mean/hr_std/hr_min/hr_max,
      spo2_mean/spo2_std/spo2_min/spo2_max,
      sbp_mean/sbp_std/sbp_min/sbp_max,
      dbp_mean/dbp_std/dbp_min/dbp_max
    """

    def __init__(self, cfg: WindowConfig):
        self.cfg = cfg
        self.buffers: Dict[str, Deque[Tuple[float, Dict[str, Any]]]] = defaultdict(deque)
        self.last_emit_ts: Dict[str, float] = {}

    def push(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        pid = str(event.get("patient_id", "unknown"))
        ts = _safe_float(event.get("timestamp"))
        vitals = event.get("vitals") or {}

        if ts is None:
            return None

        buf = self.buffers[pid]
        buf.append((ts, vitals))

        # evict old by event-time
        cutoff = ts - float(self.cfg.window_sec)
        while buf and buf[0][0] < cutoff:
            buf.popleft()

        last = self.last_emit_ts.get(pid)
        if last is None:
            # emit first time when enough samples
            if len(buf) >= self.cfg.min_samples:
                self.last_emit_ts[pid] = ts
                return self._compute_features(pid, ts)
            return None

        if ts - last >= float(self.cfg.step_sec):
            self.last_emit_ts[pid] = ts
            return self._compute_features(pid, ts)

        return None

    def _compute_features(self, patient_id: str, window_end_ts: float) -> Optional[Dict[str, Any]]:
        buf = self.buffers.get(patient_id)
        if not buf:
            return None

        hrs, spo2s, sbps, dbps = [], [], [], []
        for _, vitals in buf:
            hr = _safe_float((vitals or {}).get("hr"))
            spo2 = _safe_float((vitals or {}).get("spo2"))
            sbp = _safe_float((vitals or {}).get("sbp"))
            dbp = _safe_float((vitals or {}).get("dbp"))
            if hr is not None:
                hrs.append(hr)
            if spo2 is not None:
                spo2s.append(spo2)
            if sbp is not None:
                sbps.append(sbp)
            if dbp is not None:
                dbps.append(dbp)

        n_samples = max(len(hrs), len(spo2s), len(sbps), len(dbps))
        if n_samples < int(self.cfg.min_samples):
            return None

        feats = {
            "patient_id": patient_id,
            "window_end": float(window_end_ts),
            "n_samples": int(n_samples),

            "hr_mean": _mean(hrs) or 0.0,
            "hr_std": _std(hrs) or 0.0,
            "hr_min": _min(hrs) or 0.0,
            "hr_max": _max(hrs) or 0.0,

            "spo2_mean": _mean(spo2s) or 0.0,
            "spo2_std": _std(spo2s) or 0.0,
            "spo2_min": _min(spo2s) or 0.0,
            "spo2_max": _max(spo2s) or 0.0,

            "sbp_mean": _mean(sbps) or 0.0,
            "sbp_std": _std(sbps) or 0.0,
            "sbp_min": _min(sbps) or 0.0,
            "sbp_max": _max(sbps) or 0.0,

            "dbp_mean": _mean(dbps) or 0.0,
            "dbp_std": _std(dbps) or 0.0,
            "dbp_min": _min(dbps) or 0.0,
            "dbp_max": _max(dbps) or 0.0,
        }

        return feats
