import time
from typing import Dict, Tuple, Optional

from realtime.processor import WindowProcessor, WindowConfig
from realtime.validation import ValidationConfig, validate_and_sanitize


class EdgeNode:
    """
    Edge node: cleans raw events and produces information arrays.
    Supports per-patient sliding windows (important for multi-patient mode).
    """

    def __init__(self, window_cfg: Optional[WindowConfig] = None, vcfg: Optional[ValidationConfig] = None):
        self.window_cfg = window_cfg or WindowConfig(window_sec=30.0, min_samples=10)
        self.vcfg = vcfg or ValidationConfig(max_clock_skew_sec=5.0, clamp_values=True, drop_on_invalid=False)

        # Per-patient window processors
        self._wp: dict[int, WindowProcessor] = {}

        self.stats = {
            "in": 0,
            "dropped": 0,
            "late": 0,
            "corrected_fields": 0,
            "arrays_out": 0,
        }

    def _get_wp(self, patient_id: int) -> WindowProcessor:
        if patient_id not in self._wp:
            self._wp[patient_id] = WindowProcessor(self.window_cfg)
        return self._wp[patient_id]

    def ingest(self, event: Dict) -> Tuple[Dict | None, Dict]:
        """
        Ingest raw sensor event.
        Returns (information_array_or_none, meta).
        """
        self.stats["in"] += 1
        now_ts = time.time()

        clean, meta = validate_and_sanitize(event, now_ts=now_ts, cfg=self.vcfg)

        if meta["late"]:
            self.stats["late"] += 1
        self.stats["corrected_fields"] += int(meta["corrected_fields"])

        if clean is None:
            self.stats["dropped"] += 1
            return None, meta

        patient_id = int(clean["patient_id"])
        wp = self._get_wp(patient_id)

        wp.add_event(clean)
        if not wp.ready():
            return None, meta

        info = wp.aggregate()
        info["edge_ts"] = time.time()
        info["patient_id"] = patient_id

        self.stats["arrays_out"] += 1
        return info, meta
