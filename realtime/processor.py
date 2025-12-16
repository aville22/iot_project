from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional

import numpy as np


@dataclass
class WindowConfig:
    """
    Sliding window configuration for real-time processing.
    """
    window_sec: float = 30.0
    min_samples: int = 10
    reorder_slack_sec: float = 3.0  # how much out-of-order we tolerate


class WindowProcessor:
    """
    Maintains an event buffer and produces an "information array" aggregate.
    Handles mild out-of-order timestamps by keeping a reorder slack.
    """

    def __init__(self, cfg: Optional[WindowConfig] = None):
        self.cfg = cfg or WindowConfig()
        self.buffer: Deque[Dict] = deque()

    def add_event(self, event: Dict) -> None:
        self.buffer.append(event)

        # Keep buffer roughly ordered by timestamp (small buffer: insertion sort is OK)
        # This is a simple approach; for high throughput you'd use a heap or sorting batches.
        self._sort_buffer_if_needed()

        # Drop events outside the window (based on newest timestamp)
        if not self.buffer:
            return
        newest_ts = float(self.buffer[-1]["timestamp"])
        cutoff = newest_ts - self.cfg.window_sec

        while self.buffer and float(self.buffer[0]["timestamp"]) < cutoff:
            self.buffer.popleft()

    def _sort_buffer_if_needed(self) -> None:
        if len(self.buffer) < 2:
            return
        # If last event timestamp is earlier than previous, we might have out-of-order
        if float(self.buffer[-1]["timestamp"]) < float(self.buffer[-2]["timestamp"]):
            tmp = list(self.buffer)
            tmp.sort(key=lambda x: float(x["timestamp"]))
            self.buffer = deque(tmp)

    def ready(self) -> bool:
        return len(self.buffer) >= self.cfg.min_samples

    def aggregate(self) -> Dict:
        """
        Aggregate current window into information array features.
        """
        hr = np.array([e["heart_rate"] for e in self.buffer], dtype=float)
        spo2 = np.array([e["spo2"] for e in self.buffer], dtype=float)
        sbp = np.array([e["sbp"] for e in self.buffer], dtype=float)
        dbp = np.array([e["dbp"] for e in self.buffer], dtype=float)

        return {
            "n_samples": int(len(self.buffer)),
            "hr_mean": float(hr.mean()),
            "hr_std": float(hr.std(ddof=0)),
            "spo2_min": float(spo2.min()),
            "spo2_mean": float(spo2.mean()),
            "sbp_mean": float(sbp.mean()),
            "dbp_mean": float(dbp.mean()),
        }
