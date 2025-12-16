from collections import deque
from typing import Dict
import numpy as np


class LatencyTracker:
    """
    Tracks recent latency measurements and computes summary statistics.
    """

    def __init__(self, size: int = 200):
        self.latencies = deque(maxlen=size)

    def add(self, latency_ms: float) -> None:
        self.latencies.append(float(latency_ms))

    def stats(self) -> Dict[str, float]:
        if not self.latencies:
            return {}

        arr = np.array(self.latencies, dtype=float)
        return {
            "avg_ms": float(arr.mean()),
            "min_ms": float(arr.min()),
            "max_ms": float(arr.max()),
            "p95_ms": float(np.percentile(arr, 95)),
            "count": int(arr.size),
        }
