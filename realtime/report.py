import json
import time
from pathlib import Path
from typing import Dict


class RunReporter:
    """
    Collects and saves a final IoT run summary.
    """

    def __init__(self, out_path: str):
        self.out_path = Path(out_path)
        self.start_ts = time.time()

    def build(
        self,
        patients: int,
        edge_stats: Dict,
        infer_stats: Dict,
        e2e_stats: Dict,
        alerts_emitted: int,
    ) -> Dict:
        runtime_sec = time.time() - self.start_ts

        raw_events = edge_stats.get("in", 0)
        arrays = edge_stats.get("arrays_out", 0)
        compression = (raw_events / arrays) if arrays > 0 else None

        return {
            "runtime_sec": runtime_sec,
            "patients": patients,
            "edge": {
                **edge_stats,
                "compression_ratio": compression,
            },
            "latency": {
                "infer": infer_stats,
                "e2e": e2e_stats,
            },
            "alerts_emitted": alerts_emitted,
            "generated_at": time.time(),
        }

    def save(self, report: Dict) -> None:
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        with self.out_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
