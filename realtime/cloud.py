import json
import time
from pathlib import Path
from typing import Dict, Optional

from realtime.decision import DecisionEngine
from realtime.alerting import AlertManager, AlertConfig


class CloudNode:
    """
    Cloud node: receives information arrays and performs inference + alerting.
    Supports per-patient alert state and optional JSONL logging of emitted alerts.
    """

    def __init__(
        self,
        decision_engine: DecisionEngine,
        alert_cfg: Optional[AlertConfig] = None,
        alert_log_path: Optional[str] = None,
    ):
        self.decision = decision_engine
        self.alert_cfg = alert_cfg or AlertConfig()

        # Per-patient alert managers (important for multi-patient mode too)
        self._alerts: dict[int, AlertManager] = {}

        self.alert_log_path = Path(alert_log_path) if alert_log_path else None
        if self.alert_log_path:
            self.alert_log_path.parent.mkdir(parents=True, exist_ok=True)

        self.stats = {
            "arrays_in": 0,
            "decisions": 0,
            "alerts_emitted": 0,
        }

    def _get_manager(self, patient_id: int) -> AlertManager:
        if patient_id not in self._alerts:
            self._alerts[patient_id] = AlertManager(self.alert_cfg)
        return self._alerts[patient_id]

    def _log_alert_event(self, record: Dict) -> None:
        if not self.alert_log_path:
            return
        with self.alert_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def handle(self, info: Dict) -> Dict:
        """
        Handle an information array from edge.
        Computes end-to-end latency (edge -> cloud decision).
        """
        self.stats["arrays_in"] += 1
        cloud_rx_ts = time.time()

        result = self.decision.decide(info)
        self.stats["decisions"] += 1

        patient_id = int(info.get("patient_id", -1))

        # Stateful alerting (anti-flapping + cooldown), per patient
        manager = self._get_manager(patient_id)
        alert_state = manager.update(result["risk_level"], now_ts=cloud_rx_ts)

        if alert_state["emitted"] is not None:
            self.stats["alerts_emitted"] += 1

            # Log alert events with a vitals snapshot (IoT-style)
            self._log_alert_event(
                {
                    "ts": cloud_rx_ts,
                    "patient_id": patient_id,
                    "emitted": alert_state["emitted"],
                    "prev_state": alert_state["prev_state"],
                    "state": alert_state["state"],
                    "risk_level": int(result["risk_level"]),
                    "vitals": {
                        "hr_mean": float(info.get("hr_mean", 0.0)),
                        "spo2_min": float(info.get("spo2_min", 0.0)),
                        "sbp_mean": float(info.get("sbp_mean", 0.0)),
                        "dbp_mean": float(info.get("dbp_mean", 0.0)),
                        "n_samples": int(info.get("n_samples", 0)),
                    },
                    "latency": {
                        "infer_ms": float(result["latency_ms"]),
                    },
                }
            )

        edge_ts = float(info.get("edge_ts", cloud_rx_ts))
        e2e_ms = (time.time() - edge_ts) * 1000.0

        return {
            **result,
            "e2e_ms": e2e_ms,
            "cloud_rx_ts": cloud_rx_ts,
            "alert_state": alert_state,
        }
