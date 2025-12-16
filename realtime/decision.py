# realtime/decision.py
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

import joblib


@dataclass
class Decision:
    patient_id: str
    ts: float
    risk: str
    confidence: float
    latency_ms: float
    reason: str
    features_used: int


class DecisionEngine:
    def __init__(self, model_path: str, metrics_json_path: str):
        self.model = joblib.load(model_path)

        with open(metrics_json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        # expected feature order
        self.feature_names: List[str] = meta["feature_names"]

        # optional mapping if exists
        self.class_names = meta.get("class_names")  # may be None

    def predict(self, features: Dict[str, Any]) -> Decision:
        t0 = time.time()

        pid = str(features.get("patient_id", "unknown"))
        ts = float(features.get("window_end_ts", time.time()))

        # hard guard
        n_samples = int(features.get("n_samples", 0))
        if n_samples < 5:
            return Decision(
                patient_id=pid,
                ts=ts,
                risk="unknown",
                confidence=0.0,
                latency_ms=0.0,
                reason="insufficient_data",
                features_used=0,
            )

        row = [float(features.get(name, 0.0)) for name in self.feature_names]

        # predict_proba if available
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba([row])[0]
            cls_idx = int(proba.argmax())
            conf = float(proba[cls_idx])
            pred = int(cls_idx)
        else:
            pred = int(self.model.predict([row])[0])
            conf = 1.0

        # map class index -> label
        if self.class_names and 0 <= pred < len(self.class_names):
            risk = str(self.class_names[pred])
        else:
            # fallback: keep numeric labels
            risk = str(pred)

        latency_ms = (time.time() - t0) * 1000.0

        return Decision(
            patient_id=pid,
            ts=ts,
            risk=risk,
            confidence=conf,
            latency_ms=latency_ms,
            reason="ok",
            features_used=len(self.feature_names),
        )
