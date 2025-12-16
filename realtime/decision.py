from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List

import joblib
import pandas as pd


class DecisionEngine:
    """
    Real-time decision engine using a trained model.
    Builds a pandas DataFrame with correct feature names to avoid sklearn warnings.
    """

    def __init__(self, model_path: str, metrics_path: str):
        self.model = joblib.load(model_path)

        m = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
        self.feature_names: List[str] = list(m.get("feature_names", []))
        if not self.feature_names:
            raise ValueError("feature_names not found in metrics.json; cannot align realtime features.")

    def decide(self, features: Dict) -> Dict:
        start = time.time()

        row = {name: float(features.get(name, 0.0)) for name in self.feature_names}
        X = pd.DataFrame([row], columns=self.feature_names)

        risk = int(self.model.predict(X)[0])
        latency_ms = (time.time() - start) * 1000.0

        return {
            "risk_level": risk,
            "alert": risk >= 2,
            "latency_ms": latency_ms,
        }
