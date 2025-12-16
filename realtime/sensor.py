import asyncio
import time
from dataclasses import dataclass
from typing import AsyncIterator, Dict, Optional

import numpy as np

from generators.vitals_simulator import sample_vitals_from_severity


@dataclass
class FaultConfig:
    """
    Fault injection configuration for IoT streams.
    """
    dropout_prob: float = 0.02
    delay_prob: float = 0.03
    max_extra_delay_sec: float = 2.0
    out_of_order_prob: float = 0.02
    max_time_skew_sec: float = 3.0
    corruption_prob: float = 0.01


@dataclass
class SeverityConfig:
    """
    Dynamics for severity (latent patient state) in [0, 1].
    """
    start_mean: float = 0.35
    start_std: float = 0.10
    drift_std: float = 0.02              # random walk step per event
    worsen_prob: float = 0.01            # rare worsening event
    worsen_scale: float = 0.20           # jump magnitude
    improve_prob: float = 0.007          # rare recovery event
    improve_scale: float = 0.12          # jump magnitude


def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _corrupt_event(event: Dict) -> Dict:
    e = dict(event)
    fields = ["heart_rate", "spo2", "sbp", "dbp"]
    f = np.random.choice(fields)

    mode = np.random.choice(["nan", "spike", "negative"])
    if mode == "nan":
        e[f] = float("nan")
    elif mode == "spike":
        e[f] = float(e[f]) * np.random.uniform(2.0, 5.0)
    else:
        e[f] = -abs(float(e[f]))

    return e


async def sensor_stream(
    patient_id: int,
    interval_sec: float = 1.0,
    faults: Optional[FaultConfig] = None,
    severity_cfg: Optional[SeverityConfig] = None,
    initial_severity: Optional[float] = None,
) -> AsyncIterator[Dict]:
    """
    Async generator simulating a real-time IoT sensor with:
      - fault injection (dropout, delay, out-of-order, corruption)
      - dynamic severity (random walk + occasional jumps)
    """
    if faults is None:
        faults = FaultConfig()
    if severity_cfg is None:
        severity_cfg = SeverityConfig()

    if initial_severity is None:
        severity = float(np.random.normal(severity_cfg.start_mean, severity_cfg.start_std))
    else:
        severity = float(initial_severity)
    severity = _clip01(severity)

    while True:
        # Dropout (no event)
        if np.random.rand() < faults.dropout_prob:
            await asyncio.sleep(interval_sec)
            continue

        # Optional additional delay
        if np.random.rand() < faults.delay_prob:
            await asyncio.sleep(np.random.uniform(0.0, faults.max_extra_delay_sec))

        # Update latent severity (patient condition)
        severity += float(np.random.normal(0.0, severity_cfg.drift_std))
        if np.random.rand() < severity_cfg.worsen_prob:
            severity += float(np.random.exponential(severity_cfg.worsen_scale))
        if np.random.rand() < severity_cfg.improve_prob:
            severity -= float(np.random.exponential(severity_cfg.improve_scale))
        severity = _clip01(severity)

        hr, spo2, sbp, dbp = sample_vitals_from_severity(severity)

        ts = time.time()
        if np.random.rand() < faults.out_of_order_prob:
            ts = ts - np.random.uniform(0.1, faults.max_time_skew_sec)

        event = {
            "patient_id": patient_id,
            "timestamp": ts,
            "severity": severity,  # expose latent state for demo/analysis
            "heart_rate": hr,
            "spo2": spo2,
            "sbp": sbp,
            "dbp": dbp,
        }

        if np.random.rand() < faults.corruption_prob:
            event = _corrupt_event(event)

        yield event
        await asyncio.sleep(interval_sec)
