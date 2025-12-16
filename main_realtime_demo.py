import asyncio
import time

from realtime.sensor import sensor_stream, FaultConfig, SeverityConfig
from realtime.stream import EventStream
from realtime.processor import WindowProcessor, WindowConfig
from realtime.decision import DecisionEngine
from realtime.metrics import LatencyTracker
from realtime.validation import ValidationConfig

from realtime.edge import EdgeNode
from realtime.cloud import CloudNode
from realtime.alerting import AlertConfig


async def main():
    stream = EventStream(maxsize=5000)

    # EDGE
    wp = WindowProcessor(WindowConfig(window_sec=30.0, min_samples=10))
    vcfg = ValidationConfig(max_clock_skew_sec=5.0, clamp_values=True, drop_on_invalid=False)
    edge = EdgeNode(window_processor=wp, vcfg=vcfg)

    # CLOUD
    decision = DecisionEngine(model_path="data/rf_model.joblib", metrics_path="data/metrics.json")
    alert_cfg = AlertConfig(
        warn_level=1,
        crit_level=2,
        confirm_warn=2,
        confirm_crit=2,
        confirm_clear=3,
        cooldown_sec=10.0,
    )
    cloud = CloudNode(
        decision_engine=decision,
        alert_cfg=alert_cfg,
        alert_log_path="data/alert_events.jsonl",  # A) JSONL logging
    )

    # METRICS
    infer_lat = LatencyTracker(size=200)
    e2e_lat = LatencyTracker(size=200)

    # FAULTS
    fcfg = FaultConfig(
        dropout_prob=0.02,
        delay_prob=0.03,
        out_of_order_prob=0.02,
        corruption_prob=0.02,
        max_extra_delay_sec=2.0,
        max_time_skew_sec=3.0,
    )

    # B) TUNED severity dynamics for visible transitions
    scfg = SeverityConfig(
        start_mean=0.25,
        start_std=0.08,
        drift_std=0.03,
        worsen_prob=0.008,
        worsen_scale=0.18,
        improve_prob=0.014,
        improve_scale=0.22,
    )

    async def producer():
        async for event in sensor_stream(
            patient_id=1,
            interval_sec=1.0,
            faults=fcfg,
            severity_cfg=scfg,
            initial_severity=0.25,
        ):
            await stream.publish(event)

    async def consumer():
        while True:
            event = await stream.consume()
            sev = float(event.get("severity", 0.0))

            info, _ = edge.ingest(event)
            if info is None:
                continue

            # Add patient_id into info arrays explicitly (for logging & multi-patient later)
            info["patient_id"] = int(event["patient_id"])

            # Simulate network hop (edge -> cloud)
            await asyncio.sleep(0.02)

            result = cloud.handle(info)

            infer_lat.add(result["latency_ms"])
            e2e_lat.add(result["e2e_ms"])

            il = infer_lat.stats() or {"avg_ms": 0.0}
            el = e2e_lat.stats() or {"avg_ms": 0.0}

            hr_mean = float(info.get("hr_mean", 0.0))
            spo2_min = float(info.get("spo2_min", 0.0))
            sbp_mean = float(info.get("sbp_mean", 0.0))
            dbp_mean = float(info.get("dbp_mean", 0.0))
            n_samples = int(info.get("n_samples", 0))

            ast = result.get("alert_state", {})
            state = ast.get("state", "-")
            emitted = ast.get("emitted") or "-"

            print(
                f"sev={sev:.2f} | "
                f"HR_mean={hr_mean:.1f} | SpO2_min={spo2_min:.1f} | "
                f"SBP_mean={sbp_mean:.1f} | DBP_mean={dbp_mean:.1f} | n={n_samples} || "
                f"risk={result['risk_level']} state={state} emitted={emitted} "
                f"infer_ms={result['latency_ms']:.2f} infer_avg={il['avg_ms']:.2f} "
                f"e2e_ms={result['e2e_ms']:.2f} e2e_avg={el['avg_ms']:.2f} | "
                f"edge_in={edge.stats['in']} dropped={edge.stats['dropped']} late={edge.stats['late']} "
                f"corrected={edge.stats['corrected_fields']} arrays={edge.stats['arrays_out']} "
                f"alerts={cloud.stats.get('alerts_emitted', 0)}"
            )

    await asyncio.gather(producer(), consumer())


if __name__ == "__main__":
    asyncio.run(main())
