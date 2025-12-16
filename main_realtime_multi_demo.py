import asyncio
import random
import time

from realtime.sensor import sensor_stream, FaultConfig, SeverityConfig
from realtime.stream import EventStream
from realtime.processor import WindowConfig
from realtime.decision import DecisionEngine
from realtime.metrics import LatencyTracker
from realtime.validation import ValidationConfig

from realtime.edge import EdgeNode
from realtime.cloud import CloudNode
from realtime.alerting import AlertConfig
from realtime.report import RunReporter


async def main():
    N_PATIENTS = 10  # increase to 50 if you want
    stream = EventStream(maxsize=20000)

    # -------------------------
    # EDGE (per-patient windows)
    # -------------------------
    wcfg = WindowConfig(window_sec=30.0, min_samples=10)
    vcfg = ValidationConfig(max_clock_skew_sec=5.0, clamp_values=True, drop_on_invalid=False)
    edge = EdgeNode(window_cfg=wcfg, vcfg=vcfg)

    # -------------------------
    # CLOUD (per-patient alert state + JSONL logging)
    # -------------------------
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
        alert_log_path="data/alert_events.jsonl",
    )

    # -------------------------
    # METRICS
    # -------------------------
    infer_lat = LatencyTracker(size=500)
    e2e_lat = LatencyTracker(size=500)
    reporter = RunReporter("data/run_report.json")

    # -------------------------
    # FAULT INJECTION
    # -------------------------
    fcfg = FaultConfig(
        dropout_prob=0.02,
        delay_prob=0.03,
        out_of_order_prob=0.02,
        corruption_prob=0.02,
        max_extra_delay_sec=2.0,
        max_time_skew_sec=3.0,
    )

    # -------------------------
    # DYNAMIC SEVERITY (tuned for visible transitions)
    # -------------------------
    scfg = SeverityConfig(
        start_mean=0.25,
        start_std=0.10,
        drift_std=0.03,
        worsen_prob=0.008,
        worsen_scale=0.18,
        improve_prob=0.014,
        improve_scale=0.22,
    )

    async def producer(pid: int):
        start_sev = random.uniform(0.10, 0.45)
        async for event in sensor_stream(
            patient_id=pid,
            interval_sec=1.0,
            faults=fcfg,
            severity_cfg=scfg,
            initial_severity=start_sev,
        ):
            await stream.publish(event)

    async def consumer():
        last_print = time.time()
        while True:
            event = await stream.consume()

            info, _ = edge.ingest(event)
            if info is None:
                continue

            # Simulate network hop edge -> cloud
            await asyncio.sleep(0.02)

            result = cloud.handle(info)
            infer_lat.add(result["latency_ms"])
            e2e_lat.add(result["e2e_ms"])

            now = time.time()
            if now - last_print >= 2.0:
                il = infer_lat.stats() or {"avg_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
                el = e2e_lat.stats() or {"avg_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}

                arrays = max(1, edge.stats["arrays_out"])
                compression = edge.stats["in"] / arrays

                print(
                    f"[SUMMARY] patients={N_PATIENTS} "
                    f"in={edge.stats['in']} arrays={edge.stats['arrays_out']} "
                    f"dropped={edge.stats['dropped']} late={edge.stats['late']} corrected={edge.stats['corrected_fields']} | "
                    f"compression={compression:.2f}x | "
                    f"infer_avg={il['avg_ms']:.2f}ms p95={il['p95_ms']:.2f}ms max={il['max_ms']:.2f}ms | "
                    f"e2e_avg={el['avg_ms']:.2f}ms p95={el['p95_ms']:.2f}ms max={el['max_ms']:.2f}ms | "
                    f"alerts={cloud.stats.get('alerts_emitted', 0)}"
                )
                last_print = now

    producer_tasks = [asyncio.create_task(producer(pid=i + 1)) for i in range(N_PATIENTS)]
    consumer_task = asyncio.create_task(consumer())

    try:
        # Wait for tasks forever until cancelled (Ctrl+C)
        await asyncio.gather(*producer_tasks, consumer_task)
    except (asyncio.CancelledError, KeyboardInterrupt):
        # On Windows, Ctrl+C often results in CancelledError inside asyncio tasks.
        pass
    finally:
        # Always cancel tasks
        for t in producer_tasks:
            t.cancel()
        consumer_task.cancel()

        # Let tasks finish cancellation cleanly
        await asyncio.gather(*producer_tasks, consumer_task, return_exceptions=True)

        # Always save final report
        report = reporter.build(
            patients=N_PATIENTS,
            edge_stats=edge.stats,
            infer_stats=infer_lat.stats(),
            e2e_stats=e2e_lat.stats(),
            alerts_emitted=cloud.stats.get("alerts_emitted", 0),
        )
        reporter.save(report)

        print("[REPORT] Saved final report to data/run_report.json")
        print("[REPORT] Alert events are in data/alert_events.jsonl")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # If KeyboardInterrupt happens outside the event loop
        pass
