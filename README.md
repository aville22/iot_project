# IoT Real-Time Patient Monitoring (Edge → Cloud)

This project demonstrates a real-time IoT pipeline for monitoring patient vital signs
with fault-tolerant data ingestion, edge-level aggregation, and ML-based risk classification.

The system simulates multiple medical IoT devices streaming vital data,
injects realistic network faults (delay, loss, out-of-order),
aggregates events on the edge using event-time windows,
and performs machine learning inference in the cloud.

---

## Architecture Overview

IoT Sensors → Event Stream → Edge Node → Cloud Node → Alerts & Metrics

**Sensors**
- Generate vital signs (HR, SpO2, SBP, DBP)
- Simulate gradual patient condition changes
- Inject network faults:
  - packet loss
  - network delay
  - out-of-order events
  - corrupted measurements

**Edge Node**
- Validates and sanitizes incoming events
- Handles clock skew and invalid values
- Performs per-patient sliding window aggregation (event-time)
- Produces information arrays (feature vectors)

**Cloud Node**
- Runs ML inference (RandomForest classifier)
- Maintains per-patient alert state
- Applies confirmation and cooldown logic
- Logs alerts and performance metrics

---

## Machine Learning Model

- Model: RandomForestClassifier
- Training:
  - Supervised classification of patient risk level
  - Group-based split by `patient_id` to avoid data leakage
  - Semantic rule-based tags excluded from training
- Features:
  - Statistical aggregates over sliding windows:
    - mean, std, min, max for HR, SpO2, SBP, DBP
    - number of samples per window
- Feature schema and evaluation metrics are stored in `data/metrics.json`

---

## Demonstrated IoT Aspects

- Real-time streaming from multiple IoT devices
- Fault-tolerant ingestion under network issues
- Event-time windowing on the edge
- Edge-to-cloud separation of responsibilities
- Robust ML inference under noisy and incomplete data
- End-to-end latency measurement (average / p95)
- Multi-patient concurrent processing

---

## Project Structure
iot_project/
├── main_realtime_multi_demo.py # Multi-patient real-time demo
├── main_train_model.py # Model training
├── realtime/
│ ├── sensor.py # IoT sensor simulation + fault injection
│ ├── stream.py # Event stream abstraction
│ ├── validation.py # Event validation and sanitization
│ ├── processor.py # Sliding window aggregation (features)
│ ├── edge.py # Edge node logic
│ ├── decision.py # ML inference
│ ├── cloud.py # Cloud-side processing
│ ├── alerting.py # Alert generation
│ ├── metrics.py # Latency tracking
│ └── report.py # Run report generation
├── models/
│ └── train.py # Training pipeline
├── data/
│ ├── metrics.json # Feature schema and evaluation metrics
│ └── rf_model.joblib # Pretrained demo model
├── requirements.txt
└── README.md

---

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
2. (Optional) Train the model
python main_train_model.py
3. Run real-time multi-patient demo
python main_realtime_multi_demo.py
Outputs
Alert events: data/alert_events.jsonl
Run report: data/run_report.json
The run report includes:
number of processed events
number of produced information arrays
dropped and late events
average and p95 inference latency
end-to-end latency
number of emitted alerts
Notes
The provided model (rf_model.joblib) is a demo artifact.
It can be fully reproduced using the training script.
All real-time inference uses the same feature schema as training,
enforced via metrics.json.
Conclusion

This project demonstrates a complete IoT data pipeline with realistic constraints,
including unreliable networks, real-time processing, and edge-to-cloud ML inference.
It highlights practical challenges of IoT systems such as fault tolerance,
event-time processing, and performance monitoring.
