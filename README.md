# IoT Sensor Fault Detector — Edge-Constrained Predictive Maintenance

A machine failure prediction system built to run under strict memory
constraints, simulating deployment on low-resource edge hardware (e.g. a
Raspberry Pi) rather than a full cloud server.

**Live demo:** https://iot-fault-monitor-production.up.railway.app

## Problem

Factories rely on sensor readings (temperature, torque, rotational speed,
tool wear) to catch equipment failure before it happens. In practice, this
often needs to run close to the machine itself — with unreliable internet
and cheap hardware — rather than round-tripping every reading to the cloud.
This project builds and deploys a failure-prediction model designed
specifically for that constraint, rather than assuming unlimited compute.

## Dataset

[AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)
— 10,000 readings, ~3.4% failure rate. Features: product type, air/process
temperature, rotational speed, torque, tool wear.

## Approach

**1. Baseline model** — RandomForest (200 trees), handling class imbalance
with `class_weight='balanced'`.

**2. Model shrinking** — compared several smaller configurations to fit
edge-hardware memory limits:

| Model | Size | Precision | Recall | F1 |
|---|---|---|---|---|
| Baseline (200 trees) | 6,330 KB | 0.88 | 0.53 | 0.66 |
| **Deployed (20 trees, depth 6, compressed)** | **56 KB** | 0.38 | 0.82 | 0.52 |

**113x size reduction**, with recall actually *improving* — the smaller
model catches more real failures, at the cost of more false positives (an
acceptable trade for fault detection, where missing a real failure is
costlier than a false alarm).

**3. Constrained deployment** — packaged in Docker and tested under a hard
memory cap:
- `--memory=128m` → runs successfully
- `--memory=40m` → OOM-killed (exit code 137) — confirms the real minimum
  viable memory footprint sits between these two values

**4. Serving** — FastAPI backend exposes `/predict` and `/metrics`
(per-request latency and memory usage), plus serves a live dashboard —
sliders and charts that hit the API in real time and plot latency/memory
per request.

## Tech stack

Python · scikit-learn · pandas · FastAPI · Docker · Chart.js · vanilla JS/HTML/CSS

## Project structure

```
fault-detector-project/
├── notebooks/
│   ├── fault_detector_eda_and_model.ipynb   # EDA + baseline model
│   └── phase2_model_shrinking.ipynb          # size/accuracy trade-off comparison
├── app/
│   ├── main.py                    # FastAPI app (predict, metrics, health, dashboard)
│   ├── dashboard.html             # live monitoring UI, served by the API itself
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── fault_detector_constrained.pkl
│   └── model_config.txt           # decision threshold
└── README.md
```

## Running locally

```bash
cd app
docker build -t fault-detector .
docker run --memory=128m -p 8000:8000 fault-detector
```

Open `http://localhost:8000` for the dashboard, or call the API directly:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "machine_type": "L",
    "air_temperature_k": 302.0,
    "process_temperature_k": 312.0,
    "rotational_speed_rpm": 1350,
    "torque_nm": 65,
    "tool_wear_min": 240
  }'
```

## API

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Serves the live dashboard |
| `/predict` | POST | Returns failure prediction, probability, latency (ms), memory (KB) |
| `/metrics` | GET | Running totals: request count, avg latency, current memory, model size |
| `/health` | GET | Basic status check |

## Notes

- `resource.getrusage().ru_maxrss` (used for the memory readings in
  `/predict` and `/metrics`) reports the whole process's peak memory, which
  can read higher than Docker's own `--memory` cgroup accounting — a known
  discrepancy between Python's stdlib measurement and Docker's enforcement,
  not a bug in the app itself.
- The `fault_detector_baseline.pkl` (6.3 MB, unshrunk) is kept out of the
  deployed app on purpose — it's a reference point from the notebooks, not
  what actually runs in production.
