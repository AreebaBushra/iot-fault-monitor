import time
import os
import resource
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI(title="IoT Sensor Fault Detector (Edge-Constrained)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "fault_detector_constrained.pkl"
CONFIG_PATH = "model_config.txt"

model = joblib.load(MODEL_PATH)

with open(CONFIG_PATH) as f:
    line = f.read().strip()
    THRESHOLD = float(line.split("=")[1])

TYPE_MAP = {"L": 0, "M": 1, "H": 2}

# running totals for the /metrics endpoint
request_count = 0
total_latency_ms = 0.0


class SensorReading(BaseModel):
    machine_type: Literal["L", "M", "H"]
    air_temperature_k: float
    process_temperature_k: float
    rotational_speed_rpm: float
    torque_nm: float
    tool_wear_min: float


class PredictionResponse(BaseModel):
    failure_predicted: bool
    failure_probability: float
    latency_ms: float
    memory_kb: float


def get_memory_kb() -> float:
    # Peak resident set size for this process, in KB (Linux).
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


@app.get("/", response_class=HTMLResponse)
def root():
    with open("dashboard.html") as f:
        return f.read()


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_PATH, "threshold": THRESHOLD}


@app.post("/predict", response_model=PredictionResponse)
def predict(reading: SensorReading):
    global request_count, total_latency_ms

    start = time.perf_counter()

    features = pd.DataFrame([{
        "Type": TYPE_MAP[reading.machine_type],
        "Air temperature [K]": reading.air_temperature_k,
        "Process temperature [K]": reading.process_temperature_k,
        "Rotational speed [rpm]": reading.rotational_speed_rpm,
        "Torque [Nm]": reading.torque_nm,
        "Tool wear [min]": reading.tool_wear_min,
    }])

    proba = model.predict_proba(features)[0][1]
    prediction = bool(proba >= THRESHOLD)

    latency_ms = (time.perf_counter() - start) * 1000
    memory_kb = get_memory_kb()

    request_count += 1
    total_latency_ms += latency_ms

    return PredictionResponse(
        failure_predicted=prediction,
        failure_probability=round(float(proba), 4),
        latency_ms=round(latency_ms, 3),
        memory_kb=round(memory_kb, 1),
    )


@app.get("/metrics")
def metrics():
    avg_latency = (total_latency_ms / request_count) if request_count else 0.0
    return {
        "request_count": request_count,
        "avg_latency_ms": round(avg_latency, 3),
        "current_memory_kb": round(get_memory_kb(), 1),
        "model_file_size_kb": round(os.path.getsize(MODEL_PATH) / 1024, 2),
        "decision_threshold": THRESHOLD,
    }
