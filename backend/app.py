"""
FastAPI service for the Fraud Detection System.

Endpoints:
    GET  /health              -> service + model status
    GET  /metadata             -> training metadata / metrics
    POST /predict              -> score a single transaction
    POST /predict_batch        -> score a list of transactions
    GET  /transactions/recent  -> recent scored transactions (in-memory log)
    GET  /stats                -> aggregate dashboard stats
"""

import json
import time
import uuid
from collections import deque
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"

FEATURE_COLUMNS = [f"V{i+1}" for i in range(10)] + ["Amount", "Time"]

app = FastAPI(title="Fraud Detection API", version="1.0.0")

# Allow the React dashboard (served separately / opened as a file) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- In-memory recent-transaction log for the dashboard (simple, no DB needed) ---
RECENT_LOG = deque(maxlen=200)
STATS = {"total_scored": 0, "flagged_fraud": 0, "flagged_anomaly": 0}

# --- Load models at startup ---
xgb_model = None
iso_forest = None
scaler = None
metadata = {}


@app.on_event("startup")
def load_models():
    global xgb_model, iso_forest, scaler, metadata
    try:
        xgb_model = joblib.load(MODELS_DIR / "xgb_model.joblib")
        iso_forest = joblib.load(MODELS_DIR / "isolation_forest.joblib")
        scaler = joblib.load(MODELS_DIR / "scaler.joblib")
        meta_path = MODELS_DIR / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                metadata = json.load(f)
    except FileNotFoundError:
        # Models not trained yet; API will report unhealthy until `train_model.py` is run.
        pass


class Transaction(BaseModel):
    V1: float = 0.0
    V2: float = 0.0
    V3: float = 0.0
    V4: float = 0.0
    V5: float = 0.0
    V6: float = 0.0
    V7: float = 0.0
    V8: float = 0.0
    V9: float = 0.0
    V10: float = 0.0
    Amount: float = Field(..., ge=0)
    Time: float = Field(..., ge=0)


class PredictionResult(BaseModel):
    id: str
    fraud_probability: float
    anomaly_score: float
    is_fraud_predicted: bool
    is_anomaly: bool
    risk_level: str
    timestamp: float


def _models_ready() -> bool:
    return xgb_model is not None and iso_forest is not None and scaler is not None


def _risk_level(fraud_proba: float, is_anomaly: bool) -> str:
    if fraud_proba >= 0.85 or (fraud_proba >= 0.5 and is_anomaly):
        return "HIGH"
    if fraud_proba >= 0.5 or is_anomaly:
        return "MEDIUM"
    if fraud_proba >= 0.2:
        return "LOW"
    return "MINIMAL"


def _score_transaction(tx: Transaction) -> PredictionResult:
    if not _models_ready():
        raise HTTPException(
            status_code=503,
            detail="Models not trained yet. Run `python train_model.py` first.",
        )

    row = np.array([[getattr(tx, col) for col in FEATURE_COLUMNS]])
    scaled = scaler.transform(row)

    fraud_proba = float(xgb_model.predict_proba(scaled)[0, 1])
    anomaly_raw = float(-iso_forest.score_samples(scaled)[0])  # higher = more anomalous
    is_anomaly = bool(iso_forest.predict(scaled)[0] == -1)

    result = PredictionResult(
        id=str(uuid.uuid4())[:8],
        fraud_probability=round(fraud_proba, 4),
        anomaly_score=round(anomaly_raw, 4),
        is_fraud_predicted=fraud_proba >= 0.5,
        is_anomaly=is_anomaly,
        risk_level=_risk_level(fraud_proba, is_anomaly),
        timestamp=time.time(),
    )

    # Log for dashboard
    STATS["total_scored"] += 1
    if result.is_fraud_predicted:
        STATS["flagged_fraud"] += 1
    if result.is_anomaly:
        STATS["flagged_anomaly"] += 1

    entry = tx.dict()
    entry.update(result.dict())
    RECENT_LOG.appendleft(entry)

    return result


@app.get("/health")
def health():
    return {
        "status": "ok" if _models_ready() else "models_not_trained",
        "models_loaded": _models_ready(),
    }


@app.get("/metadata")
def get_metadata():
    if not metadata:
        raise HTTPException(status_code=404, detail="No training metadata found yet.")
    return metadata


@app.post("/predict", response_model=PredictionResult)
def predict(tx: Transaction):
    return _score_transaction(tx)


@app.post("/predict_batch", response_model=List[PredictionResult])
def predict_batch(txs: List[Transaction]):
    return [_score_transaction(tx) for tx in txs]


@app.get("/transactions/recent")
def recent_transactions(limit: int = 25):
    return list(RECENT_LOG)[:limit]


@app.get("/stats")
def stats():
    return {
        **STATS,
        "fraud_rate_scored": (
            STATS["flagged_fraud"] / STATS["total_scored"] if STATS["total_scored"] else 0
        ),
    }


@app.post("/simulate")
def simulate(n: int = 20, seed: Optional[int] = None):
    """Generate n random synthetic transactions and score them (for demo purposes)."""
    rng = np.random.default_rng(seed)
    results = []
    for _ in range(n):
        is_fraud_like = rng.random() < 0.15
        if is_fraud_like:
            vals = rng.normal(2.2, 1.8, size=10)
            amount = float(np.clip(rng.choice([rng.normal(8, 4), rng.normal(850, 300)]), 0.5, 5000))
            t = float(rng.choice(np.concatenate([np.arange(0, 18000), np.arange(158400, 172800)])))
        else:
            vals = rng.normal(0, 1, size=10)
            amount = float(np.clip(rng.normal(60, 45), 0.5, 2000))
            t = float(rng.integers(0, 172800))
        tx_dict = {f"V{i+1}": float(vals[i]) for i in range(10)}
        tx_dict["Amount"] = amount
        tx_dict["Time"] = t
        tx = Transaction(**tx_dict)
        results.append(_score_transaction(tx))
    return results


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
