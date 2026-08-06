# Sentinel — Fraud Detection System

A complete, runnable fraud detection project combining a supervised model
(XGBoost, trained on SMOTE-balanced data) with an unsupervised anomaly
detector (Isolation Forest), served through a FastAPI backend and visualized
in a live React dashboard.

```
fraud-detection-system/
├── backend/
│   ├── data/
│   │   └── generate_data.py   # creates a synthetic, imbalanced transaction dataset
│   ├── train_model.py         # SMOTE + XGBoost + Isolation Forest training pipeline
│   ├── app.py                 # FastAPI service that serves the trained models
│   ├── requirements.txt
│   └── models/                # trained model artifacts land here (created by training)
└── frontend/
    └── index.html             # single-file React dashboard (no build step needed)
```

## How it works

1. **`generate_data.py`** creates a synthetic dataset of ~20,000 transactions
   with a realistic ~0.8% fraud rate (10 anonymized features `V1..V10`,
   `Amount`, `Time`, and a `Class` label), mirroring the structure of the
   classic Kaggle credit-card-fraud dataset.
2. **`train_model.py`**:
   - Splits the data into train/test (stratified, so both sets keep the same
     fraud ratio).
   - Scales features with `StandardScaler`.
   - Applies **SMOTE** to the *training* set only, so the XGBoost model sees
     a balanced 50/50 class distribution during training without leaking
     synthetic samples into the test set.
   - Trains an **XGBoost** classifier for supervised fraud probability.
   - Trains an **Isolation Forest** on normal transactions only, so it learns
     what "normal" looks like and flags statistical outliers — this catches
     fraud patterns the supervised model wasn't explicitly trained on.
   - Saves both models, the scaler, and evaluation metrics to `models/`.
3. **`app.py`** loads the trained models and exposes a REST API to score
   transactions in real time, log recent scores, and report dashboard stats.
4. **`frontend/index.html`** is a self-contained React dashboard (loaded via
   CDN, no npm/build step) that lets you submit a transaction, run a
   simulated batch, and watch a live-scored transaction feed with
   fraud probability, anomaly flags, and a risk level (MINIMAL/LOW/MEDIUM/HIGH).

## Setup & run

### 1. Backend

```bash
cd backend
python3 -m venv venv && source venv/bin/activate   # optional but recommended
pip install -r requirements.txt

# Generate the synthetic dataset
python data/generate_data.py

# Train the models (prints evaluation metrics: ROC-AUC, PR-AUC, confusion matrix)
python train_model.py

# Start the API
uvicorn app:app --reload --port 8000
```

The API will be live at `http://localhost:8000`. Interactive docs (Swagger UI)
are automatically available at `http://localhost:8000/docs`.

### 2. Frontend

No build tooling required — just open the file:

```bash
open frontend/index.html      # macOS
xdg-open frontend/index.html  # Linux
```

Or serve it with any static server, e.g. `python3 -m http.server` from the
`frontend/` folder. The dashboard talks to the API at `http://localhost:8000`
by default. To point it elsewhere, set `window.FRAUD_API_BASE` before the
main script runs, e.g. add to `index.html`:

```html
<script>window.FRAUD_API_BASE = "https://your-api-host";</script>
```

## API reference

| Method | Endpoint                | Description                                   |
|--------|--------------------------|------------------------------------------------|
| GET    | `/health`                | Model load status                              |
| GET    | `/metadata`               | Training metrics (ROC-AUC, PR-AUC, etc.)       |
| POST   | `/predict`                 | Score a single transaction                     |
| POST   | `/predict_batch`           | Score a list of transactions                   |
| POST   | `/simulate?n=15`           | Generate & score `n` random demo transactions |
| GET    | `/transactions/recent`     | Recently scored transactions (in-memory log)  |
| GET    | `/stats`                   | Aggregate dashboard stats                      |

Example request:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"V1":2.1,"V2":1.9,"V3":2.4,"V4":1.8,"V5":2.0,"V6":2.2,"V7":1.7,"V8":2.1,"V9":1.9,"V10":2.3,"Amount":920,"Time":3600}'
```

## Notes & next steps

- The dataset here is **synthetic** (generated locally, no download needed).
  To use the real Kaggle "Credit Card Fraud Detection" dataset instead, drop
  a `creditcard.csv` with the same column structure (`V1..V28`, `Amount`,
  `Time`, `Class`) into `backend/data/`, adjust `FEATURE_COLUMNS` in
  `train_model.py` and `app.py` to match, and re-run `train_model.py`.
- The recent-transactions log and stats in `app.py` are in-memory (reset on
  restart) to keep the project dependency-free. For production, swap in a
  real database (Postgres, Redis, etc.).
- The risk-level thresholds in `app.py` (`_risk_level`) are simple starting
  heuristics — tune them against your own precision/recall requirements.
