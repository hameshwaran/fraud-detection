"""
Trains the fraud detection system:

1. XGBoost classifier  -> supervised fraud probability, trained on SMOTE-balanced data
2. Isolation Forest    -> unsupervised anomaly score, trained only on normal transactions

Both models + the feature scaler are saved to backend/models/ for the API to load.
"""

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "transactions.csv"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

FEATURE_COLUMNS = [f"V{i+1}" for i in range(10)] + ["Amount", "Time"]
RANDOM_SEED = 42


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run `python data/generate_data.py` first."
        )
    return pd.read_csv(DATA_PATH)


def main():
    print("Loading dataset...")
    df = load_data()
    X = df[FEATURE_COLUMNS].copy()
    y = df["Class"].copy()

    print(f"Total rows: {len(df)} | Fraud rate: {y.mean() * 100:.3f}%")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_SEED
    )

    # --- Scale features (fit on train only) ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- SMOTE: balance the training set only ---
    print("Applying SMOTE oversampling to training data...")
    smote = SMOTE(random_state=RANDOM_SEED)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
    print(
        f"Post-SMOTE training distribution: "
        f"{np.bincount(y_train_res)} (0=normal, 1=fraud)"
    )

    # --- XGBoost classifier ---
    print("Training XGBoost classifier...")
    xgb_model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="aucpr",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    xgb_model.fit(X_train_res, y_train_res)

    # --- Isolation Forest: unsupervised anomaly detector ---
    # Trained only on normal (non-fraud) training transactions so it learns
    # what "normal" looks like and flags deviations as anomalies.
    print("Training Isolation Forest...")
    normal_mask = y_train.values == 0
    iso_forest = IsolationForest(
        n_estimators=200,
        contamination=float(y.mean()),  # expected fraud proportion
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    iso_forest.fit(X_train_scaled[normal_mask])

    # --- Evaluation on held-out test set ---
    print("\n=== Evaluation on test set ===")
    xgb_proba = xgb_model.predict_proba(X_test_scaled)[:, 1]
    xgb_pred = (xgb_proba >= 0.5).astype(int)

    print("\nXGBoost classification report:")
    print(classification_report(y_test, xgb_pred, digits=4, target_names=["Normal", "Fraud"]))
    roc_auc = roc_auc_score(y_test, xgb_proba)
    pr_auc = average_precision_score(y_test, xgb_proba)
    print(f"ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}")
    cm = confusion_matrix(y_test, xgb_pred)
    print(f"Confusion matrix [[TN,FP],[FN,TP]]:\n{cm}")

    # Isolation Forest: raw score_samples, more negative = more anomalous
    iso_scores = -iso_forest.score_samples(X_test_scaled)  # flip so higher = more anomalous
    iso_auc = roc_auc_score(y_test, iso_scores)
    print(f"\nIsolation Forest anomaly-score ROC-AUC: {iso_auc:.4f}")

    # --- Save artifacts ---
    joblib.dump(xgb_model, MODELS_DIR / "xgb_model.joblib")
    joblib.dump(iso_forest, MODELS_DIR / "isolation_forest.joblib")
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")

    metadata = {
        "feature_columns": FEATURE_COLUMNS,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "fraud_rate": float(y.mean()),
        "metrics": {
            "xgb_roc_auc": float(roc_auc),
            "xgb_pr_auc": float(pr_auc),
            "isolation_forest_roc_auc": float(iso_auc),
            "confusion_matrix": cm.tolist(),
        },
    }
    with open(MODELS_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved models + metadata to {MODELS_DIR}/")


if __name__ == "__main__":
    main()
