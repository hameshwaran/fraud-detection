"""
Generates a synthetic, imbalanced credit-card-style transaction dataset.

Mimics the structure of the classic "Kaggle credit card fraud" dataset:
- Time, Amount + anonymized PCA-like features (V1..V10)
- ~0.8% fraud rate (realistic imbalance)
- Fraud transactions have subtly different statistical patterns
"""

import numpy as np
import pandas as pd
from pathlib import Path

RANDOM_SEED = 42
N_SAMPLES = 20000
FRAUD_RATIO = 0.008  # ~0.8% fraud, similar to real-world imbalance

OUT_PATH = Path(__file__).parent / "transactions.csv"


def generate_dataset(n_samples=N_SAMPLES, fraud_ratio=FRAUD_RATIO, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)

    n_fraud = int(n_samples * fraud_ratio)
    n_normal = n_samples - n_fraud

    n_features = 10  # V1..V10, anonymized/engineered features

    # --- Normal transactions ---
    normal_features = rng.normal(loc=0.0, scale=1.0, size=(n_normal, n_features))
    normal_amount = np.abs(rng.normal(loc=60, scale=45, size=n_normal))
    normal_amount = np.clip(normal_amount, 0.5, 2000)
    normal_time = rng.integers(0, 172800, size=n_normal)  # seconds over 2 days
    normal_labels = np.zeros(n_normal, dtype=int)

    # --- Fraudulent transactions (shifted distribution = anomalous pattern) ---
    fraud_features = rng.normal(loc=2.2, scale=1.8, size=(n_fraud, n_features))
    # Fraud tends to cluster at odd hours + has bimodal amount behavior
    fraud_amount_small = np.abs(rng.normal(loc=8, scale=4, size=n_fraud // 2))
    fraud_amount_large = np.abs(rng.normal(loc=850, scale=300, size=n_fraud - n_fraud // 2))
    fraud_amount = np.concatenate([fraud_amount_small, fraud_amount_large])
    rng.shuffle(fraud_amount)
    fraud_amount = np.clip(fraud_amount, 0.5, 5000)
    # Bias towards late-night hours (0-5am and 11pm-midnight)
    night_seconds = rng.choice(
        np.concatenate([np.arange(0, 18000), np.arange(158400, 172800)]),
        size=n_fraud,
    )
    fraud_time = night_seconds
    fraud_labels = np.ones(n_fraud, dtype=int)

    # --- Combine ---
    features = np.vstack([normal_features, fraud_features])
    amount = np.concatenate([normal_amount, fraud_amount])
    time = np.concatenate([normal_time, fraud_time])
    labels = np.concatenate([normal_labels, fraud_labels])

    columns = [f"V{i+1}" for i in range(n_features)]
    df = pd.DataFrame(features, columns=columns)
    df["Amount"] = amount
    df["Time"] = time
    df["Class"] = labels

    # Shuffle rows
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    return df


if __name__ == "__main__":
    df = generate_dataset()
    df.to_csv(OUT_PATH, index=False)
    fraud_count = int(df["Class"].sum())
    print(f"Generated {len(df)} transactions -> {OUT_PATH}")
    print(f"Fraud: {fraud_count} ({fraud_count / len(df) * 100:.2f}%)")
    print(f"Normal: {len(df) - fraud_count}")
