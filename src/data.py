"""Data loading, chronological splitting, feature engineering, and scaling.

CRITICAL INTEGRITY RULES:
- CHRONOLOGICAL split (sort by Time): 60% Train, 20% Validation, 20% Test.
- NEVER shuffle randomly (prevents temporal data leakage).
- Training set for anomaly models = LEGITIMATE rows of the train split only (Class == 0).
- StandardScaler is fitted STRICTLY on legitimate train rows only.
- Validation set is used solely for threshold selection and hyperparameter tuning.
- Test set remains completely untouched until final model evaluation.
"""

from typing import Dict, Any, Tuple
import os
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch


def set_seed(seed: int = 42) -> None:
    """Set global random seeds for full reproducibility across libraries."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_raw_data(filepath: str = "data/raw/creditcard.csv") -> pd.DataFrame:
    """Load credit card dataset from CSV and validate structure."""
    if not os.path.exists(filepath):
        alt_path = "data/creditcard.csv"
        if os.path.exists(alt_path):
            filepath = alt_path
        else:
            raise FileNotFoundError(
                f"Dataset not found at {filepath} or {alt_path}."
            )
    df = pd.read_csv(filepath)
    expected_cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
    for col in expected_cols:
        if col not in df.columns:
            raise ValueError(f"Missing expected column in dataset: {col}")
    return df


def get_dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute high-level exploratory statistics for dataset."""
    total_rows = len(df)
    missing_vals = int(df.isnull().sum().sum())
    duplicate_rows = int(df.duplicated().sum())
    fraud_count = int((df["Class"] == 1).sum())
    legit_count = int((df["Class"] == 0).sum())
    base_rate = fraud_count / total_rows

    amount_stats = {
        "legit_mean": float(df[df["Class"] == 0]["Amount"].mean()),
        "legit_median": float(df[df["Class"] == 0]["Amount"].median()),
        "legit_max": float(df[df["Class"] == 0]["Amount"].max()),
        "fraud_mean": float(df[df["Class"] == 1]["Amount"].mean()),
        "fraud_median": float(df[df["Class"] == 1]["Amount"].median()),
        "fraud_max": float(df[df["Class"] == 1]["Amount"].max()),
    }

    time_span_hours = float((df["Time"].max() - df["Time"].min()) / 3600.0)

    return {
        "total_rows": total_rows,
        "n_features": df.shape[1],
        "missing_values": missing_vals,
        "duplicate_rows": duplicate_rows,
        "legit_count": legit_count,
        "fraud_count": fraud_count,
        "fraud_base_rate": base_rate,
        "amount_stats": amount_stats,
        "time_span_hours": time_span_hours,
    }


def prepare_data(
    filepath: str = "data/raw/creditcard.csv",
    train_frac: float = 0.60,
    val_frac: float = 0.20,
    test_frac: float = 0.20,
    seed: int = 42,
) -> Dict[str, Any]:
    """Execute leakage-free chronological splitting, feature engineering, and scaling.

    Features:
    - V1 through V28 (PCA anonymized).
    - Amount_log1p: log1p(Amount) to compress heavy right-tail skewness.
    - Time is strictly used for ordering/chronological partitioning and excluded
      from feature inputs to avoid temporal overfitting over the 48-hour recording window.

    Scaling:
    - StandardScaler fitted ONLY on legitimate training samples (Class == 0).
    - Applied (transform only) to validation and test partitions.
    """
    set_seed(seed)
    df = load_raw_data(filepath)

    # 1. Chronological sorting
    df_sorted = df.sort_values("Time").reset_index(drop=True)

    # 2. Compute log1p amount
    df_sorted["Amount_log1p"] = np.log1p(df_sorted["Amount"].values)

    feature_cols = [f"V{i}" for i in range(1, 29)] + ["Amount_log1p"]

    # 3. Chronological partitioning
    n = len(df_sorted)
    n_train = int(n * train_frac)
    n_val = int(n * (train_frac + val_frac))

    train_df = df_sorted.iloc[:n_train].copy().reset_index(drop=True)
    val_df = df_sorted.iloc[n_train:n_val].copy().reset_index(drop=True)
    test_df = df_sorted.iloc[n_val:].copy().reset_index(drop=True)

    # 4. Extract legitimate subset for unsupervised model fitting
    train_legit_mask = train_df["Class"] == 0
    val_legit_mask = val_df["Class"] == 0

    train_legit_df = train_df[train_legit_mask].copy().reset_index(drop=True)
    val_legit_df = val_df[val_legit_mask].copy().reset_index(drop=True)

    # 5. Fit StandardScaler strictly on legitimate train data
    scaler = StandardScaler()
    scaler.fit(train_legit_df[feature_cols].values)

    # 6. Transform all feature sets
    X_train_legit = scaler.transform(train_legit_df[feature_cols].values)
    X_train_all = scaler.transform(train_df[feature_cols].values)
    y_train = train_df["Class"].values.astype(int)

    X_val_legit = scaler.transform(val_legit_df[feature_cols].values)
    X_val = scaler.transform(val_df[feature_cols].values)
    y_val = val_df["Class"].values.astype(int)

    X_test = scaler.transform(test_df[feature_cols].values)
    y_test = test_df["Class"].values.astype(int)

    # Metadata & Partition summaries
    def _split_stats(name: str, y_arr: np.ndarray, subset_df: pd.DataFrame) -> Dict[str, Any]:
        total = len(y_arr)
        fraud = int((y_arr == 1).sum())
        legit = int((y_arr == 0).sum())
        rate = fraud / total if total > 0 else 0.0
        time_min = float(subset_df["Time"].min())
        time_max = float(subset_df["Time"].max())
        return {
            "name": name,
            "total": total,
            "legit": legit,
            "fraud": fraud,
            "fraud_base_rate": rate,
            "time_range_hours": (time_min / 3600.0, time_max / 3600.0),
        }

    stats = {
        "train": _split_stats("Train (first 60%)", y_train, train_df),
        "val": _split_stats("Validation (middle 20%)", y_val, val_df),
        "test": _split_stats("Test (final 20%)", y_test, test_df),
        "overall": get_dataset_summary(df),
    }

    return {
        "X_train_legit": X_train_legit,
        "X_train_all": X_train_all,
        "y_train": y_train,
        "X_val_legit": X_val_legit,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "feature_cols": feature_cols,
        "scaler": scaler,
        "stats": stats,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "df_raw": df_sorted,
    }
