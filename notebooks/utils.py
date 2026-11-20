"""Shared utility functions for notebooks."""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.data import set_seed, load_raw_data, prepare_data
from src.metrics import (
    compute_pr_auc,
    compute_roc_auc,
    recall_at_fpr,
    precision_at_k,
    bootstrap_pr_auc_ci,
)

SEED = 42
set_seed(SEED)
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
sns.set_theme(style="whitegrid", font_scale=1.1)


def load_processed_data(data_dir: str = "../data/processed"):
    """Load pre-processed numpy arrays and dataframes."""
    if not os.path.exists(data_dir):
        data_dir = "data/processed"

    X_train_legit = np.load(os.path.join(data_dir, "X_train_legit.npy"))
    X_val = np.load(os.path.join(data_dir, "X_val.npy"))
    X_val_legit = np.load(os.path.join(data_dir, "X_val_legit.npy"))
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))

    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    y_val = np.load(os.path.join(data_dir, "y_val.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))

    with open(os.path.join(data_dir, "feature_list.txt")) as f:
        features = [line.strip() for line in f if line.strip()]

    return {
        "X_train_legit": X_train_legit,
        "X_val": X_val,
        "X_val_legit": X_val_legit,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "features": features,
    }
