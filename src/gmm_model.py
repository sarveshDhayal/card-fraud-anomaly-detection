"""Gaussian Mixture Model (GMM) anomaly detector for AML baseline.

Implements:
- Hyperparameter tuning over n_components and covariance_type (full, diag) using AIC/BIC.
- Documented fixed-seed subsampling for fast, responsive model selection sweep.
- Final model fitting on 100% of legitimate training transactions.
- Anomaly scoring via negative log-likelihood (-1 * score_samples).
- Numerical stability safeguards (reg_covar) for 29-dimensional covariance matrices.
"""

from typing import Dict, Any, List, Optional
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture


def tune_gmm(
    X_train_legit: np.ndarray,
    component_range: range = range(1, 11),
    covariance_types: Optional[List[str]] = None,
    subsample_size: int = 25000,
    reg_covar: float = 1e-4,
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Perform grid search over components and covariance types using AIC and BIC.

    To keep hyperparameter tuning responsive and avoid cubic complexity overhead
    in 29 dimensions on ~170k rows, we draw a deterministic random subsample
    (fixed seed) for the sweep, then fit the winning configuration on the full dataset.
    """
    if covariance_types is None:
        covariance_types = ["diag", "full"]

    n_samples = len(X_train_legit)
    if n_samples > subsample_size:
        rng = np.random.RandomState(seed)
        subsample_idx = rng.choice(n_samples, size=subsample_size, replace=False)
        X_sub = X_train_legit[subsample_idx]
    else:
        X_sub = X_train_legit

    results = []

    for cov_type in covariance_types:
        for k in component_range:
            gmm = GaussianMixture(
                n_components=k,
                covariance_type=cov_type,
                reg_covar=reg_covar,
                random_state=seed,
                max_iter=100,
                n_init=1,
            )
            gmm.fit(X_sub)
            aic = gmm.aic(X_sub)
            bic = gmm.bic(X_sub)
            results.append(
                {
                    "n_components": k,
                    "covariance_type": cov_type,
                    "aic": aic,
                    "bic": bic,
                    "converged": gmm.converged_,
                }
            )

    results_df = pd.DataFrame(results)

    # Best by BIC (stronger penalty on complexity, less prone to overfitting)
    best_bic_idx = results_df["bic"].idxmin()
    best_bic = results_df.loc[best_bic_idx].to_dict()

    # Best by AIC
    best_aic_idx = results_df["aic"].idxmin()
    best_aic = results_df.loc[best_aic_idx].to_dict()

    selection_summary = {
        "best_bic": best_bic,
        "best_aic": best_aic,
        "subsample_size": len(X_sub),
        "total_legit_samples": n_samples,
    }

    return results_df, selection_summary


def fit_gmm(
    X_train_legit: np.ndarray,
    n_components: int = 3,
    covariance_type: str = "full",
    reg_covar: float = 1e-4,
    max_iter: int = 150,
    seed: int = 42,
) -> GaussianMixture:
    """Fit final Gaussian Mixture Model on 100% of legitimate training samples.

    reg_covar adds a small diagonal regularization term to the covariance matrices,
    preventing singular or non-positive-definite covariance matrices in 29 dimensions.
    """
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        reg_covar=reg_covar,
        random_state=seed,
        max_iter=max_iter,
        n_init=1,
        verbose=0,
    )
    gmm.fit(X_train_legit)
    return gmm


def score_samples_gmm(gmm: GaussianMixture, X: np.ndarray) -> np.ndarray:
    """Compute anomaly score as negative log-likelihood (-1 * log p(x)).

    Higher score = lower probability density under legitimate distribution = more suspicious.
    """
    log_prob = gmm.score_samples(X)
    return -log_prob


def save_gmm(gmm: GaussianMixture, filepath: str = "models/gmm_model.joblib") -> None:
    """Serialize GMM model to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(gmm, filepath)


def load_gmm(filepath: str = "models/gmm_model.joblib") -> GaussianMixture:
    """Load serialized GMM model from disk."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"GMM model not found at {filepath}")
    return joblib.load(filepath)
