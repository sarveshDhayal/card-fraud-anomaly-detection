"""End-to-end training, evaluation, visualization, and artifact generation pipeline.

Executes:
1. Data loading, chronological splitting, scaling.
2. Exploratory Data Analysis & visual generation.
3. GMM AIC/BIC tuning, model fitting, and scoring.
4. Standard & Denoising Autoencoder training with early stopping.
5. Reference baselines: Isolation Forest (unsupervised) and Logistic Regression (supervised upper bound).
6. Comprehensive comparative evaluation on untouched mixed test set.
7. Validation-derived threshold analysis applied to test set.
8. Error analysis on false positives and false negatives (missed frauds).
9. Output serialization: plots, models, comparison_table.csv, and metrics.json.
"""

from typing import Dict, Any, List, Tuple
import os
import sys

# Ensure repository root is on sys.path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import json
import numpy as np
import pandas as pd

# Set matplotlib cache directory and non-interactive backend
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_curve

from src.data import set_seed, prepare_data
from src.gmm_model import tune_gmm, fit_gmm, score_samples_gmm, save_gmm
from src.autoencoder import (
    train_autoencoder,
    score_samples_ae,
    save_autoencoder,
)
from src.metrics import (
    compute_comprehensive_metrics,
    select_thresholds_on_val,
    evaluate_threshold_on_test,
)


def run_eda_visualizations(data_dict: Dict[str, Any], output_dir: str = "outputs/figures") -> None:
    """Generate and save EDA plots."""
    os.makedirs(output_dir, exist_ok=True)
    df = data_dict["df_raw"]
    stats = data_dict["stats"]
    train_df = data_dict["train_df"]
    val_df = data_dict["val_df"]
    test_df = data_dict["test_df"]

    sns.set_theme(style="whitegrid", font_scale=1.1)

    # 1. Class Imbalance
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    class_counts = df["Class"].value_counts()
    labels = ["Legitimate (0)", "Fraud (1)"]
    counts = [class_counts.get(0, 0), class_counts.get(1, 0)]
    percentages = [c / len(df) * 100 for c in counts]

    colors = ["#2b5c8f", "#d95f02"]
    bars = ax1.bar(labels, counts, color=colors, width=0.5, edgecolor="black", alpha=0.85)
    ax1.set_yscale("log")
    ax1.set_ylabel("Count (Log Scale)")
    ax1.set_title("Transaction Class Distribution (Log Scale)")
    for bar, count, pct in zip(bars, counts, percentages):
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() * 1.5,
            f"{count:,}\n({pct:.3f}%)",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Base rate across splits
    split_names = ["Train (60%)", "Validation (20%)", "Test (20%)"]
    split_rates = [
        stats["train"]["fraud_base_rate"] * 100,
        stats["val"]["fraud_base_rate"] * 100,
        stats["test"]["fraud_base_rate"] * 100,
    ]
    ax2.bar(split_names, split_rates, color="#7570b3", width=0.5, edgecolor="black", alpha=0.85)
    ax2.axhline(
        stats["overall"]["fraud_base_rate"] * 100,
        color="red",
        linestyle="--",
        label=f"Overall Base Rate ({stats['overall']['fraud_base_rate']*100:.3f}%)",
    )
    ax2.set_ylabel("Fraud Base Rate (%)")
    ax2.set_title("Fraud Prevalence Across Chronological Partitions")
    ax2.legend()
    for i, v in enumerate(split_rates):
        ax2.text(i, v + 0.005, f"{v:.3f}%", ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "eda_class_imbalance.png"), dpi=300)
    plt.close(fig)

    # 2. Amount Distribution (log1p)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    legit_amt = df[df["Class"] == 0]["Amount"]
    fraud_amt = df[df["Class"] == 1]["Amount"]

    sns.kdeplot(np.log1p(legit_amt), ax=ax1, label="Legitimate (0)", color="#2b5c8f", fill=True, alpha=0.3)
    sns.kdeplot(np.log1p(fraud_amt), ax=ax1, label="Fraud (1)", color="#d95f02", fill=True, alpha=0.3)
    ax1.set_xlabel("log1p(Amount)")
    ax1.set_ylabel("Density")
    ax1.set_title("Kernel Density: log1p(Transaction Amount)")
    ax1.legend()

    box_data = [
        pd.DataFrame({"log1p_Amount": np.log1p(legit_amt), "Class": "Legitimate"}),
        pd.DataFrame({"log1p_Amount": np.log1p(fraud_amt), "Class": "Fraud"}),
    ]
    sns.boxplot(
        data=pd.concat(box_data),
        x="Class",
        y="log1p_Amount",
        palette=["#2b5c8f", "#d95f02"],
        ax=ax2,
        width=0.4,
    )
    ax2.set_title("Boxplot Comparison: log1p(Amount)")
    ax2.set_ylabel("log1p(Amount)")

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "eda_amount_distribution.png"), dpi=300)
    plt.close(fig)

    # 3. Time Distribution over 48 hours & Chronological Splits
    fig, ax = plt.subplots(figsize=(12, 5))
    time_hours = df["Time"] / 3600.0
    ax.hist(time_hours, bins=96, color="#4daf4a", alpha=0.6, edgecolor="black", label="Transaction Volume")

    # Mark partition boundaries
    train_end = train_df["Time"].max() / 3600.0
    val_end = val_df["Time"].max() / 3600.0
    ax.axvline(train_end, color="#1f78b4", linestyle="--", linewidth=2.5, label=f"Train/Val Boundary ({train_end:.1f}h)")
    ax.axvline(val_end, color="#e31a1c", linestyle="--", linewidth=2.5, label=f"Val/Test Boundary ({val_end:.1f}h)")

    ax.set_xlabel("Elapsed Time (Hours)")
    ax.set_ylabel("Number of Transactions per 30-min Bin")
    ax.set_title("Chronological Partitioning Across the 48-Hour Dataset")
    ax.legend(loc="upper right")

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "eda_time_distribution.png"), dpi=300)
    plt.close(fig)


def run_pipeline() -> Dict[str, Any]:
    """Execute complete end-to-end experiment and save all artifacts."""
    seed = 42
    set_seed(seed)
    figures_dir = "outputs/figures"
    results_dir = "outputs/results"
    models_dir = "outputs/models"
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    print(">>> 1. Loading data & creating chronological splits...")
    data_dict = prepare_data("data/raw/creditcard.csv", seed=seed)
    run_eda_visualizations(data_dict, output_dir=figures_dir)
    print("EDA plots saved.")

    X_train_legit = data_dict["X_train_legit"]
    X_val_legit = data_dict["X_val_legit"]
    X_val = data_dict["X_val"]
    y_val = data_dict["y_val"]
    X_test = data_dict["X_test"]
    y_test = data_dict["y_test"]

    # -------------------------------------------------------------
    # 2. AML Baseline: Gaussian Mixture Model (GMM)
    # -------------------------------------------------------------
    print(">>> 2. Tuning Gaussian Mixture Model via AIC / BIC...")
    component_range = range(1, 9)
    cov_types = ["diag", "full"]
    gmm_tuning_df, gmm_selection = tune_gmm(
        X_train_legit,
        component_range=component_range,
        covariance_types=cov_types,
        subsample_size=25000,
        reg_covar=1e-4,
        seed=seed,
    )

    # Plot AIC/BIC Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for c_type in cov_types:
        subset = gmm_tuning_df[gmm_tuning_df["covariance_type"] == c_type]
        ax1.plot(subset["n_components"], subset["bic"], marker="o", label=f"Cov: {c_type}")
        ax2.plot(subset["n_components"], subset["aic"], marker="s", label=f"Cov: {c_type}")

    ax1.set_title("Bayesian Information Criterion (BIC)")
    ax1.set_xlabel("Number of Components")
    ax1.set_ylabel("BIC (Lower is better)")
    ax1.legend()

    ax2.set_title("Akaike Information Criterion (AIC)")
    ax2.set_xlabel("Number of Components")
    ax2.set_ylabel("AIC (Lower is better)")
    ax2.legend()

    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, "gmm_aic_bic_curve.png"), dpi=300)
    plt.close(fig)

    best_k = int(gmm_selection["best_bic"]["n_components"])
    best_cov = str(gmm_selection["best_bic"]["covariance_type"])
    print(f"Optimal GMM configuration by BIC: k={best_k}, covariance_type='{best_cov}'")

    print(f"Fitting final GMM on full legitimate training set ({len(X_train_legit):,} samples)...")
    final_gmm = fit_gmm(X_train_legit, n_components=best_k, covariance_type=best_cov, reg_covar=1e-4, seed=seed)
    save_gmm(final_gmm, os.path.join(models_dir, "gmm_model.joblib"))

    gmm_val_scores = score_samples_gmm(final_gmm, X_val)
    gmm_test_scores = score_samples_gmm(final_gmm, X_test)

    # -------------------------------------------------------------
    # 3. DL Model: Autoencoders (Standard & Denoising)
    # -------------------------------------------------------------
    print(">>> 3. Training Standard Undercomplete Autoencoder...")
    ae_model, ae_meta = train_autoencoder(
        X_train_legit=X_train_legit,
        X_val_legit=X_val_legit,
        input_dim=29,
        latent_dim=5,
        epochs=35,
        batch_size=512,
        lr=1e-3,
        weight_decay=1e-5,
        patience=6,
        is_denoising=False,
        seed=seed,
    )
    save_autoencoder(ae_model, os.path.join(models_dir, "autoencoder.pt"))
    ae_val_scores = score_samples_ae(ae_model, X_val)
    ae_test_scores = score_samples_ae(ae_model, X_test)

    print(">>> Training Denoising Autoencoder (noise_factor=0.15)...")
    dae_model, dae_meta = train_autoencoder(
        X_train_legit=X_train_legit,
        X_val_legit=X_val_legit,
        input_dim=29,
        latent_dim=5,
        epochs=35,
        batch_size=512,
        lr=1e-3,
        weight_decay=1e-5,
        patience=6,
        is_denoising=True,
        noise_factor=0.15,
        seed=seed,
    )
    save_autoencoder(dae_model, os.path.join(models_dir, "denoising_autoencoder.pt"))
    dae_val_scores = score_samples_ae(dae_model, X_val)
    dae_test_scores = score_samples_ae(dae_model, X_test)

    # Plot AE Loss Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(ae_meta["history"]["train_loss"], label="Train MSE Loss", color="#1f78b4", linewidth=2)
    ax1.plot(ae_meta["history"]["val_loss"], label="Val MSE Loss (Legit Only)", color="#33a02c", linestyle="--", linewidth=2)
    ax1.set_title(f"Standard AE Training (Best Val MSE: {ae_meta['best_val_loss']:.4f})")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MSE Loss")
    ax1.legend()

    ax2.plot(dae_meta["history"]["train_loss"], label="Train MSE Loss", color="#e31a1c", linewidth=2)
    ax2.plot(dae_meta["history"]["val_loss"], label="Val MSE Loss (Legit Only)", color="#ff7f00", linestyle="--", linewidth=2)
    ax2.set_title(f"Denoising AE Training (Best Val MSE: {dae_meta['best_val_loss']:.4f})")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("MSE Loss")
    ax2.legend()

    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, "autoencoder_loss_curves.png"), dpi=300)
    plt.close(fig)

    # -------------------------------------------------------------
    # 4. Reference Baselines
    # -------------------------------------------------------------
    print(">>> 4. Fitting Reference Baselines (Isolation Forest & Supervised Upper Bound)...")
    # Isolation Forest: unsupervised on legit train data
    iforest = IsolationForest(
        n_estimators=150,
        max_samples=min(25000, len(X_train_legit)),
        contamination=0.002,
        random_state=seed,
        n_jobs=-1,
    )
    iforest.fit(X_train_legit)
    # Higher score = more anomalous: -score_samples
    iforest_val_scores = -iforest.score_samples(X_val)
    iforest_test_scores = -iforest.score_samples(X_test)

    # Supervised Upper Bound (Logistic Regression trained on train_df with Class labels)
    X_train_all = data_dict["X_train_all"]
    y_train = data_dict["y_train"]
    logreg = LogisticRegression(max_iter=500, class_weight="balanced", random_state=seed)
    logreg.fit(X_train_all, y_train)
    logreg_val_scores = logreg.predict_proba(X_val)[:, 1]
    logreg_test_scores = logreg.predict_proba(X_test)[:, 1]

    # -------------------------------------------------------------
    # 5. Comprehensive Comparative Evaluation
    # -------------------------------------------------------------
    print(">>> 5. Evaluating all models on the untouched test set...")
    # Standardized score fusion for Hybrid ML+DL Model
    z_gmm_test = (gmm_test_scores - np.mean(gmm_val_scores)) / (np.std(gmm_val_scores) + 1e-6)
    z_gmm_val = (gmm_val_scores - np.mean(gmm_val_scores)) / (np.std(gmm_val_scores) + 1e-6)
    z_ae_test = (ae_test_scores - np.mean(ae_val_scores)) / (np.std(ae_val_scores) + 1e-6)
    z_ae_val = (ae_val_scores - np.mean(ae_val_scores)) / (np.std(ae_val_scores) + 1e-6)

    hybrid_val_scores = 0.5 * z_gmm_val + 0.5 * z_ae_val
    hybrid_test_scores = 0.5 * z_gmm_test + 0.5 * z_ae_test

    models_to_eval = [
        ("GMM (AML Baseline)", gmm_test_scores, gmm_val_scores),
        ("Autoencoder", ae_test_scores, ae_val_scores),
        ("Denoising Autoencoder", dae_test_scores, dae_val_scores),
        ("Hybrid (GMM + Autoencoder)", hybrid_test_scores, hybrid_val_scores),
        ("Isolation Forest (Extra Baseline)", iforest_test_scores, iforest_val_scores),
        ("Logistic Regression (Supervised Upper Bound)", logreg_test_scores, logreg_val_scores),
    ]

    metrics_dict = {}
    table_rows = []

    for name, t_scores, v_scores in models_to_eval:
        m = compute_comprehensive_metrics(
            model_name=name,
            y_true=y_test,
            scores=t_scores,
            val_y=y_val,
            val_scores=v_scores,
            seed=seed,
        )
        metrics_dict[name] = m
        table_rows.append(
            {
                "Model": name,
                "PR-AUC (Primary)": f"{m['pr_auc']:.4f} [{m['pr_auc_ci_lower']:.4f}, {m['pr_auc_ci_upper']:.4f}]",
                "ROC-AUC": f"{m['roc_auc']:.4f}",
                "Recall @ 0.1% FPR": f"{m['recall_at_fpr_0.1pct']*100:.2f}%",
                "Recall @ 0.5% FPR": f"{m['recall_at_fpr_0.5pct']*100:.2f}%",
                "Recall @ 1.0% FPR": f"{m['recall_at_fpr_1.0pct']*100:.2f}%",
                "Precision @ 100": f"{m['precision_at_100']*100:.2f}%",
                "Precision @ 500": f"{m['precision_at_500']*100:.2f}%",
                "Precision @ 1000": f"{m['precision_at_1000']*100:.2f}%",
                "Test Base Rate": f"{m['base_rate']*100:.3f}%",
            }
        )

    comparison_df = pd.DataFrame(table_rows)
    comparison_df.to_csv(os.path.join(results_dir, "comparison_table.csv"), index=False)
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics_dict, f, indent=2)

    # -------------------------------------------------------------
    # 6. Comparative Visualizations
    # -------------------------------------------------------------
    print(">>> 6. Generating comparative evaluation charts...")
    palette = {
        "GMM (AML Baseline)": "#1f78b4",
        "Autoencoder": "#33a02c",
        "Denoising Autoencoder": "#e31a1c",
        "Hybrid (GMM + Autoencoder)": "#984ea3",
        "Isolation Forest (Extra Baseline)": "#ff7f00",
        "Logistic Regression (Supervised Upper Bound)": "#6a3d9a",
    }

    # PR Curves
    fig, ax = plt.subplots(figsize=(10, 7))
    for name, t_scores, _ in models_to_eval:
        prec, rec, _ = precision_recall_curve(y_test, t_scores)
        m = metrics_dict[name]
        ax.plot(
            rec,
            prec,
            label=f"{name} (PR-AUC = {m['pr_auc']:.3f})",
            color=palette.get(name, "gray"),
            linewidth=2.2,
        )

    base_rate = float((y_test == 1).mean())
    ax.axhline(base_rate, color="black", linestyle=":", label=f"Random / Base Rate ({base_rate*100:.3f}%)")
    ax.set_title("Precision-Recall Curves on Untouched Test Set (Primary Metric)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Recall (Detection Rate)", fontsize=12)
    ax.set_ylabel("Precision (Triage Purity)", fontsize=12)
    ax.set_ylim([0.0, 1.02])
    ax.legend(loc="upper right")
    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, "pr_curves_comparison.png"), dpi=300)
    plt.close(fig)

    # ROC Curves
    fig, ax = plt.subplots(figsize=(10, 7))
    for name, t_scores, _ in models_to_eval:
        fpr_arr, tpr_arr, _ = roc_curve(y_test, t_scores)
        m = metrics_dict[name]
        ax.plot(
            fpr_arr,
            tpr_arr,
            label=f"{name} (ROC-AUC = {m['roc_auc']:.4f})",
            color=palette.get(name, "gray"),
            linewidth=2.2,
        )

    ax.plot([0, 1], [0, 1], "k--", label="Random Classifier")
    ax.set_title("Receiver Operating Characteristic (ROC) Curves", fontsize=14, fontweight="bold")
    ax.set_xlabel("False Positive Rate (FPR)", fontsize=12)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=12)
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, "roc_curves_comparison.png"), dpi=300)
    plt.close(fig)

    # Anomaly Score Distributions (GMM vs AE vs DAE)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    score_triplets = [
        ("GMM Negative Log-Likelihood", gmm_test_scores, axes[0]),
        ("Autoencoder Reconstruction MSE", ae_test_scores, axes[1]),
        ("Denoising AE Reconstruction MSE", dae_test_scores, axes[2]),
    ]

    for title, scores, ax in score_triplets:
        # Standardize score for visual distribution comparison
        norm_scores = (scores - np.median(scores)) / (np.std(scores) + 1e-6)
        sns.kdeplot(norm_scores[y_test == 0], ax=ax, label="Legitimate (0)", color="#2b5c8f", fill=True, alpha=0.3)
        sns.kdeplot(norm_scores[y_test == 1], ax=ax, label="Fraud (1)", color="#d95f02", fill=True, alpha=0.3)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Standardized Anomaly Score")
        ax.set_ylabel("Density")
        ax.set_xlim([np.percentile(norm_scores, 1), np.percentile(norm_scores[y_test == 1], 98) + 2])
        ax.legend()

    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, "score_distributions.png"), dpi=300)
    plt.close(fig)

    # Recall at FPR budgets Bar Chart
    fig, ax = plt.subplots(figsize=(12, 6))
    bar_width = 0.16
    fpr_budgets = ["0.1% FPR", "0.5% FPR", "1.0% FPR"]
    indices = np.arange(len(fpr_budgets))

    for idx, (name, _, _) in enumerate(models_to_eval):
        m = metrics_dict[name]
        recalls = [
            m["recall_at_fpr_0.1pct"] * 100,
            m["recall_at_fpr_0.5pct"] * 100,
            m["recall_at_fpr_1.0pct"] * 100,
        ]
        ax.bar(
            indices + idx * bar_width,
            recalls,
            bar_width,
            label=name,
            color=palette.get(name, "gray"),
            edgecolor="black",
            alpha=0.9,
        )

    ax.set_title("Recall at Fixed False Positive Rate (FPR) Budgets on Test Set", fontsize=14, fontweight="bold")
    ax.set_xlabel("False Positive Rate Budget", fontsize=12)
    ax.set_ylabel("Recall (%) [Fraud Caught]", fontsize=12)
    ax.set_xticks(indices + bar_width * 2)
    ax.set_xticklabels(fpr_budgets, fontsize=11, fontweight="bold")
    ax.legend(loc="upper left")
    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, "recall_at_fpr_comparison.png"), dpi=300)
    plt.close(fig)

    # Precision at K Bar Chart
    fig, ax = plt.subplots(figsize=(12, 6))
    ks = ["Top 100 (~0.18% Test)", "Top 500 (~0.88% Test)", "Top 1000 (~1.76% Test)"]
    indices = np.arange(len(ks))

    for idx, (name, _, _) in enumerate(models_to_eval):
        m = metrics_dict[name]
        precs = [
            m["precision_at_100"] * 100,
            m["precision_at_500"] * 100,
            m["precision_at_1000"] * 100,
        ]
        ax.bar(
            indices + idx * bar_width,
            precs,
            bar_width,
            label=name,
            color=palette.get(name, "gray"),
            edgecolor="black",
            alpha=0.9,
        )

    ax.set_title("Precision at Review Capacity Budgets (Precision@K)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Triage Review Capacity (K Flagged Transactions)", fontsize=12)
    ax.set_ylabel("Precision (%) [Hit Rate]", fontsize=12)
    ax.set_xticks(indices + bar_width * 2)
    ax.set_xticklabels(ks, fontsize=11, fontweight="bold")
    ax.legend(loc="upper right")
    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, "precision_at_k_comparison.png"), dpi=300)
    plt.close(fig)

    # -------------------------------------------------------------
    # 7. Threshold Analysis & Trade-off Curve
    # -------------------------------------------------------------
    print(">>> 7. Analyzing validation-selected thresholds applied to test set...")
    # Plot False Positives per Fraud caught vs Recall trade-off
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for name, t_scores, _ in [models_to_eval[0], models_to_eval[1], models_to_eval[2]]:
        prec, rec, thresholds = precision_recall_curve(y_test, t_scores)
        # Avoid division by zero
        prec_clean = np.clip(prec[:-1], 1e-6, 1.0)
        # FP per TP = (1 - Precision) / Precision
        fp_per_tp = (1.0 - prec_clean) / prec_clean

        ax1.plot(rec[:-1], fp_per_tp, label=name, color=palette.get(name, "gray"), linewidth=2.2)

    ax1.set_yscale("log")
    ax1.set_title("False Positives per Fraud Caught vs Recall (Lower is Better)", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Recall (Fraction of Fraud Caught)")
    ax1.set_ylabel("False Positives per Fraud Caught (Log Scale)")
    ax1.set_xlim([0.1, 0.95])
    ax1.legend()

    # Confusion matrix visual for GMM and AE at Val Target FPR = 0.5%
    gmm_thresh_05 = metrics_dict["GMM (AML Baseline)"]["val_derived_thresholds"]["fpr_0.5pct"]["val_threshold"]
    ae_thresh_05 = metrics_dict["Autoencoder"]["val_derived_thresholds"]["fpr_0.5pct"]["val_threshold"]

    gmm_test_eval = evaluate_threshold_on_test(y_test, gmm_test_scores, gmm_thresh_05)
    ae_test_eval = evaluate_threshold_on_test(y_test, ae_test_scores, ae_thresh_05)

    cm_summary = pd.DataFrame(
        [
            {
                "Model": "GMM (AML Baseline)",
                "TP (Fraud Caught)": gmm_test_eval["tp"],
                "FP (Legit Flagged)": gmm_test_eval["fp"],
                "FN (Missed Fraud)": gmm_test_eval["fn"],
                "Recall": f"{gmm_test_eval['recall']*100:.1f}%",
                "Precision": f"{gmm_test_eval['precision']*100:.1f}%",
                "FP per Fraud": f"{gmm_test_eval['fp_per_fraud_caught']:.1f}",
            },
            {
                "Model": "Autoencoder",
                "TP (Fraud Caught)": ae_test_eval["tp"],
                "FP (Legit Flagged)": ae_test_eval["fp"],
                "FN (Missed Fraud)": ae_test_eval["fn"],
                "Recall": f"{ae_test_eval['recall']*100:.1f}%",
                "Precision": f"{ae_test_eval['precision']*100:.1f}%",
                "FP per Fraud": f"{ae_test_eval['fp_per_fraud_caught']:.1f}",
            },
        ]
    )

    # Plot grouped bar of TP vs FP at FPR 0.5%
    cm_plot_df = pd.DataFrame(
        {
            "Model": ["GMM", "GMM", "Autoencoder", "Autoencoder"],
            "Outcome": ["True Positives (Fraud Caught)", "False Positives (Legit Flagged)", "True Positives (Fraud Caught)", "False Positives (Legit Flagged)"],
            "Count": [gmm_test_eval["tp"], gmm_test_eval["fp"], ae_test_eval["tp"], ae_test_eval["fp"]],
        }
    )
    sns.barplot(data=cm_plot_df, x="Model", y="Count", hue="Outcome", ax=ax2, palette=["#2ca02c", "#d62728"])
    ax2.set_title("Test Set Outcomes at Validation-Chosen FPR = 0.5% Threshold", fontsize=13, fontweight="bold")
    ax2.set_ylabel("Transaction Count")

    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, "threshold_tradeoff.png"), dpi=300)
    plt.close(fig)

    # -------------------------------------------------------------
    # 8. Error Analysis: False Positives & False Negatives
    # -------------------------------------------------------------
    print(">>> 8. Performing error analysis...")
    test_df = data_dict["test_df"].copy()
    test_df["gmm_score"] = gmm_test_scores
    test_df["ae_score"] = ae_test_scores

    # Examine Top False Positives (Legitimate with highest anomaly scores)
    top_fps_gmm = test_df[test_df["Class"] == 0].sort_values("gmm_score", ascending=False).head(20)
    top_fps_ae = test_df[test_df["Class"] == 0].sort_values("ae_score", ascending=False).head(20)

    # Examine Missed Frauds (Frauds with lowest anomaly scores)
    missed_frauds_gmm = test_df[test_df["Class"] == 1].sort_values("gmm_score", ascending=True).head(20)
    missed_frauds_ae = test_df[test_df["Class"] == 1].sort_values("ae_score", ascending=True).head(20)

    # Feature inspection plot: Amount and key PCA components (e.g. V14, V12, V17)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    feature_inspect = ["V14", "V12", "V17", "Amount"]

    for ax, feat in zip(axes.flatten(), feature_inspect):
        if feat == "Amount":
            sns.boxplot(
                data=test_df,
                x="Class",
                y="Amount",
                ax=ax,
                showfliers=False,
                palette=["#2b5c8f", "#d95f02"],
            )
            ax.set_title("Transaction Amount Distribution (Without Extreme Fliers)")
        else:
            sns.kdeplot(test_df[test_df["Class"] == 0][feat], ax=ax, label="Legit", color="#2b5c8f", fill=True, alpha=0.3)
            sns.kdeplot(test_df[test_df["Class"] == 1][feat], ax=ax, label="Fraud", color="#d95f02", fill=True, alpha=0.3)
            ax.axvline(top_fps_gmm[feat].mean(), color="#1f78b4", linestyle="--", label="Top FP (GMM) Mean")
            ax.axvline(missed_frauds_gmm[feat].mean(), color="#e31a1c", linestyle=":", label="Missed Fraud Mean")
            ax.set_title(f"Distribution of {feat} with Error Centers")
            ax.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, "error_analysis.png"), dpi=300)
    plt.close(fig)

    print(">>> All experiments complete! Comparison table:")
    print(comparison_df.to_string(index=False))

    return {
        "metrics": metrics_dict,
        "comparison_table": comparison_df,
        "gmm_selection": gmm_selection,
        "ae_meta": ae_meta,
        "dae_meta": dae_meta,
        "cm_summary": cm_summary,
    }


if __name__ == "__main__":
    run_pipeline()
