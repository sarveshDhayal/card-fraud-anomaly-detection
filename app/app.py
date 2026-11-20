"""Streamlit Interactive Demo: Card Fraud Triage via GMM & Autoencoder Anomaly Detection.

Standalone Cloud-Ready App (Streamlit Cloud, Hugging Face, or Local).
Author: Sarvesh Choudhary
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import torch

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) if os.path.basename(os.path.dirname(__file__)) == "app" else os.path.abspath(os.path.dirname(__file__))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.gmm_model import score_samples_gmm, load_gmm
from src.autoencoder import score_samples_ae, load_autoencoder

st.set_page_config(
    page_title="Card Fraud Triage | Anomaly Detection",
    page_icon="💳",
    layout="wide",
)

st.title("💳 Card Fraud Triage via Density & Reconstruction Anomaly Detection")
st.markdown(
    """
    **Author:** Sarvesh Choudhary | *Individual Research & Engineering Project*  
    > **DISCLAIMER:** *Research / educational triage prototype, not a production fraud detection system.*  
    > Models are trained **strictly on legitimate transactions** to detect novel fraud patterns as distribution anomalies.
    """
)


@st.cache_resource
def load_all_models_and_resources():
    """Load serialized models, scaler, and sample test pool."""
    # Find scaler
    scaler_paths = [
        os.path.join(repo_root, "models/scaler.joblib"),
        os.path.join(repo_root, "outputs/models/scaler.joblib"),
        "models/scaler.joblib",
        "outputs/models/scaler.joblib",
    ]
    scaler = None
    for p in scaler_paths:
        if os.path.exists(p):
            scaler = joblib.load(p)
            break

    # Find GMM
    gmm_paths = [
        os.path.join(repo_root, "models/gmm_model.joblib"),
        os.path.join(repo_root, "outputs/models/gmm_model.joblib"),
        "models/gmm_model.joblib",
    ]
    gmm = None
    for p in gmm_paths:
        if os.path.exists(p):
            gmm = load_gmm(p)
            break

    # Find AE
    ae_paths = [
        os.path.join(repo_root, "models/autoencoder.pt"),
        os.path.join(repo_root, "outputs/models/autoencoder.pt"),
        "models/autoencoder.pt",
    ]
    ae = None
    for p in ae_paths:
        if os.path.exists(p):
            ae = load_autoencoder(p, device="cpu")
            break

    # Find DAE
    dae_paths = [
        os.path.join(repo_root, "models/denoising_autoencoder.pt"),
        os.path.join(repo_root, "outputs/models/denoising_autoencoder.pt"),
        "models/denoising_autoencoder.pt",
    ]
    dae = None
    for p in dae_paths:
        if os.path.exists(p):
            dae = load_autoencoder(p, device="cpu")
            break

    # Find sample transactions pool
    sample_pool = None
    sample_paths = [
        os.path.join(repo_root, "app/sample_transactions.csv"),
        os.path.join(repo_root, "outputs/sample_transactions.csv"),
        os.path.join(repo_root, "data/processed/test_features.csv"),
        "app/sample_transactions.csv",
        "sample_transactions.csv",
    ]
    for p in sample_paths:
        if os.path.exists(p):
            sample_pool = pd.read_csv(p)
            break

    feature_cols = [f"V{i}" for i in range(1, 29)] + ["Amount_log1p"]

    return {
        "scaler": scaler,
        "feature_cols": feature_cols,
        "gmm": gmm,
        "ae": ae,
        "dae": dae,
        "sample_pool": sample_pool,
    }


resources = load_all_models_and_resources()
scaler = resources["scaler"]
feature_cols = resources["feature_cols"]
gmm = resources["gmm"]
ae = resources["ae"]
dae = resources["dae"]
sample_pool = resources["sample_pool"]

if scaler is None or gmm is None or ae is None:
    st.error("Error: Models or Scaler could not be located. Please ensure 'models/' exists.")
else:
    # Sidebar
    st.sidebar.header("⚙️ Configuration & Model Selection")
    model_choice = st.sidebar.selectbox(
        "Select Anomaly Scoring Architecture",
        [
            "Hybrid Model (GMM + Autoencoder Fusion)",
            "Autoencoder (DL Reconstruction MSE)",
            "Gaussian Mixture Model (AML Density Baseline)",
            "Denoising Autoencoder (Regularized DL)",
        ],
    )

    st.sidebar.subheader("📁 Data Source")
    data_mode = st.sidebar.radio(
        "Choose Evaluation Data:",
        ["Curated Test Pool (Sampled from 48h Stream)", "Upload Custom CSV"],
    )

    df_input = None
    if data_mode == "Upload Custom CSV":
        uploaded_file = st.sidebar.file_uploader("Upload CSV (must contain V1-V28, Amount)", type=["csv"])
        if uploaded_file is not None:
            df_input = pd.read_csv(uploaded_file)
    else:
        if sample_pool is not None:
            sample_size = st.sidebar.slider("Sample Size", min_value=100, max_value=min(1000, len(sample_pool)), value=500, step=50)
            df_input = sample_pool.sample(n=sample_size, random_state=42).reset_index(drop=True)
        else:
            st.warning("No local sample pool found. Please upload a CSV.")

    if df_input is not None:
        # Preprocessing
        if "Amount_log1p" not in df_input.columns and "Amount" in df_input.columns:
            df_input["Amount_log1p"] = np.log1p(df_input["Amount"])

        missing_features = [col for col in feature_cols if col not in df_input.columns]
        if missing_features:
            st.error(f"Missing required columns in dataset: {missing_features}")
        else:
            X_scaled = scaler.transform(df_input[feature_cols].values)

            # Compute individual scores
            scores_gmm = score_samples_gmm(gmm, X_scaled)
            scores_ae = score_samples_ae(ae, X_scaled)
            scores_dae = score_samples_ae(dae, X_scaled)

            # Standardized z-scores for hybrid fusion
            z_gmm = (scores_gmm - np.mean(scores_gmm)) / (np.std(scores_gmm) + 1e-6)
            z_ae = (scores_ae - np.mean(scores_ae)) / (np.std(scores_ae) + 1e-6)
            hybrid_scores = 0.5 * z_gmm + 0.5 * z_ae

            df_input["Score_GMM"] = scores_gmm
            df_input["Score_Autoencoder"] = scores_ae
            df_input["Score_Denoising_AE"] = scores_dae
            df_input["Score_Hybrid"] = hybrid_scores

            if model_choice.startswith("Hybrid"):
                active_score_col = "Score_Hybrid"
            elif model_choice.startswith("Autoencoder"):
                active_score_col = "Score_Autoencoder"
            elif model_choice.startswith("Gaussian"):
                active_score_col = "Score_GMM"
            else:
                active_score_col = "Score_Denoising_AE"

            active_scores = df_input[active_score_col].values

            # Threshold Controls
            st.sidebar.subheader("🎯 Triage Thresholding Strategy")
            thresh_mode = st.sidebar.radio(
                "Operational Budgeting Mode",
                ["Percentile / Review Capacity (Top %)", "Manual Score Cutoff"],
            )

            if thresh_mode == "Percentile / Review Capacity (Top %)":
                capacity_pct = st.sidebar.slider("Top Review Capacity Budget (%)", 0.1, 10.0, 1.0, 0.1)
                cutoff = float(np.percentile(active_scores, 100.0 - capacity_pct))
            else:
                min_s = float(active_scores.min())
                max_s = float(active_scores.max())
                cutoff = st.sidebar.slider(
                    "Raw Score Cutoff",
                    min_value=min_s,
                    max_value=max_s,
                    value=float(np.percentile(active_scores, 98.0)),
                )

            df_input["Triage_Flag"] = (active_scores >= cutoff).astype(int)
            flagged_count = int(df_input["Triage_Flag"].sum())
            total_count = len(df_input)

            # Metrics row
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Batch Transactions", f"{total_count:,}")
            col2.metric("Flagged for Investigation", f"{flagged_count:,}", f"{(flagged_count/total_count)*100:.2f}%")
            col3.metric("Selected Decision Cutoff", f"{cutoff:.3f}")

            has_labels = "Class" in df_input.columns
            if has_labels:
                tp = int(((df_input["Triage_Flag"] == 1) & (df_input["Class"] == 1)).sum())
                fp = int(((df_input["Triage_Flag"] == 1) & (df_input["Class"] == 0)).sum())
                fn = int(((df_input["Triage_Flag"] == 0) & (df_input["Class"] == 1)).sum())
                tn = int(((df_input["Triage_Flag"] == 0) & (df_input["Class"] == 0)).sum())
                total_fraud = int((df_input["Class"] == 1).sum())

                precision = (tp / (tp + fp)) * 100 if (tp + fp) > 0 else 0.0
                recall = (tp / total_fraud) * 100 if total_fraud > 0 else 0.0
                col4.metric("Recall (Fraud Caught)", f"{recall:.1f}%", f"{tp}/{total_fraud} frauds caught")

                # Operational Triage Performance Banner
                st.subheader("📊 Operational Triage Metrics (On Loaded Batch)")
                mcol1, mcol2, mcol3, mcol4 = st.columns(4)
                mcol1.metric("Precision (Triage Purity)", f"{precision:.1f}%")
                mcol2.metric("False Alerts (Customer Friction)", f"{fp:,}")
                mcol3.metric("Missed Frauds (FN)", f"{fn:,}")
                mcol4.metric("Review Burden (FP per Fraud)", f"{fp/max(tp,1):.1f} : 1")
            else:
                col4.metric("Labels Present", "No (Unsupervised Live Stream)")

            # Visualizations
            st.subheader("📈 Anomaly Score Distribution & Operating Threshold")
            fig, ax = plt.subplots(figsize=(10, 3.5))
            if has_labels:
                sns.kdeplot(df_input[df_input["Class"] == 0][active_score_col], ax=ax, label="Legit (0)", color="#2b5c8f", fill=True, alpha=0.3)
                sns.kdeplot(df_input[df_input["Class"] == 1][active_score_col], ax=ax, label="Fraud (1)", color="#d95f02", fill=True, alpha=0.3)
            else:
                sns.kdeplot(active_scores, ax=ax, label="Live Transactions", color="#2b5c8f", fill=True, alpha=0.3)
            ax.axvline(cutoff, color="red", linestyle="--", linewidth=2, label=f"Triage Cutoff ({cutoff:.2f})")
            ax.set_title(f"Score Separation: {model_choice}", fontsize=12, fontweight="bold")
            ax.set_xlabel("Anomaly Score (Higher = More Suspicious)")
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)

            # High Risk Priority Queue
            st.subheader("🚨 Priority Triage Queue (Top Suspicious Transactions)")
            flagged_df = df_input[df_input["Triage_Flag"] == 1].sort_values(active_score_col, ascending=False)
            display_cols = ["Time", "Amount", active_score_col, "Triage_Flag"]
            if has_labels:
                display_cols.insert(2, "Class")
            for c in ["V14", "V12", "V17"]:
                if c in flagged_df.columns:
                    display_cols.append(c)

            st.dataframe(flagged_df[display_cols].head(50), use_container_width=True)

            # CSV Download
            csv_data = flagged_df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Flagged Queue as CSV", csv_data, "triage_queue.csv", "text/csv")
