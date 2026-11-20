"""Streamlit Interactive Demo: Card Fraud Triage via GMM & Autoencoder Anomaly Detection.

Research/educational prototype, not a production fraud detection system.
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

repo_root = os.path.abspath(os.path.dirname(__file__))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.gmm_model import load_gmm, score_samples_gmm
from src.autoencoder import load_autoencoder, score_samples_ae
from src.data import prepare_data

st.set_page_config(
    page_title="Card Fraud Triage | Anomaly Detection",
    page_icon="💳",
    layout="wide",
)

st.title("💳 Card Fraud Triage via Density & Reconstruction Anomaly Detection")
st.markdown(
    """
    > **DISCLAIMER:** *Research / educational triage prototype, not a production fraud detection system.*
    > Models are trained **strictly on legitimate transactions** to detect novel fraud patterns as distribution anomalies.
    """
)


@st.cache_resource
def load_all_models_and_scaler():
    """Load pretrained models and scaler."""
    data_dict = prepare_data("data/creditcard.csv")
    scaler = data_dict["scaler"]
    feature_cols = data_dict["feature_cols"]

    gmm = load_gmm("models/gmm_model.joblib")
    ae = load_autoencoder("models/autoencoder.pt", device="cpu")
    dae = load_autoencoder("models/denoising_autoencoder.pt", device="cpu")

    return {
        "scaler": scaler,
        "feature_cols": feature_cols,
        "gmm": gmm,
        "ae": ae,
        "dae": dae,
        "test_df": data_dict["test_df"],
    }


try:
    resources = load_all_models_and_scaler()
    models_loaded = True
except Exception as e:
    st.error(f"Error loading models or dataset: {e}. Please ensure pipeline has completed.")
    models_loaded = False

if models_loaded:
    scaler = resources["scaler"]
    feature_cols = resources["feature_cols"]
    gmm = resources["gmm"]
    ae = resources["ae"]
    dae = resources["dae"]
    sample_pool = resources["test_df"]

    # Sidebar
    st.sidebar.header("⚙️ Configuration & Model Selection")
    model_choice = st.sidebar.selectbox(
        "Select Anomaly Scoring Model",
        ["Autoencoder (Reconstruction Error)", "Denoising Autoencoder", "Gaussian Mixture Model (AML Baseline)", "Ensemble (Mean Normalized Score)"],
    )

    st.sidebar.subheader("Data Source")
    data_mode = st.sidebar.radio(
        "Choose Evaluation Data:",
        ["Sample Batch from Untouched Test Set (500 Txns)", "Upload Custom CSV"],
    )

    df_input = None
    if data_mode == "Upload Custom CSV":
        uploaded_file = st.sidebar.file_uploader("Upload CSV (must contain V1-V28, Amount)", type=["csv"])
        if uploaded_file is not None:
            df_input = pd.read_csv(uploaded_file)
    else:
        sample_size = st.sidebar.slider("Sample Size", min_value=100, max_value=2000, value=500, step=100)
        # Mix legitimate and fraud transactions from test set
        test_fraud = sample_pool[sample_pool["Class"] == 1]
        test_legit = sample_pool[sample_pool["Class"] == 0]
        n_fraud_sample = min(len(test_fraud), max(5, int(sample_size * 0.05)))
        n_legit_sample = sample_size - n_fraud_sample
        df_input = pd.concat([
            test_fraud.sample(n=n_fraud_sample, random_state=42),
            test_legit.sample(n=n_legit_sample, random_state=42),
        ]).sample(frac=1.0, random_state=42).reset_index(drop=True)

    if df_input is not None:
        # Preprocessing
        if "Amount_log1p" not in df_input.columns and "Amount" in df_input.columns:
            df_input["Amount_log1p"] = np.log1p(df_input["Amount"])

        missing_features = [col for col in feature_cols if col not in df_input.columns]
        if missing_features:
            st.error(f"Missing required columns in dataset: {missing_features}")
        else:
            X_scaled = scaler.transform(df_input[feature_cols].values)

            # Compute scores
            scores_gmm = score_samples_gmm(gmm, X_scaled)
            scores_ae = score_samples_ae(ae, X_scaled)
            scores_dae = score_samples_ae(dae, X_scaled)

            # Standardized z-scores for ensemble
            z_gmm = (scores_gmm - np.mean(scores_gmm)) / (np.std(scores_gmm) + 1e-6)
            z_ae = (scores_ae - np.mean(scores_ae)) / (np.std(scores_ae) + 1e-6)
            z_dae = (scores_dae - np.mean(scores_dae)) / (np.std(scores_dae) + 1e-6)
            z_ensemble = (z_gmm + z_ae + z_dae) / 3.0

            df_input["Score_GMM"] = scores_gmm
            df_input["Score_Autoencoder"] = scores_ae
            df_input["Score_Denoising_AE"] = scores_dae
            df_input["Score_Ensemble"] = z_ensemble

            if model_choice.startswith("Autoencoder"):
                active_score_col = "Score_Autoencoder"
            elif model_choice.startswith("Denoising"):
                active_score_col = "Score_Denoising_AE"
            elif model_choice.startswith("Gaussian"):
                active_score_col = "Score_GMM"
            else:
                active_score_col = "Score_Ensemble"

            active_scores = df_input[active_score_col].values

            # Threshold Controls
            st.sidebar.subheader("🎯 Triage Thresholding")
            thresh_mode = st.sidebar.radio(
                "Thresholding Strategy",
                ["Percentile / Review Capacity (Top %)", "Manual Score Cutoff"],
            )

            if thresh_mode == "Percentile / Review Capacity (Top %)":
                capacity_pct = st.sidebar.slider("Top Review Capacity (%)", 0.1, 10.0, 1.0, 0.1)
                cutoff = float(np.percentile(active_scores, 100.0 - capacity_pct))
            else:
                min_s = float(active_scores.min())
                max_s = float(active_scores.max())
                cutoff = st.sidebar.slider(
                    "Score Cutoff",
                    min_value=min_s,
                    max_value=max_s,
                    value=float(np.percentile(active_scores, 98.0)),
                )

            df_input["Triage_Flag"] = (active_scores >= cutoff).astype(int)
            flagged_count = int(df_input["Triage_Flag"].sum())
            total_count = len(df_input)

            # Metrics row
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Transactions", f"{total_count:,}")
            col2.metric("Flagged for Review", f"{flagged_count:,}", f"{(flagged_count/total_count)*100:.2f}%")
            col3.metric("Selected Cutoff", f"{cutoff:.3f}")

            has_labels = "Class" in df_input.columns
            if has_labels:
                tp = int(((df_input["Triage_Flag"] == 1) & (df_input["Class"] == 1)).sum())
                fp = int(((df_input["Triage_Flag"] == 1) & (df_input["Class"] == 0)).sum())
                fn = int(((df_input["Triage_Flag"] == 0) & (df_input["Class"] == 1)).sum())
                tn = int(((df_input["Triage_Flag"] == 0) & (df_input["Class"] == 0)).sum())
                total_fraud = int((df_input["Class"] == 1).sum())

                precision = (tp / (tp + fp)) * 100 if (tp + fp) > 0 else 0.0
                recall = (tp / total_fraud) * 100 if total_fraud > 0 else 0.0
                col4.metric("Recall (Fraud Caught)", f"{recall:.1f}%", f"{tp}/{total_fraud} frauds")

                # Confusion Matrix Banner
                st.subheader("📊 Triage Performance on Loaded Batch")
                mcol1, mcol2, mcol3, mcol4 = st.columns(4)
                mcol1.metric("Precision (Triage Purity)", f"{precision:.1f}%")
                mcol2.metric("False Positives", f"{fp:,}")
                mcol3.metric("False Negatives (Missed)", f"{fn:,}")
                mcol4.metric("Review Burden (FP/TP)", f"{fp/max(tp,1):.1f}")
            else:
                col4.metric("Labels Present", "No (Unsupervised)")

            # Visualizations
            st.subheader("📈 Anomaly Score Distribution & Triage Cutoff")
            fig, ax = plt.subplots(figsize=(10, 4))
            if has_labels:
                sns.kdeplot(df_input[df_input["Class"] == 0][active_score_col], ax=ax, label="Legit (0)", color="#2b5c8f", fill=True, alpha=0.3)
                sns.kdeplot(df_input[df_input["Class"] == 1][active_score_col], ax=ax, label="Fraud (1)", color="#d95f02", fill=True, alpha=0.3)
            else:
                sns.kdeplot(active_scores, ax=ax, label="Transactions", color="#2b5c8f", fill=True, alpha=0.3)
            ax.axvline(cutoff, color="red", linestyle="--", linewidth=2, label=f"Triage Cutoff ({cutoff:.2f})")
            ax.set_title(f"Score Distribution: {model_choice}")
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)

            # High Risk Transactions Table
            st.subheader("🚨 Priority Triage Queue (Top Suspicious Transactions)")
            flagged_df = df_input[df_input["Triage_Flag"] == 1].sort_values(active_score_col, ascending=False)
            display_cols = ["Time", "Amount", active_score_col, "Triage_Flag"]
            if has_labels:
                display_cols.insert(2, "Class")
            for c in ["V14", "V12", "V17"]:
                if c in flagged_df.columns:
                    display_cols.append(c)

            st.dataframe(flagged_df[display_cols].head(50), use_container_width=True)

            # Download CSV
            csv_data = flagged_df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Flagged Queue as CSV", csv_data, "flagged_transactions.csv", "text/csv")
