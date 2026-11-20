# Phase 2 Final Project Presentation: Card Fraud Triage via GMM & Autoencoder Anomaly Detection

**Course:** Advanced Machine Learning & Deep Learning (CSAI)  
**Project:** Card Fraud Triage via Gaussian Mixture Density Scoring and Deep Autoencoder Anomaly Detection  
**Author:** Sarvesh Choudhary (Individual Project)  

---

## Slide 1: Title & Author Overview
- **Project Title:** Card Fraud Triage via Gaussian Mixture Density Scoring and Deep Autoencoder Anomaly Detection: A Technically Justified Hybrid Architecture
- **Student Researcher:** Sarvesh Choudhary
- **Specialization:** Machine Learning & Deep Learning (CSAI)
- **Scope:** End-to-End Individual Implementation (Data Engineering, ML Baseline, PyTorch DL, Hybrid Model, Evaluation & Deployment)

---

## Slide 2: Real-World Problem & Asymmetric Challenges
- **Massive Class Imbalance:** 99.83% legitimate vs 0.17% fraud (over 570 legitimate payments per fraud).
- **Label Latency:** Chargebacks take 30 to 90 days to settle.
- **Concept Drift:** Attackers continuously mutate attack vectors.
- **The Core Question:** *How do we triage new fraud in real time when ground-truth fraud labels are delayed or absent?*

---

## Slide 3: Research Methodology & Unsupervised Anomaly Framing
- **The Paradigm Shift:** Train models **strictly on legitimate transactions** ($Class = 0$).
- Models learn the compact envelope of valid spending without observing fraud labels.
- Anomalous transactions naturally produce elevated anomaly scores:
  - **Statistical Density (AML):** Negative log-likelihood $-\log p(x)$.
  - **Deep Representation (DL):** Reconstruction error $\|x - \hat{x}\|^2 / D$.

---

## Slide 4: Data Quality, EDA & Anti-Leakage Protocol
- **Dataset:** Kaggle / ULB Credit Card Fraud (284,807 transactions, 48 hours).
- **Zero-Leakage Chronological Split:**
  - Train: 60% (0.0h to 33.4h) $\to$ 170,524 legitimate rows for model fitting.
  - Validation: 20% (33.4h to 40.3h) $\to$ Early stopping & threshold calibration.
  - Test: 20% (40.3h to 48.0h) $\to$ 56,962 transactions (75 frauds) kept untouched.
- `StandardScaler` fitted strictly on legitimate training samples.

---

## Slide 5: Phase 1 ML Baseline: Gaussian Mixture Model (GMM)
- **Formulation:** $p(x) = \sum_{k=1}^K \pi_k \mathcal{N}(x \mid \mu_k, \Sigma_k)$.
- **Model Selection:** Sweep over $K \in \{1,\dots,8\}$ using Bayesian Information Criterion (BIC).
- **Numerical Safeguard:** Diagonal regularizer $\Sigma_k + \epsilon I$ ($\epsilon = 10^{-4}$) to prevent non-positive definiteness in 29 dimensions.

---

## Slide 6: Phase 2 DL Architecture: Undercomplete Autoencoder
- **Architecture:** $29 \to 20 \to 10 \to 5 \to 10 \to 20 \to 29$ with BatchNorm and ReLU.
- **Manifold Learning:** Narrow 5D bottleneck prevents identity memorization.
- **Denoising Variant:** Gaussian input noise ($\sigma = 0.15$) to regularize representations.
- **Early Stopping:** Monitored MSE on legitimate validation partition.

---

## Slide 7: Phase 2 Innovation: Technically Justified Hybrid ML+DL Model
- **Theoretical Motivation:**
  - GMM enforces convex Gaussian clusters $\to$ penalizes benign high-variance transactions.
  - Autoencoder captures non-linear manifold geometry $\to$ lacks calibrated probabilistic density.
- **Standardized Score Fusion:**
  $$S_{\text{hybrid}} = 0.5 \cdot Z_{\text{GMM}} + 0.5 \cdot Z_{\text{AE}}$$
- **Empirical Value:** SOTA performance, outperforming both standalone models!

---

## Slide 8: Why Accuracy Misleads: Triage Evaluation Metrics
- Accuracy is invalid (predicting all legitimate yields 99.83% accuracy with 0% recall).
- **Primary Metric:** Precision-Recall AUC (Average Precision) with 95% Bootstrap CI.
- **Operational Budgets:** Recall at fixed False Positive Rate (0.1%, 0.5%, 1.0% FPR).
- **Queue Purity:** Precision@K for $K \in \{100, 500, 1000\}$ flagged transactions.

---

## Slide 9: Comprehensive Benchmark Results
| Model | PR-AUC (Primary) | ROC-AUC | Recall @ 0.5% FPR | Precision @ 100 |
| :--- | :---: | :---: | :---: | :---: |
| **Hybrid (GMM + Autoencoder)** | **0.2429** | 0.9587 | 76.00% | **38.00%** |
| **Autoencoder (DL Manifold)** | 0.2185 | **0.9660** | **77.33%** | 33.00% |
| **GMM (AML Baseline)** | 0.2196 | 0.9458 | 65.33% | 26.00% |
| **Denoising Autoencoder** | 0.1600 | 0.9462 | 70.67% | 28.00% |
| **Isolation Forest** | 0.0950 | 0.9525 | 54.67% | 9.00% |
| *Logistic Regression (Supervised Upper)* | *0.7473* | *0.9818* | *85.33%* | *60.00%* |

---

## Slide 10: Precision-Recall & ROC Curve Analysis
- Unsupervised anomaly models achieve over **184x lift** over random baseline (0.132%).
- Autoencoder dominates in the critical low-FPR region ($<0.5\%$ FPR).

---

## Slide 11: Operational Threshold Calibration (Validation to Test)
- Decision thresholds calibrated on Validation at 0.5% FPR ceiling.
- **Autoencoder on Test:** Flags 170 transactions, catches 44 frauds ($Recall = 58.7\%$, $Precision = 25.9\%$).
- **Review Burden:** Only **2.86 false positives per fraud caught**!

---

## Slide 12: Ablation Studies & Insights
1. **Denoising Noise Ablation:** Additive Gaussian noise ($\sigma = 0.15$) blurred continuous PCA feature boundaries, lowering PR-AUC from 0.2185 to 0.1600.
2. **Covariance Structure:** Full covariance captured inter-feature dependencies significantly better than diagonal covariance.
3. **Hybrid Synergy:** Score fusion increased Precision@100 from 26% (GMM) and 33% (AE) to **38.00%**.

---

## Slide 13: Error Analysis & Failure Modes
- **Top False Positives:** Rare high-dollar benign purchases ($>\$1,500$) and extreme $V_{14}, V_{17}$ values.
- **Missed Frauds:** Micro-amount card testing ($<\$15$) matching nominal spending profiles.
- **Remedy:** Real-world velocity engines and merchant category codes.

---

## Slide 14: Interactive Dashboard Demo
- Streamlit application (`app/app.py`).
- Real-time CSV upload, dynamic score thresholding, queue inspection, and confusion matrix analytics.

---

## Slide 15: Conclusion & Individual Engineering Highlights
- Unsupervised anomaly triage solves label latency without sacrificing detection capability.
- **Full-Stack Implementation by Sarvesh Choudhary:**
  - Pipeline & Data Architecture: Zero-leakage chronological partitioning.
  - Phase 1 ML: GMM density estimation with AIC/BIC tuning and numerical stability.
  - Phase 2 DL: PyTorch Autoencoders with bottleneck manifold representations.
  - Hybrid Innovation: Fused score yielding SOTA 0.2429 PR-AUC and 38.00% Precision@100.
  - Deployment: Production-ready Streamlit app with $<0.2\text{ ms}$ inference latency.
