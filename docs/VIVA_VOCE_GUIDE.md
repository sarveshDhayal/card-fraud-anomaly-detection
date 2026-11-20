# CSAI Master Individual Viva Voce Assessment Guide

**Project:** Card Fraud Triage via Gaussian Mixture Density Scoring and Deep Autoencoder Anomaly Detection: A Technically Justified Hybrid Approach  
**Candidate:** Sarvesh Choudhary  
**Type:** Individual Research & Engineering Project (CSAI Advanced ML & DL)

---

## Candidate Overview: Full Architecture Ownership

As the sole researcher and developer of this project, you own the entire end-to-end implementation across both evaluation phases:

1. **Phase 1 (Machine Learning):** Anti-leakage chronological pipeline, EDA on severe imbalance, GMM negative log-likelihood density estimation, AIC/BIC hyperparameter sweep, covariance regularization (`reg_covar`), Isolation Forest baseline, and supervised Logistic Regression upper bound.
2. **Phase 2 (Deep Learning):** Undercomplete PyTorch Autoencoder ($29 \to 20 \to 10 \to 5 \to 10 \to 20 \to 29$), early stopping tracking legitimate validation MSE, Denoising Autoencoder variant ($\sigma = 0.15$), and ablation studies.
3. **Phase 2 Hybrid Innovation (Bonus Marks):** Standardized score fusion combining parametric GMM density with non-linear Autoencoder manifold reconstruction ($S_{\text{hybrid}} = 0.5 \cdot Z_{\text{GMM}} + 0.5 \cdot Z_{\text{AE}}$), achieving state-of-the-art **0.2429 PR-AUC** and **38.00% Precision@100**.
4. **Metrics & Deployment:** PR-AUC with 95% bootstrap confidence intervals, Recall@FPR budgets (0.1%, 0.5%, 1.0%), validation-to-test threshold transfer, and an interactive Streamlit triage application.

---

## Individual Viva Voce Question Bank (Top 15 Questions & Exact Answers)

### Module 1: Data Architecture & Anti-Leakage Protocol
**Q1: Why did you use chronological splitting instead of random train_test_split (e.g. 80/20)?**  
*Answer:* In payment card transactions, data is generated sequentially over time. Random shuffling causes severe temporal lookahead leakage because future transaction patterns (e.g., fraudulent spending sprees or merchant trends) bleed into the training set. Chronological splitting (first 60% Train, next 20% Val, final 20% Test) ensures the model strictly learns from past patterns and is tested on out-of-time future data.

**Q2: How did you ensure there was zero data leakage when scaling features?**  
*Answer:* I fit `StandardScaler` strictly on the *legitimate transactions of the training set only* (`train_df[train_df['Class'] == 0]`). I never touched the validation or test sets during fitting; I only called `.transform()`. Furthermore, fraud labels were completely excluded from the training split.

**Q3: Why was `Time` excluded from the feature matrix?**  
*Answer:* The dataset only spans 48 hours (~2 days). If `Time` were included as an input feature, the models would memorize time-of-day or day-of-week quirks specific to that 48-hour window, leading to spurious temporal overfitting rather than learning genuine spending behavior.

---

### Module 2: Statistical AML & Gaussian Mixture Model (GMM)
**Q4: What is the mathematical formulation of your GMM anomaly score?**  
*Answer:* I model the legitimate feature density as $p(x) = \sum_{k=1}^K \pi_k \mathcal{N}(x \mid \mu_k, \Sigma_k)$. The anomaly score is the negative log-likelihood: $Score(x) = -\log p(x)$. Transactions in dense, high-probability regions produce small scores; rare, outlier transactions in low-density regions produce large positive anomaly scores.

**Q5: How did you select the number of components and covariance type?**  
*Answer:* I evaluated $K \in \{1, \dots, 8\}$ across `diag` and `full` covariance structures using the Bayesian Information Criterion (BIC): $\text{BIC} = -2\ln\hat{L} + p\ln(N)$. BIC penalizes model complexity more heavily than AIC to guard against overfitting. Full covariance with $K=8$ components achieved the lowest BIC on the legitimate training subset.

**Q6: What is `reg_covar` and why is it necessary in 29 dimensions?**  
*Answer:* In 29-dimensional continuous feature spaces, covariance matrices can become ill-conditioned, singular, or non-positive-definite due to local linear correlations. `reg_covar = 1e-4` adds a small constant to the diagonal of the covariance matrix ($\Sigma_k + \epsilon I$), guaranteeing positive definiteness and preventing numerical underflow during determinant and inverse calculations.

---

### Module 3: Deep Learning & Autoencoder Architecture
**Q7: Explain the architecture of your undercomplete Autoencoder.**  
*Answer:* The network employs an intentional bottleneck: Encoder ($29 \to 20 \to 10 \to 5$) and Decoder ($5 \to 10 \to 20 \to 29$) with Batch Normalization and ReLU non-linearities. The bottleneck latent dimension ($d=5$) forces the network to learn a compressed, non-linear manifold of legitimate transactions rather than memorizing an identity mapping.

**Q8: How does the Autoencoder produce an anomaly score?**  
*Answer:* The anomaly score is the per-sample Mean Squared Reconstruction Error: $Score(x) = \frac{1}{D}\sum_{j=1}^D (x_j - \hat{x}_j)^2$. Because the network was trained exclusively to reconstruct legitimate patterns, anomalous fraudulent transactions cannot be reconstructed accurately through the bottleneck, yielding elevated reconstruction error.

**Q9: How was early stopping configured to prevent overfitting?**  
*Answer:* Early stopping tracked MSE reconstruction loss strictly on *legitimate validation transactions* with a patience of 6 epochs. When validation loss failed to improve by $10^{-5}$, training halted and the optimal model weights were restored.

---

### Module 4: Hybrid Innovation & Ablation Studies (Bonus Marks Criteria)
**Q10: What is the technical justification for your Hybrid ML+DL approach?**  
*Answer:* GMM enforces convex Gaussian clusters, which in 29 dimensions struggle with high-variance legitimate outliers (e.g., rare high-dollar purchases). Standalone Autoencoders capture non-linear manifold geometry but lack global probabilistic density calibration. My Hybrid model computes standardized Z-scores of both signals calibrated on the validation set: $S_{\text{hybrid}} = 0.5 \cdot Z_{\text{GMM}} + 0.5 \cdot Z_{\text{AE}}$. This fuses parametric density rarity with deep reconstruction deviation.

**Q11: What empirical value did your Hybrid model demonstrate over individual models?**  
*Answer:* On the untouched test set, the Hybrid model achieved a PR-AUC of **0.2429**, outperforming standalone GMM (0.2196) and standalone Autoencoder (0.2185). Even more importantly for operational queues, its **Precision@100 jumped to 38.00%** (vs 26.00% for GMM and 33.00% for AE), providing a significantly cleaner queue for fraud investigators.

**Q12: What did your ablation study show regarding the Denoising Autoencoder?**  
*Answer:* In the Denoising Autoencoder, adding Gaussian noise ($\sigma = 0.15$) regularized representation learning but slightly blurred fine-grained reconstruction differences for low-magnitude fraud anomalies, reducing PR-AUC from 0.2185 to 0.1600. This ablation proved that for PCA-transformed continuous features, clean undercomplete bottleneck compression preserves sharp anomaly signals better than additive input noise.

---

### Module 5: Triage Metrics, Threshold Transfer & Deployment
**Q13: Why is Accuracy completely invalid, and why is PR-AUC superior to ROC-AUC here?**  
*Answer:* In a dataset with 0.17% fraud prevalence, predicting all transactions are legitimate gives 99.83% accuracy but 0% recall. ROC-AUC is misleading because its False Positive Rate denominator is dominated by 56,887 true negatives. PR-AUC focuses exclusively on Precision (triage purity) and Recall (fraud detection), making it the gold standard under extreme class imbalance.

**Q14: How did you select decision thresholds, and why did you not select them on the test set?**  
*Answer:* Selecting thresholds on the test set is data leakage. I calibrated operating thresholds strictly on the validation set (Target FPR = 0.5%, Target FPR = 1.0%, Max F1) and transferred them unchanged to the test set. At a 0.5% FPR threshold, the Autoencoder detected 58 of 75 test frauds (77.3% recall) with a review burden of only 5.3 false positives per fraud caught.

**Q15: How does your system operate in a live payment gateway?**  
*Answer:* The model operates as an asynchronous Tier-1 anomaly triage filter. PyTorch inference takes $<0.2\text{ ms}$ on CPU per transaction, well within the 50 ms payment authorization SLA. High-scoring transactions can be routed to step-up authentication (OTP/biometrics) or sent to the human review queue, while clean transactions pass with zero customer friction.
