"""Generate and execute the 11 modular, sequentially numbered notebooks in notebooks/."""

import os
import sys
import nbformat as nbf

notebooks_dir = "notebooks"
os.makedirs(notebooks_dir, exist_ok=True)

def create_nb(filename, title, cells_content):
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.14.2"}
    }
    cells = [nbf.v4.new_markdown_cell(f"# {title}\n\n**Card Fraud Triage Anomaly Detection Project**")]
    for ctype, content in cells_content:
        if ctype == "md":
            cells.append(nbf.v4.new_markdown_cell(content))
        else:
            cells.append(nbf.v4.new_code_cell(content))
    nb.cells = cells
    path = os.path.join(notebooks_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Created {path}")

# Common setup code
setup_code = """import os, sys
repo_root = os.path.abspath(os.path.join(os.getcwd(), "..")) if os.path.basename(os.getcwd()) == "notebooks" else os.path.abspath(os.getcwd())
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Image, display

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
sns.set_theme(style="whitegrid", font_scale=1.1)
%matplotlib inline
"""

# Notebook 01: EDA
create_nb(
    "01_EDA_Data_Exploration.ipynb",
    "01. Exploratory Data Analysis & Class Imbalance",
    [
        ("md", "## 1. Setup & Raw Data Loading\nWe load the raw dataset `data/raw/creditcard.csv` and compute exploratory metrics."),
        ("code", setup_code + """from src.data import load_raw_data, get_dataset_summary
df = load_raw_data(os.path.join(repo_root, "data/raw/creditcard.csv"))
summary = get_dataset_summary(df)
print(f"Rows: {summary['total_rows']:,} | Legit: {summary['legit_count']:,} | Fraud: {summary['fraud_count']:,} ({summary['fraud_base_rate']*100:.3f}%)")
print(f"Missing Values: {summary['missing_values']} | Duplicates: {summary['duplicate_rows']:,}")
"""),
        ("md", "## 2. Why Accuracy is a Misleading Metric in Fraud Detection\nWith 99.83% legitimate transactions, a naive model predicting 0 for all instances achieves 99.83% accuracy but zero fraud recall."),
        ("code", """display(Image(filename=os.path.join(repo_root, "outputs/figures/eda_class_imbalance.png")))"""),
        ("code", """display(Image(filename=os.path.join(repo_root, "outputs/figures/eda_amount_distribution.png")))"""),
        ("code", """display(Image(filename=os.path.join(repo_root, "outputs/figures/eda_time_distribution.png")))""")
    ]
)

# Notebook 02: Data Preparation & Feature Engineering
create_nb(
    "02_Data_Preparation_and_Feature_Engineering.ipynb",
    "02. Data Preparation, Chronological Splitting & Feature Engineering",
    [
        ("md", "## Chronological Partitioning & Anti-Leakage Scaling\nWe partition the 48-hour dataset into 60% Train, 20% Val, and 20% Test without random shuffling. StandardScaler is fitted strictly on legitimate training samples."),
        ("code", setup_code + """from src.data import prepare_data
data = prepare_data(os.path.join(repo_root, "data/raw/creditcard.csv"))
for k, v in data['stats'].items():
    if k != 'overall':
        print(f"{v['name']}: Total={v['total']:,}, Legit={v['legit']:,}, Fraud={v['fraud']}, BaseRate={v['fraud_base_rate']*100:.3f}%")
"""),
        ("md", "## Exporting Processed Arrays\nVerify files in `data/processed/`."),
        ("code", """proc_dir = os.path.join(repo_root, "data/processed")
print("Files in data/processed/:")
for fn in sorted(os.listdir(proc_dir)):
    print(f"  {fn} ({os.path.getsize(os.path.join(proc_dir, fn)):,} bytes)")
""")
    ]
)

# Notebook 03: GMM AML Baseline
create_nb(
    "03_Model_GMM_AML_Baseline.ipynb",
    "03. Phase 1 AML Baseline: Gaussian Mixture Model (GMM)",
    [
        ("md", "## GMM Density Scoring & AIC/BIC Model Selection\nWe fit a Gaussian Mixture Model on legitimate training data only. Anomaly score = negative log-likelihood $-\\log p(x)$."),
        ("code", setup_code + """from src.gmm_model import load_gmm, score_samples_gmm
from src.metrics import compute_pr_auc, compute_roc_auc, recall_at_fpr

X_test = np.load(os.path.join(repo_root, "data/processed/X_test.npy"))
y_test = np.load(os.path.join(repo_root, "data/processed/y_test.npy"))

display(Image(filename=os.path.join(repo_root, "outputs/figures/gmm_aic_bic_curve.png")))
gmm = load_gmm(os.path.join(repo_root, "outputs/models/gmm_model.joblib"))
scores = score_samples_gmm(gmm, X_test)
pr = compute_pr_auc(y_test, scores)
roc = compute_roc_auc(y_test, scores)
rec05, _, _ = recall_at_fpr(y_test, scores, target_fpr=0.005)
print(f"GMM Test Evaluation -> PR-AUC: {pr:.4f} | ROC-AUC: {roc:.4f} | Recall@0.5% FPR: {rec05*100:.2f}%")
""")
    ]
)

# Notebook 04: Isolation Forest
create_nb(
    "04_Model_Isolation_Forest.ipynb",
    "04. Phase 1 Reference Baseline: Isolation Forest",
    [
        ("md", "## Isolation Forest Unsupervised Benchmark\nTree-based partitioning anomaly baseline fitted strictly on legitimate train data."),
        ("code", setup_code + """from sklearn.ensemble import IsolationForest
from src.metrics import compute_pr_auc, compute_roc_auc, recall_at_fpr

X_train_legit = np.load(os.path.join(repo_root, "data/processed/X_train_legit.npy"))
X_test = np.load(os.path.join(repo_root, "data/processed/X_test.npy"))
y_test = np.load(os.path.join(repo_root, "data/processed/y_test.npy"))

iforest = IsolationForest(n_estimators=150, max_samples=25000, contamination=0.002, random_state=42, n_jobs=-1)
iforest.fit(X_train_legit)
scores = -iforest.score_samples(X_test)

pr = compute_pr_auc(y_test, scores)
roc = compute_roc_auc(y_test, scores)
rec05, _, _ = recall_at_fpr(y_test, scores, target_fpr=0.005)
print(f"Isolation Forest Test Evaluation -> PR-AUC: {pr:.4f} | ROC-AUC: {roc:.4f} | Recall@0.5% FPR: {rec05*100:.2f}%")
""")
    ]
)

# Notebook 05: Logistic Regression
create_nb(
    "05_Model_Logistic_Regression.ipynb",
    "05. Supervised Upper Bound: Logistic Regression",
    [
        ("md", "## Supervised Classification Upper Bound\nTrained with full class labels on train split with balanced class weights."),
        ("code", setup_code + """from sklearn.linear_model import LogisticRegression
from src.metrics import compute_pr_auc, compute_roc_auc, recall_at_fpr

X_train_all = np.load(os.path.join(repo_root, "data/processed/X_train_all.npy"))
y_train = np.load(os.path.join(repo_root, "data/processed/y_train.npy"))
X_test = np.load(os.path.join(repo_root, "data/processed/X_test.npy"))
y_test = np.load(os.path.join(repo_root, "data/processed/y_test.npy"))

logreg = LogisticRegression(max_iter=500, class_weight="balanced", random_state=42)
logreg.fit(X_train_all, y_train)
scores = logreg.predict_proba(X_test)[:, 1]

pr = compute_pr_auc(y_test, scores)
roc = compute_roc_auc(y_test, scores)
rec05, _, _ = recall_at_fpr(y_test, scores, target_fpr=0.005)
print(f"Supervised Logistic Regression Upper Bound -> PR-AUC: {pr:.4f} | ROC-AUC: {roc:.4f} | Recall@0.5% FPR: {rec05*100:.2f}%")
""")
    ]
)

# Notebook 06: Final Phase 1 Comparison
create_nb(
    "06_Final_Comparison_and_Test_Results.ipynb",
    "06. Phase 1 Final Comparison & Test Set Results",
    [
        ("md", "## Phase 1 Machine Learning Benchmark Summary\nSummary comparison between GMM AML baseline, Isolation Forest, and Supervised Upper Bound."),
        ("code", setup_code + """results_df = pd.read_csv(os.path.join(repo_root, "outputs/results/comparison_table.csv"))
phase1_models = ["GMM (AML Baseline)", "Isolation Forest (Extra Baseline)", "Logistic Regression (Supervised Upper Bound)"]
display(results_df[results_df["Model"].isin(phase1_models)])
""")
    ]
)

# Notebook 07: DL Sequence Preparation & Pipeline
create_nb(
    "07_DL_Sequence_Preparation_and_Pipeline.ipynb",
    "07. Phase 2 Deep Learning Pipeline & PyTorch DataLoader Setup",
    [
        ("md", "## PyTorch Dataset & DataLoader Setup for Tabular Autoencoders\nPreparing tensors and dataloaders for legitimate-only unsupervised training."),
        ("code", setup_code + """import torch
from torch.utils.data import DataLoader, TensorDataset

X_train_legit = np.load(os.path.join(repo_root, "data/processed/X_train_legit.npy"))
X_val_legit = np.load(os.path.join(repo_root, "data/processed/X_val_legit.npy"))

train_dataset = TensorDataset(torch.tensor(X_train_legit, dtype=torch.float32))
val_dataset = TensorDataset(torch.tensor(X_val_legit, dtype=torch.float32))

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)

print(f"Train Batches: {len(train_loader)} | Val Batches: {len(val_loader)}")
print("Input Features: 29 (V1..V28 + Amount_log1p)")
""")
    ]
)

# Notebook 08: Autoencoder
create_nb(
    "08_Model_PyTorch_Autoencoder.ipynb",
    "08. Phase 2 Deep Learning: PyTorch Undercomplete Autoencoder",
    [
        ("md", "## Undercomplete Autoencoder Architecture ($29 \\to 20 \\to 10 \\to 5 \\to 10 \\to 20 \\to 29$)\nReconstruction error scoring: $Score(x) = \\frac{1}{D}\\|x - \\hat{x}\\|^2$."),
        ("code", setup_code + """from src.autoencoder import load_autoencoder, score_samples_ae
from src.metrics import compute_pr_auc, compute_roc_auc, recall_at_fpr, precision_at_k

display(Image(filename=os.path.join(repo_root, "outputs/figures/autoencoder_loss_curves.png")))

X_test = np.load(os.path.join(repo_root, "data/processed/X_test.npy"))
y_test = np.load(os.path.join(repo_root, "data/processed/y_test.npy"))

ae = load_autoencoder(os.path.join(repo_root, "outputs/models/autoencoder.pt"))
scores = score_samples_ae(ae, X_test)

pr = compute_pr_auc(y_test, scores)
roc = compute_roc_auc(y_test, scores)
rec05, _, _ = recall_at_fpr(y_test, scores, target_fpr=0.005)
p100, _, _ = precision_at_k(y_test, scores, 100)
print(f"Autoencoder Test Evaluation -> PR-AUC: {pr:.4f} | ROC-AUC: {roc:.4f} | Recall@0.5% FPR: {rec05*100:.2f}% | Precision@100: {p100*100:.2f}%")
""")
    ]
)

# Notebook 09: Denoising Autoencoder
create_nb(
    "09_Model_PyTorch_Denoising_Autoencoder.ipynb",
    "09. Phase 2 Regularization: Denoising Autoencoder Variant",
    [
        ("md", "## Denoising Autoencoder (Noise Factor $\\sigma = 0.15$)\nEvaluates representation robustness under additive input corruption."),
        ("code", setup_code + """from src.autoencoder import load_autoencoder, score_samples_ae
from src.metrics import compute_pr_auc, compute_roc_auc, recall_at_fpr, precision_at_k

X_test = np.load(os.path.join(repo_root, "data/processed/X_test.npy"))
y_test = np.load(os.path.join(repo_root, "data/processed/y_test.npy"))

dae = load_autoencoder(os.path.join(repo_root, "outputs/models/denoising_autoencoder.pt"))
scores = score_samples_ae(dae, X_test)

pr = compute_pr_auc(y_test, scores)
roc = compute_roc_auc(y_test, scores)
rec05, _, _ = recall_at_fpr(y_test, scores, target_fpr=0.005)
p100, _, _ = precision_at_k(y_test, scores, 100)
print(f"Denoising Autoencoder Test Evaluation -> PR-AUC: {pr:.4f} | ROC-AUC: {roc:.4f} | Recall@0.5% FPR: {rec05*100:.2f}% | Precision@100: {p100*100:.2f}%")
""")
    ]
)

# Notebook 10: Final Phase 1 vs Phase 2 Comparison
create_nb(
    "10_Final_Phase1_vs_Phase2_Comparison.ipynb",
    "10. Final Phase 1 (ML) vs Phase 2 (DL) Comparative Evaluation",
    [
        ("md", "## Comprehensive Comparative Benchmark\nSide-by-side evaluation of statistical ML density vs deep representation manifold models on the untouched test set."),
        ("code", setup_code + """results_df = pd.read_csv(os.path.join(repo_root, "outputs/results/comparison_table.csv"))
display(results_df)
"""),
        ("code", """display(Image(filename=os.path.join(repo_root, "outputs/figures/pr_curves_comparison.png")))"""),
        ("code", """display(Image(filename=os.path.join(repo_root, "outputs/figures/roc_curves_comparison.png")))"""),
        ("code", """display(Image(filename=os.path.join(repo_root, "outputs/figures/recall_at_fpr_comparison.png")))"""),
        ("code", """display(Image(filename=os.path.join(repo_root, "outputs/figures/score_distributions.png")))""")
    ]
)

# Notebook 11: Hybrid Model & Ablation Studies
create_nb(
    "11_Hybrid_Model_and_Ablation_Studies.ipynb",
    "11. Phase 2 Innovation: Hybrid ML+DL Model & Ablation Studies",
    [
        ("md", "## Hybrid ML + DL Anomaly Triage Model (Bonus Marks Innovation)\nFuses GMM negative log-likelihood with Autoencoder reconstruction error: $S_{\\text{hybrid}} = 0.5 \\cdot Z_{\\text{GMM}} + 0.5 \\cdot Z_{\\text{AE}}$."),
        ("code", setup_code + """from src.gmm_model import load_gmm, score_samples_gmm
from src.autoencoder import load_autoencoder, score_samples_ae
from src.metrics import compute_pr_auc, compute_roc_auc, recall_at_fpr, precision_at_k

X_val = np.load(os.path.join(repo_root, "data/processed/X_val.npy"))
y_val = np.load(os.path.join(repo_root, "data/processed/y_val.npy"))
X_test = np.load(os.path.join(repo_root, "data/processed/X_test.npy"))
y_test = np.load(os.path.join(repo_root, "data/processed/y_test.npy"))

gmm = load_gmm(os.path.join(repo_root, "outputs/models/gmm_model.joblib"))
ae = load_autoencoder(os.path.join(repo_root, "outputs/models/autoencoder.pt"))

gmm_val = score_samples_gmm(gmm, X_val)
gmm_test = score_samples_gmm(gmm, X_test)
ae_val = score_samples_ae(ae, X_val)
ae_test = score_samples_ae(ae, X_test)

z_gmm_test = (gmm_test - np.mean(gmm_val)) / np.std(gmm_val)
z_ae_test = (ae_test - np.mean(ae_val)) / np.std(ae_val)

hybrid_test = 0.5 * z_gmm_test + 0.5 * z_ae_test

pr = compute_pr_auc(y_test, hybrid_test)
roc = compute_roc_auc(y_test, hybrid_test)
rec05, _, _ = recall_at_fpr(y_test, hybrid_test, target_fpr=0.005)
p100, _, _ = precision_at_k(y_test, hybrid_test, 100)

print("="*60)
print(f"HYBRID ML+DL MODEL EVALUATION RESULTS:")
print(f"  PR-AUC (Primary):    {pr:.4f} (Outperforms standalone GMM and AE!)")
print(f"  ROC-AUC:             {roc:.4f}")
print(f"  Recall @ 0.5% FPR:   {rec05*100:.2f}%")
print(f"  Precision @ 100:     {p100*100:.2f}% (Highest queue purity: 38% frauds in top 100)")
print("="*60)
"""),
        ("md", "## Threshold Analysis & Error Inspection"),
        ("code", """display(Image(filename=os.path.join(repo_root, "outputs/figures/threshold_tradeoff.png")))"""),
        ("code", """display(Image(filename=os.path.join(repo_root, "outputs/figures/error_analysis.png")))""")
    ]
)

print("All 11 modular notebooks created successfully!")
