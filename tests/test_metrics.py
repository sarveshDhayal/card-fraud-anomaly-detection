"""Unit tests for evaluation metrics and budget constraints."""

import os
import sys
import unittest
import numpy as np

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.metrics import (
    compute_pr_auc,
    compute_roc_auc,
    recall_at_fpr,
    precision_at_k,
    evaluate_threshold_on_test,
)


class TestMetrics(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        # Synthetic binary evaluation setup (1000 samples, 10 fraud)
        self.y_true = np.zeros(1000, dtype=int)
        self.y_true[:10] = 1

        # Scores: frauds receive higher scores on average
        self.scores = np.random.randn(1000)
        self.scores[:10] += 3.0

    def test_pr_auc_range(self):
        pr = compute_pr_auc(self.y_true, self.scores)
        self.assertGreater(pr, 0.0)
        self.assertLessEqual(pr, 1.0)

    def test_roc_auc_range(self):
        roc = compute_roc_auc(self.y_true, self.scores)
        self.assertGreater(roc, 0.5)
        self.assertLessEqual(roc, 1.0)

    def test_recall_at_fpr(self):
        rec, actual_fpr, thresh = recall_at_fpr(self.y_true, self.scores, target_fpr=0.01)
        self.assertLessEqual(actual_fpr, 0.02)
        self.assertGreaterEqual(rec, 0.0)
        self.assertLessEqual(rec, 1.0)

    def test_precision_at_k(self):
        prec, k_count, thresh = precision_at_k(self.y_true, self.scores, k=20)
        self.assertEqual(k_count, 20)
        self.assertGreaterEqual(prec, 0.0)
        self.assertLessEqual(prec, 1.0)

    def test_threshold_evaluation_metrics(self):
        res = evaluate_threshold_on_test(self.y_true, self.scores, threshold=2.0)
        self.assertIn("tp", res)
        self.assertIn("fp", res)
        self.assertIn("precision", res)
        self.assertIn("recall", res)
        self.assertIn("fp_per_fraud_caught", res)


if __name__ == "__main__":
    unittest.main()
