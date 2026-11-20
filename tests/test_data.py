"""Unit tests for data pipeline and anti-leakage splitting."""

import os
import sys
import unittest
import numpy as np
import pandas as pd

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.data import set_seed, prepare_data, load_raw_data


class TestDataPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        set_seed(42)
        cls.data = prepare_data("data/raw/creditcard.csv")

    def test_chronological_ordering(self):
        df_raw = self.data["df_raw"]
        times = df_raw["Time"].values
        self.assertTrue(np.all(np.diff(times) >= 0), "Data is not chronologically sorted")

    def test_split_proportions(self):
        n_total = len(self.data["df_raw"])
        n_train = len(self.data["train_df"])
        n_val = len(self.data["val_df"])
        n_test = len(self.data["test_df"])
        self.assertEqual(n_train + n_val + n_test, n_total)
        self.assertAlmostEqual(n_train / n_total, 0.60, places=2)
        self.assertAlmostEqual(n_val / n_total, 0.20, places=2)
        self.assertAlmostEqual(n_test / n_total, 0.20, places=2)

    def test_unsupervised_train_isolation(self):
        """Verify train_legit contains ZERO fraudulent transactions."""
        train_legit_y = self.data["train_df"][self.data["train_df"]["Class"] == 0]["Class"]
        self.assertEqual(train_legit_y.sum(), 0, "Train legit subset must have 0 frauds")

    def test_no_nan_in_scaled_data(self):
        self.assertFalse(np.isnan(self.data["X_train_legit"]).any())
        self.assertFalse(np.isnan(self.data["X_val"]).any())
        self.assertFalse(np.isnan(self.data["X_test"]).any())

    def test_feature_dimensions(self):
        self.assertEqual(self.data["X_train_legit"].shape[1], 29)
        self.assertEqual(len(self.data["feature_cols"]), 29)


if __name__ == "__main__":
    unittest.main()
