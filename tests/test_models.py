"""Unit tests for GMM, Autoencoder, and Hybrid models."""

import os
import sys
import unittest
import numpy as np
import torch

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.gmm_model import load_gmm, score_samples_gmm
from src.autoencoder import load_autoencoder, score_samples_ae, Autoencoder


class TestModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gmm = load_gmm("outputs/models/gmm_model.joblib")
        cls.ae = load_autoencoder("outputs/models/autoencoder.pt", device="cpu")
        cls.dummy_x = np.random.randn(20, 29).astype(np.float32)

    def test_gmm_scoring_shape(self):
        scores = score_samples_gmm(self.gmm, self.dummy_x)
        self.assertEqual(scores.shape, (20,))
        self.assertTrue(np.all(np.isfinite(scores)))

    def test_autoencoder_forward_pass(self):
        t_x = torch.tensor(self.dummy_x, dtype=torch.float32)
        with torch.no_grad():
            recon = self.ae(t_x)
        self.assertEqual(recon.shape, (20, 29))

    def test_autoencoder_scoring_shape_and_non_negativity(self):
        scores = score_samples_ae(self.ae, self.dummy_x)
        self.assertEqual(scores.shape, (20,))
        self.assertTrue(np.all(scores >= 0))
        self.assertTrue(np.all(np.isfinite(scores)))

    def test_autoencoder_bottleneck_dimension(self):
        self.assertEqual(self.ae.latent_dim, 5)
        self.assertEqual(self.ae.input_dim, 29)


if __name__ == "__main__":
    unittest.main()
