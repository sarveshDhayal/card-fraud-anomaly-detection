"""Deep Learning Anomaly Detection: PyTorch Undercomplete & Denoising Autoencoder.

Architecture:
- Input (29) -> Linear(20) -> BatchNorm -> ReLU -> Linear(10) -> BatchNorm -> ReLU -> Linear(5) [Latent Bottleneck]
- Decoder: Linear(5) -> Linear(10) -> BatchNorm -> ReLU -> Linear(20) -> BatchNorm -> ReLU -> Linear(29)

Features:
- Standard undercomplete Autoencoder trained on legitimate samples.
- Denoising Autoencoder variant: injects zero-mean Gaussian noise into inputs during training,
  optimizing MSE against the clean original inputs to encourage robust manifold learning.
- Early stopping based strictly on reconstruction loss on LEGITIMATE validation transactions.
- Anomaly score = Mean Squared Reconstruction Error per sample: ||x - x_recon||^2 / D.
"""

from typing import Dict, Any, Tuple, Optional
import os
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class Autoencoder(nn.Module):
    """Undercomplete tabular autoencoder with bottleneck latent dimension."""

    def __init__(self, input_dim: int = 29, latent_dim: int = 5):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 20),
            nn.BatchNorm1d(20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.BatchNorm1d(10),
            nn.ReLU(),
            nn.Linear(10, latent_dim),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 10),
            nn.BatchNorm1d(10),
            nn.ReLU(),
            nn.Linear(10, 20),
            nn.BatchNorm1d(20),
            nn.ReLU(),
            nn.Linear(20, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon


def train_autoencoder(
    X_train_legit: np.ndarray,
    X_val_legit: np.ndarray,
    input_dim: int = 29,
    latent_dim: int = 5,
    epochs: int = 30,
    batch_size: int = 512,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    patience: int = 6,
    is_denoising: bool = False,
    noise_factor: float = 0.15,
    device: Optional[str] = None,
    seed: int = 42,
) -> Tuple[Autoencoder, Dict[str, Any]]:
    """Train Autoencoder on legitimate samples with early stopping on legitimate validation set."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    # Prepare PyTorch datasets
    train_tensor = torch.tensor(X_train_legit, dtype=torch.float32)
    val_tensor = torch.tensor(X_val_legit, dtype=torch.float32)

    train_dataset = TensorDataset(train_tensor)
    val_dataset = TensorDataset(val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = Autoencoder(input_dim=input_dim, latent_dim=latent_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {"train_loss": [], "val_loss": [], "stopped_epoch": epochs}
    best_loss = float("inf")
    best_weights = copy.deepcopy(model.state_dict())
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        # Training Phase
        model.train()
        train_running_loss = 0.0
        n_batches = 0

        for (batch_x,) in train_loader:
            batch_x = batch_x.to(device)
            target = batch_x.clone()

            if is_denoising:
                noise = torch.randn_like(batch_x) * noise_factor
                input_x = batch_x + noise
            else:
                input_x = batch_x

            optimizer.zero_grad()
            recon = model(input_x)
            loss = criterion(recon, target)
            loss.backward()
            optimizer.step()

            train_running_loss += loss.item()
            n_batches += 1

        avg_train_loss = train_running_loss / max(n_batches, 1)

        # Validation Phase (Always on clean legitimate validation set)
        model.eval()
        val_running_loss = 0.0
        n_val_batches = 0

        with torch.no_grad():
            for (batch_val_x,) in val_loader:
                batch_val_x = batch_val_x.to(device)
                val_recon = model(batch_val_x)
                val_loss = criterion(val_recon, batch_val_x)
                val_running_loss += val_loss.item()
                n_val_batches += 1

        avg_val_loss = val_running_loss / max(n_val_batches, 1)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)

        # Early stopping logic
        if avg_val_loss < best_loss - 1e-5:
            best_loss = avg_val_loss
            best_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                history["stopped_epoch"] = epoch
                break

    # Restore best weights
    model.load_state_dict(best_weights)
    model.eval()

    meta = {
        "is_denoising": is_denoising,
        "noise_factor": noise_factor if is_denoising else 0.0,
        "best_val_loss": best_loss,
        "epochs_trained": len(history["train_loss"]),
        "device": str(device),
        "history": history,
    }

    return model, meta


def score_samples_ae(
    model: Autoencoder,
    X: np.ndarray,
    batch_size: int = 1024,
    device: Optional[str] = None,
) -> np.ndarray:
    """Compute per-sample mean squared reconstruction error as anomaly score.

    Score = mean((x - x_hat)^2) across features.
    Higher score = higher reconstruction error = more suspicious.
    """
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    model.to(device)
    model.eval()

    tensor_data = torch.tensor(X, dtype=torch.float32)
    loader = DataLoader(TensorDataset(tensor_data), batch_size=batch_size, shuffle=False)

    scores_list = []
    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            recon = model(batch_x)
            # Per-sample MSE error across the 29 features
            mse = torch.mean((batch_x - recon) ** 2, dim=1)
            scores_list.append(mse.cpu().numpy())

    return np.concatenate(scores_list, axis=0)


def save_autoencoder(model: Autoencoder, filepath: str = "outputs/models/autoencoder.pt") -> None:
    """Save PyTorch autoencoder state dict."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": model.input_dim,
            "latent_dim": model.latent_dim,
        },
        filepath,
    )


def load_autoencoder(filepath: str = "outputs/models/autoencoder.pt", device: str = "cpu") -> Autoencoder:
    """Load PyTorch autoencoder from saved state dict."""
    if not os.path.exists(filepath):
        alt = filepath.replace("outputs/models/", "models/") if "outputs/" in filepath else os.path.join("outputs", filepath)
        if os.path.exists(alt):
            filepath = alt
        else:
            raise FileNotFoundError(f"Model file not found at {filepath} or {alt}")
    checkpoint = torch.load(filepath, map_location=device)
    model = Autoencoder(
        input_dim=checkpoint.get("input_dim", 29),
        latent_dim=checkpoint.get("latent_dim", 5),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model
