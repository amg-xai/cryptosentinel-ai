"""
Autoencoder and Variational Autoencoder for anomaly detection.

WHY Autoencoder over Isolation Forest:
  IF measures isolation depth — a geometric property.
  AE learns the actual data distribution — what normal looks like.
  High reconstruction error = the model has never seen this pattern = anomalous.

WHY train on licit-only:
  We teach the model what NORMAL looks like.
  Illicit transactions reconstruct poorly because they were never in training.
  Reconstruction error becomes the anomaly score.

WHY VAE over plain AE:
  VAE learns a probabilistic latent space (mean + variance per dimension).
  This gives uncertainty estimates on anomaly scores.
  The KL divergence term regularizes the latent space — prevents overfitting.
  Better generalization to unseen normal patterns.

Architecture:
  Encoder: 165 -> 128 -> 64 -> 32
  Latent:  32 -> 16 (VAE: split into mu and log_var)
  Decoder: 16 -> 32 -> 64 -> 128 -> 165
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config.logging_config import get_logger

logger = get_logger(__name__)

MODELS_DIR = Path("data/models")


class Encoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        # VAE: separate heads for mean and log variance
        self.mu_head = nn.Linear(32, latent_dim)
        self.logvar_head = nn.Linear(32, latent_dim)

    def forward(self, x: torch.Tensor):
        h = self.net(x)
        mu = self.mu_head(h)
        log_var = self.logvar_head(h)
        return mu, log_var


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class VAE(nn.Module):
    """
    Variational Autoencoder.
    Learns a probabilistic latent representation of normal transactions.
    """

    def __init__(self, input_dim: int = 165, latent_dim: int = 16):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)

    def reparameterize(
        self,
        mu: torch.Tensor,
        log_var: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reparameterization trick: z = mu + eps * std
        Allows gradients to flow through the sampling operation.
        During inference (eval mode), returns mu directly.
        """
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, mu, log_var

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample reconstruction error — the anomaly score."""
        self.eval()
        with torch.no_grad():
            x_recon, _, _ = self.forward(x)
            error = torch.mean((x - x_recon) ** 2, dim=1)
        return error


def vae_loss(
    x: torch.Tensor,
    x_recon: torch.Tensor,
    mu: torch.Tensor,
    log_var: torch.Tensor,
    kl_weight: float = 0.001,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    VAE loss = reconstruction loss + KL divergence.

    Reconstruction loss: MSE between input and output.
    KL divergence: regularizes latent space toward N(0,1).
    kl_weight: controls regularization strength.
      Too high -> latent space collapses (mode collapse).
      Too low -> VAE behaves like plain AE.
      0.001 is a safe starting point.
    """
    recon_loss = nn.functional.mse_loss(x_recon, x, reduction="mean")
    kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    total_loss = recon_loss + kl_weight * kl_loss
    return total_loss, recon_loss, kl_loss


class ThreatAutoencoder:
    """
    VAE wrapper with training, scoring, and threshold management.
    """

    def __init__(
        self,
        input_dim: int = 165,
        latent_dim: int = 16,
        learning_rate: float = 1e-3,
        kl_weight: float = 0.001,
    ):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.kl_weight = kl_weight

        # Device selection: CUDA > CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("device_selected", device=str(self.device))

        self.model = VAE(input_dim, latent_dim).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-5,
        )

        self._threshold: float = 0.5
        self._threshold_raw: float = 0.0
        self._is_fitted: bool = False
        self.training_history: list[dict] = []

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        train_mask: np.ndarray,
        epochs: int = 50,
        batch_size: int = 512,
    ) -> list[dict]:
        """
        Train VAE on licit transactions only.
        Records loss history for monitoring.
        """
        # Licit only: label=0 in binary encoding
        licit_mask = train_mask & (y_train == 0)
        X_licit = X_train[licit_mask]

        logger.info(
            "autoencoder_training_started",
            device=str(self.device),
            training_samples=len(X_licit),
            epochs=epochs,
            batch_size=batch_size,
            latent_dim=self.latent_dim,
        )

        # Create DataLoader
        X_tensor = torch.tensor(X_licit, dtype=torch.float32)
        dataset = TensorDataset(X_tensor)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=self.device.type == "cuda",
            num_workers=0,
        )

        self.model.train()
        history = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_recon = 0.0
            epoch_kl = 0.0
            n_batches = 0

            for (batch,) in loader:
                batch = batch.to(self.device)
                self.optimizer.zero_grad()

                x_recon, mu, log_var = self.model(batch)
                loss, recon_loss, kl_loss = vae_loss(
                    batch,
                    x_recon,
                    mu,
                    log_var,
                    kl_weight=self.kl_weight,
                )

                loss.backward()
                # Gradient clipping — prevents exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                epoch_loss += loss.item()
                epoch_recon += recon_loss.item()
                epoch_kl += kl_loss.item()
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            avg_recon = epoch_recon / n_batches
            avg_kl = epoch_kl / n_batches

            record = {
                "epoch": epoch + 1,
                "total_loss": round(avg_loss, 6),
                "recon_loss": round(avg_recon, 6),
                "kl_loss": round(avg_kl, 6),
            }
            history.append(record)

            if (epoch + 1) % 10 == 0:
                logger.info("autoencoder_epoch", **record)

        self._is_fitted = True
        self.training_history = history

        logger.info(
            "autoencoder_training_complete",
            final_loss=history[-1]["total_loss"],
            final_recon=history[-1]["recon_loss"],
        )

        return history

    def _get_raw_scores(self, X: np.ndarray) -> np.ndarray:
        """Raw reconstruction errors (not normalized)."""
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            errors = self.model.reconstruction_error(X_tensor)

        return errors.cpu().numpy()

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Normalized anomaly scores in [0, 1].
        Higher = more anomalous = higher risk.
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call train() first.")

        raw = self._get_raw_scores(X)
        r_min, r_max = raw.min(), raw.max()
        if r_max > r_min:
            normalized = (raw - r_min) / (r_max - r_min)
        else:
            normalized = np.full_like(raw, 0.5)

        return normalized.astype(np.float32)

    def set_threshold_from_percentile(
        self,
        X_licit: np.ndarray,
        percentile: float = 95.0,
    ) -> float:
        """
        Set threshold as the Nth percentile of licit reconstruction errors.
        Transactions with error above this are flagged as anomalous.

        WHY percentile approach:
          We define 'anomalous' as reconstruction error exceeding what
          95% of normal transactions produce. This gives a principled,
          data-driven threshold — not an arbitrary value.
        """
        raw_errors = self._get_raw_scores(X_licit)
        self._threshold_raw = float(np.percentile(raw_errors, percentile))

        # Also compute normalized threshold for predict()
        all_errors = self._get_raw_scores(X_licit)
        r_min, r_max = all_errors.min(), all_errors.max()
        if r_max > r_min:
            self._threshold = float((self._threshold_raw - r_min) / (r_max - r_min))
        else:
            self._threshold = 0.95

        logger.info(
            "threshold_set_from_percentile",
            percentile=percentile,
            threshold_raw=self._threshold_raw,
            threshold_normalized=self._threshold,
        )
        return self._threshold

    def predict(
        self,
        X: np.ndarray,
        threshold: float | None = None,
    ) -> np.ndarray:
        """Binary predictions: 1=anomalous/illicit, 0=normal/licit."""
        t = threshold if threshold is not None else self._threshold
        scores = self.score_samples(X)
        return (scores >= t).astype(np.int64)

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        test_mask: np.ndarray,
    ) -> dict:
        """Evaluate on labeled test samples."""
        from sklearn.metrics import (
            average_precision_score,
            f1_score,
            roc_auc_score,
        )

        X_known = X_test[test_mask]
        y_known = y_test[test_mask]

        scores = self.score_samples(X_known)
        predictions = self.predict(X_known)

        f1 = f1_score(y_known, predictions, zero_division=0)
        pr_auc = average_precision_score(y_known, scores)
        try:
            roc_auc = roc_auc_score(y_known, scores)
        except ValueError:
            roc_auc = 0.0

        metrics = {
            "f1_score": round(float(f1), 4),
            "pr_auc": round(float(pr_auc), 4),
            "roc_auc": round(float(roc_auc), 4),
            "threshold": round(self._threshold, 4),
            "test_samples": len(X_known),
            "illicit_detected": int(predictions[y_known == 1].sum()),
            "illicit_total": int((y_known == 1).sum()),
        }

        logger.info("autoencoder_evaluation", **metrics)
        return metrics

    def save(self, path: str | None = None) -> str:
        """Save model weights and metadata."""
        save_path = path or str(MODELS_DIR / "autoencoder.pt")
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "threshold": self._threshold,
                "threshold_raw": self._threshold_raw,
                "input_dim": self.input_dim,
                "latent_dim": self.latent_dim,
                "training_history": self.training_history,
            },
            save_path,
        )
        logger.info("autoencoder_saved", path=save_path)
        return save_path

    @classmethod
    def load(cls, path: str | None = None) -> "ThreatAutoencoder":
        """Load model from disk."""
        load_path = path or str(MODELS_DIR / "autoencoder.pt")
        checkpoint = torch.load(load_path, map_location="cpu")
        ae = cls(
            input_dim=checkpoint["input_dim"],
            latent_dim=checkpoint["latent_dim"],
        )
        ae.model.load_state_dict(checkpoint["model_state_dict"])
        ae._threshold = checkpoint["threshold"]
        ae._threshold_raw = checkpoint["threshold_raw"]
        ae._is_fitted = True
        ae.training_history = checkpoint.get("training_history", [])
        logger.info("autoencoder_loaded", path=load_path)
        return ae
