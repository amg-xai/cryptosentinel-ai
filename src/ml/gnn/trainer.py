"""
GNN training pipeline with NeighborLoader mini-batch training.

WHY mini-batch training:
  The full Elliptic graph (200K nodes, 234K edges) won't fit in GPU memory
  for full-batch message passing. NeighborLoader samples subgraphs.

WHY class weights:
  Illicit: ~3,462 samples, Licit: ~26,432 samples → ratio 1:7.6
  Without weighting: model predicts all-licit and gets 88% accuracy.
  Class weight for illicit = 7.6 so each illicit sample counts 7.6x more.

WHY AMP (Automatic Mixed Precision):
  RTX 3050 has Tensor Cores for FP16 computation.
  AMP uses FP16 for forward/backward (2x faster) and FP32 for weights.
  Gives ~50% speedup with identical numerical results.
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score

from config.logging_config import get_logger
from src.ml.gnn.model import ThreatGNN

logger = get_logger(__name__)

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("data/models")


def build_pyg_data(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    edge_index: np.ndarray,
) -> Data:
    """
    Build a single PyTorch Geometric Data object containing the full graph.
    Masks indicate which nodes are used for training vs testing.
    """
    # Combine train and test into single feature matrix
    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])

    # Build combined masks
    n_train = len(X_train)
    n_total = len(X_all)

    train_mask_full = np.zeros(n_total, dtype=bool)
    train_mask_full[:n_train] = train_mask

    test_mask_full = np.zeros(n_total, dtype=bool)
    test_mask_full[n_train:] = test_mask

    data = Data(
        x=torch.tensor(X_all, dtype=torch.float32),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        y=torch.tensor(y_all, dtype=torch.long),
        train_mask=torch.tensor(train_mask_full, dtype=torch.bool),
        test_mask=torch.tensor(test_mask_full, dtype=torch.bool),
    )

    logger.info(
        "pyg_data_built",
        num_nodes=data.num_nodes,
        num_edges=data.num_edges,
        num_features=data.num_node_features,
        train_nodes=int(train_mask_full.sum()),
        test_nodes=int(test_mask_full.sum()),
    )

    return data


class GNNTrainer:
    """Handles training, evaluation, and saving of ThreatGNN."""

    def __init__(
        self,
        hidden_channels: int = 128,
        heads: int = 4,
        dropout: float = 0.3,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
    ):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        logger.info("gnn_trainer_init", device=str(self.device))

        self.hidden_channels = hidden_channels
        self.heads = heads
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.model: ThreatGNN | None = None
        self.history: list[dict] = []

    def _compute_class_weights(
        self,
        y: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute inverse frequency class weights.
        Ensures illicit class gets proportionally more gradient signal.
        """
        y_known = y[mask]
        n_licit = (y_known == 0).sum().item()
        n_illicit = (y_known == 1).sum().item()

        if n_illicit == 0:
            return torch.ones(2).to(self.device)

        weight_licit = 1.0
        weight_illicit = n_licit / n_illicit

        logger.info(
            "class_weights_computed",
            n_licit=n_licit,
            n_illicit=n_illicit,
            weight_illicit=round(weight_illicit, 2),
        )

        return torch.tensor(
            [weight_licit, weight_illicit],
            dtype=torch.float32,
        ).to(self.device)

    def train(
        self,
        data: Data,
        epochs: int = 100,
        batch_size: int = 512,
        num_neighbors: list[int] | None = None,
    ) -> list[dict]:
        """
        Train ThreatGNN with NeighborLoader mini-batching and AMP.
        """
        if num_neighbors is None:
            num_neighbors = [25, 10, 5]

        # Initialize model
        self.model = ThreatGNN(
            in_channels=data.num_node_features,
            hidden_channels=self.hidden_channels,
            heads=self.heads,
            dropout=self.dropout,
        ).to(self.device)

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=10, factor=0.5
        )

        # Class weights for imbalanced data
        class_weights = self._compute_class_weights(
            data.y, data.train_mask
        )

        # NeighborLoader for mini-batch training
        train_loader = NeighborLoader(
            data,
            num_neighbors=num_neighbors,
            batch_size=batch_size,
            input_nodes=data.train_mask,
            shuffle=True,
            num_workers=0,
        )

        # AMP scaler for mixed precision
        use_amp = self.device.type == "cuda"
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        logger.info(
            "gnn_training_started",
            epochs=epochs,
            batch_size=batch_size,
            num_neighbors=num_neighbors,
            hidden_channels=self.hidden_channels,
            use_amp=use_amp,
            device=str(self.device),
        )

        best_loss = float("inf")
        self.history = []

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0

            for batch in train_loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()

                with torch.cuda.amp.autocast(enabled=use_amp):
                    out = self.model(batch.x, batch.edge_index)
                    # Only compute loss on labeled nodes in this batch
                    mask = batch.train_mask
                    if mask.sum() == 0:
                        continue
                    loss = F.cross_entropy(
                        out[mask],
                        batch.y[mask],
                        weight=class_weights,
                    )

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=1.0
                )
                scaler.step(optimizer)
                scaler.update()

                epoch_loss += loss.item()
                n_batches += 1

            if n_batches == 0:
                continue

            avg_loss = epoch_loss / n_batches
            scheduler.step(avg_loss)

            record = {
                "epoch": epoch + 1,
                "loss": round(avg_loss, 6),
                "lr": round(optimizer.param_groups[0]["lr"], 6),
            }
            self.history.append(record)

            if avg_loss < best_loss:
                best_loss = avg_loss
                self.save(str(MODELS_DIR / "gnn_best.pt"))

            if (epoch + 1) % 10 == 0:
                logger.info("gnn_epoch", **record)

        logger.info(
            "gnn_training_complete",
            best_loss=round(best_loss, 6),
            final_loss=self.history[-1]["loss"],
        )

        return self.history

    @torch.no_grad()
    def evaluate(self, data: Data) -> dict:
        """Evaluate on test nodes."""
        if self.model is None:
            raise RuntimeError("Model not trained.")

        self.model.eval()
        data = data.to(self.device)

        out = self.model(data.x, data.edge_index)
        probs = torch.softmax(out, dim=1)[:, 1]  # illicit probability

        mask = data.test_mask
        y_true = data.y[mask].cpu().numpy()
        y_scores = probs[mask].cpu().numpy()
        y_pred = (y_scores >= 0.5).astype(int)

        f1 = f1_score(y_true, y_pred, zero_division=0)
        pr_auc = average_precision_score(y_true, y_scores)
        try:
            roc_auc = roc_auc_score(y_true, y_scores)
        except ValueError:
            roc_auc = 0.0

        metrics = {
            "f1_score": round(float(f1), 4),
            "pr_auc": round(float(pr_auc), 4),
            "roc_auc": round(float(roc_auc), 4),
            "test_samples": int(mask.sum()),
            "illicit_detected": int(y_pred[y_true == 1].sum()),
            "illicit_total": int((y_true == 1).sum()),
        }

        logger.info("gnn_evaluation", **metrics)
        return metrics

    def save(self, path: str) -> None:
        if self.model is None:
            return
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "hidden_channels": self.hidden_channels,
            "heads": self.heads,
            "dropout": self.dropout,
            "history": self.history,
        }, path)

    def load(self, path: str, in_channels: int) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.model = ThreatGNN(
            in_channels=in_channels,
            hidden_channels=checkpoint["hidden_channels"],
            heads=checkpoint["heads"],
            dropout=checkpoint["dropout"],
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.history = checkpoint.get("history", [])
        logger.info("gnn_loaded", path=path)
