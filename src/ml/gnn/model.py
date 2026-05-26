"""
ThreatGNN — Graph Neural Network for blockchain threat detection.

Architecture: GraphSAGE + GAT hybrid with skip connections

WHY GraphSAGE (layer 1 and 3):
  Inductive learning — works on nodes not seen during training.
  New wallets appear on blockchain constantly.
  GCN is transductive (fixed graph only) — useless for production.
  GraphSAGE learns an aggregation FUNCTION applicable to any node.

WHY GAT (layer 2):
  Attention weights per edge — learned importance of each neighbor.
  A wallet receiving from 10 suspicious addresses gets high attention
  toward those addresses — their embeddings dominate the update.
  Attention weights = explainability signal (which neighbors matter).

WHY skip connections:
  Prevent oversmoothing — deep GNNs make all nodes look identical.
  Skip connection from layer 1 to layer 3 preserves local features
  while layer 2 captures neighborhood patterns.

WHY NeighborLoader:
  Full graph (200K nodes) doesn't fit in GPU memory for message passing.
  NeighborLoader samples k neighbors per node per layer.
  [25, 10, 5] means: sample 25 neighbors for layer 1,
  10 of those neighbors' neighbors for layer 2, etc.
  Makes training scalable to graphs of any size.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv, BatchNorm

from config.logging_config import get_logger

logger = get_logger(__name__)


class ThreatGNN(nn.Module):
    """
    3-layer GNN: SAGEConv -> GATConv -> SAGEConv
    with skip connection and dropout regularization.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        out_channels: int = 2,
        heads: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.dropout = dropout
        self.hidden_channels = hidden_channels

        # Layer 1: SAGEConv — inductive, captures local features
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.bn1 = BatchNorm(hidden_channels)

        # Layer 2: GATConv — attention-based neighborhood aggregation
        # concat=False averages the heads → output dim = hidden_channels
        self.conv2 = GATConv(
            hidden_channels,
            hidden_channels,
            heads=heads,
            dropout=dropout,
            concat=False,
        )
        self.bn2 = BatchNorm(hidden_channels)

        # Layer 3: SAGEConv — output embeddings
        self.conv3 = SAGEConv(hidden_channels, hidden_channels // 2)

        # Classification head
        self.classifier = nn.Linear(hidden_channels // 2, out_channels)

        # Skip connection projection (layer 1 → layer 3 residual)
        self.skip = nn.Linear(hidden_channels, hidden_channels // 2)

        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for stable training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.
        x: node features [num_nodes, in_channels]
        edge_index: graph connectivity [2, num_edges]
        Returns: logits [num_nodes, out_channels]
        """
        # Layer 1: SAGEConv
        h1 = self.conv1(x, edge_index)
        h1 = self.bn1(h1)
        h1 = F.relu(h1)
        h1 = F.dropout(h1, p=self.dropout, training=self.training)

        # Layer 2: GATConv
        h2 = self.conv2(h1, edge_index)
        h2 = self.bn2(h2)
        h2 = F.relu(h2)
        h2 = F.dropout(h2, p=self.dropout, training=self.training)

        # Skip connection: h1 projected to same dim as h3
        h_skip = self.skip(h1)

        # Layer 3: SAGEConv + skip
        h3 = self.conv3(h2, edge_index)
        h3 = F.relu(h3 + h_skip)

        # Classification
        out = self.classifier(h3)
        return out

    def get_embeddings(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get node embeddings before classification head.
        Used for visualization and clustering.
        """
        h1 = F.relu(self.bn1(self.conv1(x, edge_index)))
        h2 = F.relu(self.bn2(self.conv2(h1, edge_index)))
        h_skip = self.skip(h1)
        h3 = F.relu(self.conv3(h2, edge_index) + h_skip)
        return h3
