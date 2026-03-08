"""
Singapore Smart City — Congestion Prediction Models

Two-stage prediction architecture:
  Stage 1 (Baseline): Per-camera LSTM for vehicle count forecasting
  Stage 2 (Advanced): Spatial-Temporal Graph (GAT + Transformer) for
                      multi-camera, multi-modal congestion prediction

Both models are designed to train on Colab/Kaggle T4 GPU.
"""

import json
import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


# =============================================================================
# Dataset Classes
# =============================================================================

class TrafficTimeSeriesDataset(Dataset):
    """Dataset for per-camera LSTM prediction.

    Each sample is a windowed time series:
        Input: [vehicle_count, temperature, pm25, taxi_count, hour_sin, hour_cos]
              for the past `window_size` timesteps
        Target: vehicle_count at `horizon` timesteps ahead
    """

    def __init__(
        self,
        data: np.ndarray,
        window_size: int = 30,
        horizon: int = 15,
    ):
        """
        Args:
            data: Array of shape (T, num_features) — chronological.
            window_size: Number of past timesteps as input.
            horizon: How many steps ahead to predict.
        """
        self.data = torch.FloatTensor(data)
        self.window_size = window_size
        self.horizon = horizon

    def __len__(self):
        return len(self.data) - self.window_size - self.horizon + 1

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.window_size]              # (window, features)
        y = self.data[idx + self.window_size + self.horizon - 1, 0]  # vehicle count
        return x, y


class SpatialTemporalDataset(Dataset):
    """Dataset for the graph-based spatial-temporal model.

    Each sample contains:
        - Node features for all cameras over a time window
        - Target: congestion scores for all cameras at horizon
    """

    def __init__(
        self,
        node_features: np.ndarray,
        adjacency: np.ndarray,
        window_size: int = 30,
        horizon: int = 15,
    ):
        """
        Args:
            node_features: Shape (T, num_nodes, num_features).
            adjacency: Shape (num_nodes, num_nodes) — road connectivity.
            window_size: Past timesteps.
            horizon: Future prediction steps.
        """
        self.features = torch.FloatTensor(node_features)
        self.adj = torch.FloatTensor(adjacency)
        self.window_size = window_size
        self.horizon = horizon

    def __len__(self):
        return len(self.features) - self.window_size - self.horizon + 1

    def __getitem__(self, idx):
        x = self.features[idx: idx + self.window_size]                    # (window, nodes, features)
        y = self.features[idx + self.window_size + self.horizon - 1, :, 0]  # (nodes,) — vehicle counts
        return x, y, self.adj


# =============================================================================
# Stage 1: LSTM Baseline — Per-Camera Prediction
# =============================================================================

class TrafficLSTM(nn.Module):
    """LSTM-based time series predictor for single-camera traffic.

    Architecture:
        Input → LSTM (2 layers, bidirectional) → Attention → FC → Output

    Input features: [vehicle_count, temperature, pm25, taxi_count, hour_sin, hour_cos]
    Output: predicted vehicle_count at horizon
    """

    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )

        # Temporal attention — learn which timesteps matter most
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        # Output head
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        """
        Args:
            x: (batch, window_size, input_dim)

        Returns:
            predictions: (batch,)
        """
        # LSTM encoding
        lstm_out, _ = self.lstm(x)  # (batch, window, hidden*2)

        # Attention weights
        attn_weights = self.attention(lstm_out)  # (batch, window, 1)
        attn_weights = F.softmax(attn_weights, dim=1)

        # Weighted sum
        context = torch.sum(lstm_out * attn_weights, dim=1)  # (batch, hidden*2)

        # Predict
        out = self.fc(context).squeeze(-1)  # (batch,)
        return out


# =============================================================================
# Stage 2: Spatial-Temporal Graph Model
# =============================================================================

class GraphAttentionLayer(nn.Module):
    """Graph Attention Network (GAT) layer for spatial message passing.

    Each camera attends to its neighbors on the road network to
    capture spatial dependencies (upstream/downstream traffic flow).
    """

    def __init__(self, in_features: int, out_features: int, num_heads: int = 4):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = out_features // num_heads

        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Parameter(torch.zeros(num_heads, 2 * self.head_dim))
        nn.init.xavier_uniform_(self.a.unsqueeze(0))

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.norm = nn.LayerNorm(out_features)

    def forward(self, x, adj):
        """
        Args:
            x: Node features (batch, num_nodes, in_features)
            adj: Adjacency matrix (num_nodes, num_nodes)

        Returns:
            Updated node features (batch, num_nodes, out_features)
        """
        B, N, _ = x.shape
        h = self.W(x)  # (B, N, out_features)

        # Reshape for multi-head attention
        h = h.view(B, N, self.num_heads, self.head_dim)  # (B, N, heads, head_dim)

        # Compute attention scores
        # For each pair (i, j), compute a^T [h_i || h_j]
        h_i = h.unsqueeze(2).expand(B, N, N, self.num_heads, self.head_dim)
        h_j = h.unsqueeze(1).expand(B, N, N, self.num_heads, self.head_dim)
        concat = torch.cat([h_i, h_j], dim=-1)  # (B, N, N, heads, 2*head_dim)

        e = (concat * self.a).sum(dim=-1)  # (B, N, N, heads)
        e = self.leaky_relu(e)

        # Mask with adjacency matrix (only attend to connected nodes)
        mask = (adj == 0).unsqueeze(0).unsqueeze(-1)  # (1, N, N, 1)
        e = e.masked_fill(mask, float("-inf"))

        attention = F.softmax(e, dim=2)  # (B, N, N, heads)
        attention = torch.nan_to_num(attention, 0.0)

        # Aggregate neighbor features
        h_prime = torch.einsum("bnjh,bnjhd->bnhd", attention, h_j.transpose(2, 3).contiguous().view(B, N, N, self.num_heads, self.head_dim))
        h_prime = h_prime.reshape(B, N, -1)  # (B, N, out_features)

        return self.norm(h_prime + self.W(x))  # Residual connection


class TemporalTransformerLayer(nn.Module):
    """Transformer layer for temporal attention across timesteps.

    Captures temporal patterns like rush hour patterns, event propagation.
    """

    def __init__(self, d_model: int, nhead: int = 4, dropout: float = 0.1):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)

        Returns:
            (batch, seq_len, d_model)
        """
        # Self-attention
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))

        # Feed-forward
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))

        return x


class SpatialTemporalPredictor(nn.Module):
    """Spatial-Temporal Graph Transformer for multi-camera congestion prediction.

    Architecture:
        1. Input projection
        2. Spatial GAT layers (message passing between cameras)
        3. Temporal Transformer layers (attention across time)
        4. Prediction head (per-node output)

    Models the 90 cameras as a graph:
        - Nodes = cameras with multi-modal features
        - Edges = road connections between cameras
        - Temporal = sequential frames over time window
    """

    def __init__(
        self,
        num_features: int = 6,
        hidden_dim: int = 64,
        num_gat_layers: int = 2,
        num_transformer_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Input projection
        self.input_proj = nn.Linear(num_features, hidden_dim)

        # Spatial GAT layers
        self.gat_layers = nn.ModuleList([
            GraphAttentionLayer(hidden_dim, hidden_dim, num_heads)
            for _ in range(num_gat_layers)
        ])

        # Temporal Transformer layers
        self.temporal_layers = nn.ModuleList([
            TemporalTransformerLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_transformer_layers)
        ])

        # Prediction head
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, adj):
        """
        Args:
            x: Node features (batch, window_size, num_nodes, num_features)
            adj: Adjacency matrix (num_nodes, num_nodes)

        Returns:
            Predictions (batch, num_nodes) — predicted vehicle count per camera
        """
        B, T, N, F = x.shape

        # Project input features
        x = self.input_proj(x)  # (B, T, N, hidden)

        # Apply spatial GAT at each timestep
        spatial_out = []
        for t in range(T):
            h = x[:, t, :, :]  # (B, N, hidden)
            for gat in self.gat_layers:
                h = F.relu(gat(h, adj))
            spatial_out.append(h)
        x = torch.stack(spatial_out, dim=1)  # (B, T, N, hidden)

        # Apply temporal Transformer for each node
        temporal_out = []
        for n in range(N):
            h = x[:, :, n, :]  # (B, T, hidden)
            for transformer in self.temporal_layers:
                h = transformer(h)
            temporal_out.append(h[:, -1, :])  # Take last timestep: (B, hidden)
        x = torch.stack(temporal_out, dim=1)  # (B, N, hidden)

        # Predict
        out = self.predictor(x).squeeze(-1)  # (B, N)
        return out


# =============================================================================
# Training Utilities
# =============================================================================

class PredictionTrainer:
    """Handles training and evaluation of prediction models."""

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-3,
        device: str = "auto",
    ):
        self.device = self._resolve_device(device)
        self.model = model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=1e-4
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", patience=5, factor=0.5
        )
        self.criterion = nn.MSELoss()
        self.best_val_loss = float("inf")

    def _resolve_device(self, device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch. Returns average loss."""
        self.model.train()
        total_loss = 0
        batches = 0

        for batch in dataloader:
            if len(batch) == 2:
                # LSTM: (x, y)
                x, y = batch
                x, y = x.to(self.device), y.to(self.device)
                pred = self.model(x)
            else:
                # Graph: (x, y, adj)
                x, y, adj = batch
                x, y, adj = x.to(self.device), y.to(self.device), adj[0].to(self.device)
                pred = self.model(x, adj)

            loss = self.criterion(pred, y)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            batches += 1

        return total_loss / max(batches, 1)

    def evaluate(self, dataloader: DataLoader) -> dict:
        """Evaluate model. Returns loss and metrics."""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        batches = 0

        with torch.no_grad():
            for batch in dataloader:
                if len(batch) == 2:
                    x, y = batch
                    x, y = x.to(self.device), y.to(self.device)
                    pred = self.model(x)
                else:
                    x, y, adj = batch
                    x, y, adj = x.to(self.device), y.to(self.device), adj[0].to(self.device)
                    pred = self.model(x, adj)

                loss = self.criterion(pred, y)
                total_loss += loss.item()
                batches += 1

                all_preds.append(pred.cpu().numpy())
                all_targets.append(y.cpu().numpy())

        preds = np.concatenate(all_preds)
        targets = np.concatenate(all_targets)

        mse = float(np.mean((preds - targets) ** 2))
        mae = float(np.mean(np.abs(preds - targets)))
        rmse = float(np.sqrt(mse))

        # MAPE (avoid division by zero)
        mask = targets > 0
        mape = float(np.mean(np.abs((preds[mask] - targets[mask]) / targets[mask]))) * 100 if mask.any() else 0.0

        return {
            "loss": total_loss / max(batches, 1),
            "mse": round(mse, 4),
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "mape": round(mape, 2),
        }

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        patience: int = 15,
        save_path: Optional[str] = None,
    ) -> dict:
        """Full training loop with early stopping.

        Returns:
            Training history dict with losses and metrics.
        """
        history = {"train_loss": [], "val_loss": [], "val_metrics": []}
        no_improve = 0

        logger.info(
            f"Training on {self.device} | "
            f"epochs={epochs}, patience={patience}"
        )

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            val_loss = val_metrics["loss"]

            self.scheduler.step(val_loss)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_metrics"].append(val_metrics)

            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                no_improve = 0
                if save_path:
                    torch.save(self.model.state_dict(), save_path)
                    logger.debug(f"Model saved to {save_path}")
            else:
                no_improve += 1

            if (epoch + 1) % 5 == 0 or epoch == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                logger.info(
                    f"Epoch {epoch+1}/{epochs} | "
                    f"Train: {train_loss:.4f} | "
                    f"Val: {val_loss:.4f} | "
                    f"RMSE: {val_metrics['rmse']:.3f} | "
                    f"MAE: {val_metrics['mae']:.3f} | "
                    f"LR: {lr:.2e}"
                )

            if no_improve >= patience:
                logger.info(
                    f"Early stopping at epoch {epoch+1} "
                    f"(no improvement for {patience} epochs)"
                )
                break

        logger.info(
            f"Training complete | Best val loss: {self.best_val_loss:.4f}"
        )
        return history


# =============================================================================
# Feature Engineering Utilities
# =============================================================================

def prepare_features(
    metadata_records: list[dict],
    feature_columns: list[str] = None,
) -> np.ndarray:
    """Convert metadata records to a feature matrix for model input.

    Returns array with columns:
        [vehicle_count, temperature, pm25, taxi_count, hour_sin, hour_cos]
    """
    if feature_columns is None:
        feature_columns = [
            "num_vehicles", "temperature_celsius", "pm25_reading",
            "taxi_count_nearby_5km",
        ]

    features = []
    for record in metadata_records:
        row = []

        # Base features
        for col in feature_columns:
            val = record.get(col, 0)
            row.append(float(val) if val is not None else 0.0)

        # Cyclical time encoding
        timestamp = record.get("timestamp", "")
        try:
            hour = int(timestamp.split("T")[1][:2])
        except (IndexError, ValueError):
            hour = 12  # Default to noon

        row.append(math.sin(2 * math.pi * hour / 24))  # hour_sin
        row.append(math.cos(2 * math.pi * hour / 24))  # hour_cos

        features.append(row)

    arr = np.array(features, dtype=np.float32)

    # Normalize (z-score per feature)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    std[std == 0] = 1  # Avoid division by zero
    arr = (arr - mean) / std

    return arr


def build_adjacency_matrix(
    camera_locations: dict[str, tuple[float, float]],
    distance_threshold_km: float = 3.0,
) -> np.ndarray:
    """Build adjacency matrix from camera GPS coordinates.

    Two cameras are connected if they are within distance_threshold_km.
    This approximates road network connectivity.

    Args:
        camera_locations: Dict of camera_id → (latitude, longitude).
        distance_threshold_km: Max distance for connection.

    Returns:
        Adjacency matrix of shape (num_cameras, num_cameras).
    """
    camera_ids = sorted(camera_locations.keys())
    n = len(camera_ids)
    adj = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        for j in range(n):
            if i == j:
                adj[i][j] = 1.0  # Self-loop
                continue

            lat1, lng1 = camera_locations[camera_ids[i]]
            lat2, lng2 = camera_locations[camera_ids[j]]

            # Simple distance approximation (sufficient for Singapore)
            dist_km = np.sqrt(
                ((lat1 - lat2) * 111) ** 2 +
                ((lng1 - lng2) * 111 * np.cos(np.radians(1.35))) ** 2
            )

            if dist_km <= distance_threshold_km:
                adj[i][j] = 1.0

    logger.info(
        f"Adjacency matrix: {n} cameras, "
        f"{int(adj.sum() - n)} connections "
        f"(threshold: {distance_threshold_km}km)"
    )

    return adj
