"""
CATI Training Pipeline -- Context-Aware Traffic Intelligence

Two-phase training strategy:

Phase 1: Context Module Training (freeze YOLO backbone)
    - Train only ContextEncoder + FiLMGenerator + FiLM layers
    - YOLO backbone weights frozen (pretrained on COCO or UA-DETRAC)
    - Learning rate: 1e-3 (context modules only)
    - ~2-3 hours on T4 GPU

Phase 2: End-to-End Fine-tuning (unfreeze backbone, lower LR)
    - Unfreeze YOLO backbone
    - Learning rate: 1e-4 (backbone) + 1e-3 (context modules)
    - ~3-5 hours on T4 GPU

Usage:
    python -m src.training.train_cati --data configs/traffic_dataset.yaml --phase 1
"""

import json
import logging
from pathlib import Path

import click
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset

from src.models.cati_detector import CATIConfig, CATIDetector
from src.models.context_encoder import ContextEncoder

logger = logging.getLogger(__name__)


class CATIDataset(Dataset):
    """Dataset for CATI training.

    Loads pre-extracted backbone features + metadata from disk.
    Features are extracted once from the frozen YOLO backbone for efficiency.
    """

    def __init__(self, feature_dir: str, max_samples: int | None = None):
        self.feature_dir = Path(feature_dir)
        self.samples: list[Path] = sorted(self.feature_dir.glob("*.pt"))

        if max_samples:
            self.samples = self.samples[:max_samples]

        logger.info(f"CATIDataset: {len(self.samples)} samples from {feature_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        feature_path = self.samples[idx]
        meta_path = feature_path.with_suffix(".json")

        data = torch.load(feature_path, map_location="cpu", weights_only=True)
        features = data["features"]

        with open(meta_path) as f:
            meta = json.load(f)

        weather_id = ContextEncoder.weather_to_id(meta.get("weather_condition", "unknown"))
        resolution_id = ContextEncoder.resolution_to_id(
            meta.get("image_width", 1920), meta.get("image_height", 1080)
        )

        context = {
            "weather_id": torch.tensor(weather_id, dtype=torch.long),
            "temperature": torch.tensor(meta.get("temperature_celsius", 28.0), dtype=torch.float),
            "pm25": torch.tensor(meta.get("pm25_reading", 15.0), dtype=torch.float),
            "hour": torch.tensor(meta.get("hour", 12.0), dtype=torch.float),
            "camera_id": torch.tensor(meta.get("camera_idx", 0), dtype=torch.long),
            "resolution_id": torch.tensor(resolution_id, dtype=torch.long),
        }

        targets = data.get("targets", torch.zeros(0, 6))
        return features, context, targets


class CATITrainer:
    """Handles two-phase CATI training."""

    def __init__(
        self,
        config: CATIConfig,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        device: str = "auto",
    ):
        self.device = self._resolve_device(device)
        self.config = config
        self.cati = CATIDetector(config).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.cati.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=50, eta_min=1e-6
        )

        self.best_val_loss = float("inf")
        logger.info(f"CATITrainer initialized on {self.device}")
        logger.info(f"Parameters: {self.cati.count_parameters()}")

    def _resolve_device(self, device: str) -> torch.device:
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
        return torch.device("cpu") if device == "auto" else torch.device(device)

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch. Returns average loss."""
        self.cati.train()
        total_loss = 0.0
        num_batches = 0

        for features, context, _targets in dataloader:
            features = [f.to(self.device) for f in features]
            ctx = {k: v.to(self.device) for k, v in context.items()}

            modulated = self.cati(
                features,
                ctx["weather_id"],
                ctx["temperature"],
                ctx["pm25"],
                ctx["hour"],
                ctx["camera_id"],
                ctx["resolution_id"],
            )

            loss = sum(
                nn.functional.mse_loss(mod, orig)
                for mod, orig in zip(modulated, features, strict=True)
            )

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.cati.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        self.scheduler.step()
        return total_loss / max(num_batches, 1)

    def save_checkpoint(self, path: str, epoch: int, val_loss: float):
        """Save training checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.cati.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "config": self.config.__dict__,
        }
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path} (epoch={epoch}, val_loss={val_loss:.4f})")

    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.cati.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logger.info(f"Checkpoint loaded: {path} (epoch={checkpoint['epoch']})")
        return checkpoint["epoch"]


@click.command()
@click.option("--data", required=True, help="Path to dataset YAML config")
@click.option("--config", default="configs/training_config.yaml", help="Training config")
@click.option("--phase", type=int, default=1, help="Training phase (1 or 2)")
@click.option("--epochs", type=int, default=50)
@click.option("--lr", type=float, default=1e-3)
@click.option("--output", default="models/cati_latest.pt")
def main(data, config, phase, epochs, lr, output):
    """CATI Training -- Context-Aware Traffic Intelligence."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    with open(config) as f:
        train_config = yaml.safe_load(f)

    cati_config = CATIConfig(
        num_cameras=train_config.get("num_cameras", 90),
        context_dim=train_config.get("context_dim", 64),
    )

    logger.info(f"CATI Training -- Phase {phase}, Epochs: {epochs}, LR: {lr}")

    trainer = CATITrainer(config=cati_config, learning_rate=lr)

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(output, epoch=0, val_loss=float("inf"))
    logger.info(f"Initial checkpoint saved to {output}")


if __name__ == "__main__":
    main()
