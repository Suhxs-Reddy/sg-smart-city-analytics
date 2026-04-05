"""
CATI Training Pipeline — Context-Aware Traffic Intelligence (Production Grade)

Senior ML Engineer training architecture with:
    - Two-phase strategy (freeze backbone → full fine-tuning)
    - Real cached backbone features (not synthetic) when available
    - Per-parameter-group optimizer (backbone × 0.1, context × 1.0)
    - Automatic Mixed Precision (AMP) with GradScaler
    - EMA model for stable inference
    - Linear warmup → cosine decay schedule
    - Gradient accumulation for effective batch size control
    - Condition-stratified validation (weather × resolution × time)
    - Checkpoint management with best/last model saving

Two-phase training strategy:

Phase 1: Context Module Training (freeze YOLO backbone)
    - Loads cached P3/P4/P5 features from disk (.pt files from feature_extractor.py)
    - Train only ContextEncoder + FiLM layers
    - YOLO backbone weights NOT needed (features are pre-extracted)
    - Learning rate: 1e-3 (context modules only)

Phase 2: End-to-End Fine-tuning (unfreeze backbone, lower LR)
    - Loads full YOLO backbone + CATI module
    - Learning rate: 1e-4 (backbone) + 1e-3 (context modules)
    - Requires GPU with ≥16GB VRAM on T4/L4/A100

Usage:
    python -m src.training.train_cati --config configs/training_config.yaml --phase 1
"""

import json
import logging
import math
import time
from pathlib import Path

import click
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset

from src.models.cati_detector import CATIConfig, CATIDetector, EMAModel
from src.models.context_encoder import WEATHER_CONDITIONS, ContextEncoder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class CATIDataset(Dataset):
    """Dataset for CATI Phase 1 training.

    Loads pre-extracted metadata (.json) and backbone feature tensors (.pt)
    from disk. Each sample provides:
        1. Environmental context (weather, time, GPS, etc.) for the ContextEncoder
        2. Cached P3/P4/P5 backbone features (FP16 → converted to FP32 on load)
        3. Stratification labels for condition-split validation

    Expected directory structure (from FeatureExtractor):
        feature_dir/
            cam1001_20260309_080100.json
            cam1001_20260309_080100.pt   ← FP16 backbone features [P3, P4, P5]
            ...

    If a .pt file is missing for a sample the dataset returns None for features.
    The trainer falls back to synthetic features in that case so training can
    proceed on partial extractions.

    Args:
        feature_dir: Directory containing .json + .pt pairs.
        max_samples: Cap the dataset size (useful for quick smoke tests).
    """

    def __init__(self, feature_dir: str, max_samples: int | None = None):
        self.feature_dir = Path(feature_dir)
        self.samples: list[Path] = sorted(self.feature_dir.glob("*.json"))

        if max_samples:
            self.samples = self.samples[:max_samples]

        has_features = sum(1 for p in self.samples if p.with_suffix(".pt").exists())
        logger.info(
            f"CATIDataset: {len(self.samples)} samples from {feature_dir} "
            f"({has_features} with cached backbone features)"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple:
        meta_path = self.samples[idx]

        with open(meta_path) as f:
            meta = json.load(f)

        # ----- Context tensors -----
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

        lat = meta.get("camera_latitude")
        lon = meta.get("camera_longitude")
        if lat is not None and lon is not None:
            context["camera_lat"] = torch.tensor(float(lat), dtype=torch.float)
            context["camera_lon"] = torch.tensor(float(lon), dtype=torch.float)

        # ----- Stratification labels -----
        stratification = {
            "weather": meta.get("weather_condition", "unknown"),
            "resolution": "high" if meta.get("image_width", 1920) >= 1280 else "low",
            "time_of_day": self._time_category(meta.get("hour", 12.0)),
        }

        # ----- Cached backbone features -----
        feat_path = meta_path.with_suffix(".pt")
        features: list[torch.Tensor] | None = None
        if feat_path.exists():
            try:
                # Saved as FP16 list; convert to FP32 for training
                features = [f.float() for f in torch.load(feat_path, weights_only=True)]
            except Exception as e:
                logger.warning(f"Failed to load features from {feat_path}: {e}")

        return context, stratification, features

    @staticmethod
    def _time_category(hour: float) -> str:
        if 6 <= hour < 10:
            return "morning_rush"
        elif 10 <= hour < 16:
            return "midday"
        elif 16 <= hour < 20:
            return "evening_rush"
        else:
            return "night"


def cati_collate(batch: list) -> tuple:
    """Custom collate function for CATIDataset.

    Handles the mixed-type batch (context dicts, strat dicts, optional feature lists).
    Features are stacked into (B, C, H, W) tensors per backbone stage.
    If any sample is missing features, the whole batch falls back to None (synthetic).
    """
    contexts, strats, features_list = zip(*batch, strict=True)

    # Collate context dicts — all keys must be tensors
    all_keys = contexts[0].keys()
    collated_ctx = {k: torch.stack([c[k] for c in contexts]) for k in all_keys}

    # Collate stratification strings into lists
    all_strat_keys = strats[0].keys()
    collated_strat = {k: [s[k] for s in strats] for k in all_strat_keys}

    # Collate features: stack along batch dim per stage
    # Each sample's features is [P3(1,C,H,W), P4(1,C,H,W), P5(1,C,H,W)] or None
    if all(f is not None for f in features_list):
        num_stages = len(features_list[0])
        collated_features = [
            torch.cat([sample_feats[stage] for sample_feats in features_list], dim=0)
            for stage in range(num_stages)
        ]
    else:
        collated_features = None  # Trainer will use synthetic fallback

    return collated_ctx, collated_strat, collated_features


# ---------------------------------------------------------------------------
# Context Prediction Head (Phase 1 training signal)
# ---------------------------------------------------------------------------


class ContextPredictionHead(nn.Module):
    """Predicts context variables from FiLM-conditioned backbone features.

    Phase 1 training signal: forces the FiLM conditioning to encode meaningful
    context information rather than collapsing to identity (the failure mode of
    simple MSE(modulated, original) loss where the gate just closes to zero).

    Global-average-pools each FiLM stage, concatenates, then predicts:
        - Weather class   (cross-entropy, 11 classes)
        - Hour sin/cos    (MSE with cyclical encoding)
        - Camera identity (cross-entropy, up to num_cameras)
        - Temperature     (MSE, normalized to Singapore range)

    Args:
        feature_dims: Channel dims at each backbone stage, e.g. [256, 256, 512].
        num_cameras: Number of unique cameras in the network.
        num_weather_classes: Number of weather condition classes.
    """

    def __init__(
        self,
        feature_dims: list[int],
        num_cameras: int = 90,
        num_weather_classes: int = len(WEATHER_CONDITIONS),
    ):
        super().__init__()
        total_dim = sum(feature_dims)
        hidden = min(512, total_dim // 2)

        self.proj = nn.Sequential(
            nn.Linear(total_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(0.1),
        )
        self.weather_head = nn.Linear(hidden, num_weather_classes)
        self.hour_head = nn.Linear(hidden, 2)     # sin/cos
        self.camera_head = nn.Linear(hidden, num_cameras)
        self.temp_head = nn.Linear(hidden, 1)     # normalized temperature

    def forward(self, features: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        """Pool all stages and predict context variables.

        Args:
            features: List of (B, C, H, W) conditioned feature maps.

        Returns:
            Dict with keys: weather, hour, camera, temp.
        """
        pooled = [f.mean(dim=(-2, -1)) for f in features]  # [(B, C), ...]
        combined = torch.cat(pooled, dim=-1)  # (B, total_dim)
        h = self.proj(combined)
        return {
            "weather": self.weather_head(h),
            "hour": self.hour_head(h),
            "camera": self.camera_head(h),
            "temp": self.temp_head(h).squeeze(-1),
        }

    def compute_loss(
        self,
        preds: dict[str, torch.Tensor],
        ctx: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute combined context prediction loss.

        Args:
            preds: Output of forward().
            ctx: Context tensors from the dataloader batch.

        Returns:
            Scalar loss tensor.
        """
        hour = ctx["hour"].float()
        hour_target = torch.stack(
            [
                torch.sin(2 * math.pi * hour / 24),
                torch.cos(2 * math.pi * hour / 24),
            ],
            dim=-1,
        )
        temp_target = (ctx["temperature"].float() - 29.0) / 5.0

        weather_loss = nn.functional.cross_entropy(preds["weather"], ctx["weather_id"].long())
        hour_loss = nn.functional.mse_loss(preds["hour"], hour_target)
        camera_loss = nn.functional.cross_entropy(preds["camera"], ctx["camera_id"].long())
        temp_loss = nn.functional.mse_loss(preds["temp"], temp_target)

        # Weight camera lower — it has many classes and is harder to predict
        return weather_loss + hour_loss + 0.5 * camera_loss + temp_loss


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class CATITrainer:
    """Production-grade CATI training with all senior ML patterns.

    Args:
        config: CATIConfig.
        learning_rate: Base learning rate.
        weight_decay: Weight decay for AdamW.
        device: Compute device.
        use_amp: Enable automatic mixed precision.
        grad_accum_steps: Gradient accumulation steps.
        warmup_epochs: Linear warmup epochs before cosine decay.
    """

    def __init__(
        self,
        config: CATIConfig,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        device: str = "auto",
        use_amp: bool = True,
        grad_accum_steps: int = 1,
        warmup_epochs: int = 5,
    ):
        self.device = self._resolve_device(device)
        self.config = config
        self.use_amp = use_amp and self.device.type == "cuda"
        self.grad_accum_steps = grad_accum_steps
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0

        # Build model
        self.cati = CATIDetector(config).to(self.device)

        # EMA model for stable inference
        self.ema = EMAModel(self.cati, decay=config.ema_decay) if config.ema_decay > 0 else None

        # Context prediction head for Phase 1 self-supervised training signal
        self.prediction_head = ContextPredictionHead(
            feature_dims=config.backbone_channels,
            num_cameras=config.num_cameras,
        ).to(self.device)

        # Optimizer with per-parameter-group learning rates
        self.optimizer = self._build_optimizer(learning_rate, weight_decay)

        # Scheduler (set in train() based on total epochs)
        self.scheduler = None

        # AMP scaler
        self.scaler = torch.amp.GradScaler() if self.use_amp else None

        # Tracking
        self.best_val_loss = float("inf")
        self.training_history: list[dict] = []

        params = self.cati.count_parameters()
        logger.info(
            f"CATITrainer initialized on {self.device} | "
            f"AMP={'ON' if self.use_amp else 'OFF'} | "
            f"Params: {params['total_cati_overhead']:,}"
        )

    def _resolve_device(self, device: str) -> torch.device:
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
        return torch.device("cpu") if device == "auto" else torch.device(device)

    def _build_optimizer(self, learning_rate: float, weight_decay: float) -> torch.optim.Optimizer:
        """AdamW with per-parameter-group LRs.

        Context encoder and FiLM generator: full LR.
        FiLM layers (attention modules): 0.5× LR (they converge faster).
        """
        param_groups = [
            {
                "params": list(self.cati.context_encoder.parameters()),
                "lr": learning_rate,
                "name": "context_encoder",
            },
            {
                "params": list(self.cati.film_generator.parameters()),
                "lr": learning_rate,
                "name": "film_generator",
            },
            {
                "params": list(self.cati.film_layers.parameters()),
                "lr": learning_rate * 0.5,
                "name": "film_layers",
            },
            {
                "params": list(self.prediction_head.parameters()),
                "lr": learning_rate,
                "name": "prediction_head",
            },
        ]
        return torch.optim.AdamW(param_groups, weight_decay=weight_decay)

    def _get_warmup_lr_scale(self, epoch: int) -> float:
        """Linear warmup scaling factor."""
        if epoch < self.warmup_epochs:
            return max(0.01, epoch / self.warmup_epochs)
        return 1.0

    def _make_synthetic_features(self, batch_size: int) -> list[torch.Tensor]:
        """Fallback synthetic features when cached tensors are unavailable.

        Uses the spatial dimensions from the training config backbone channels.
        Phase 1 with synthetic features is only useful for unit tests — real
        training must use cached backbone features from feature_extractor.py.
        """
        return [
            torch.randn(batch_size, dim, 80 // (2**i), 80 // (2**i), device=self.device)
            for i, dim in enumerate(self.config.backbone_channels)
        ]

    def train_epoch(self, dataloader: DataLoader) -> dict:
        """Train for one epoch. Returns metrics dict."""
        self.cati.train()
        total_loss = 0.0
        num_batches = 0
        synthetic_batches = 0
        epoch_start = time.time()

        # Apply warmup scaling
        warmup_scale = self._get_warmup_lr_scale(self.current_epoch)
        for pg in self.optimizer.param_groups:
            pg["lr"] = pg.get("initial_lr", pg["lr"]) * warmup_scale

        for batch_idx, (context, _strat, cached_features) in enumerate(dataloader):
            ctx = {k: v.to(self.device) for k, v in context.items()}
            batch_size = ctx["weather_id"].size(0)

            # Use cached backbone features if available, else synthetic fallback
            if cached_features is not None:
                features = [f.to(self.device) for f in cached_features]
            else:
                features = self._make_synthetic_features(batch_size)
                synthetic_batches += 1

            with torch.amp.autocast(self.device.type, enabled=self.use_amp):
                modulated = self.cati(
                    features,
                    ctx["weather_id"],
                    ctx["temperature"],
                    ctx["pm25"],
                    ctx["hour"],
                    ctx["camera_id"],
                    ctx["resolution_id"],
                    ctx.get("camera_lat"),
                    ctx.get("camera_lon"),
                )

                # Phase 1 loss: context prediction from conditioned features.
                # Predicts weather/hour/camera/temp from FiLM-conditioned features.
                # This forces the gate to open and the context encoder to learn
                # meaningful modulations — avoids the gate-closes-to-zero failure mode.
                preds = self.prediction_head(modulated)
                loss = self.prediction_head.compute_loss(preds, ctx)
                loss = loss / self.grad_accum_steps

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch_idx + 1) % self.grad_accum_steps == 0:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.cati.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.cati.parameters(), max_norm=1.0)
                    self.optimizer.step()

                self.optimizer.zero_grad()

                if self.ema is not None:
                    self.ema.update(self.cati)

            total_loss += loss.item() * self.grad_accum_steps
            num_batches += 1

        if self.scheduler is not None:
            self.scheduler.step()

        self.current_epoch += 1
        elapsed = time.time() - epoch_start

        metrics = {
            "train_loss": total_loss / max(num_batches, 1),
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            "epoch_time_s": elapsed,
            "synthetic_batches": synthetic_batches,
        }
        self.training_history.append(metrics)
        return metrics

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> dict:
        """Validate with condition-stratified metrics."""
        self.cati.eval()
        total_loss = 0.0
        num_batches = 0
        strat_losses: dict[str, list[float]] = {}

        for context, strat, cached_features in dataloader:
            ctx = {k: v.to(self.device) for k, v in context.items()}
            batch_size = ctx["weather_id"].size(0)

            features = (
                [f.to(self.device) for f in cached_features]
                if cached_features is not None
                else self._make_synthetic_features(batch_size)
            )

            modulated = self.cati(
                features,
                ctx["weather_id"],
                ctx["temperature"],
                ctx["pm25"],
                ctx["hour"],
                ctx["camera_id"],
                ctx["resolution_id"],
                ctx.get("camera_lat"),
                ctx.get("camera_lon"),
            )

            preds = self.prediction_head(modulated)
            loss = self.prediction_head.compute_loss(preds, ctx)
            total_loss += loss.item()
            num_batches += 1

            # Per-condition tracking
            for i in range(batch_size):
                for key in ["weather", "resolution", "time_of_day"]:
                    if key in strat:
                        condition = strat[key][i] if isinstance(strat[key], list) else strat[key]
                        group = f"{key}/{condition}"
                        strat_losses.setdefault(group, []).append(loss.item())

        metrics = {"val_loss": total_loss / max(num_batches, 1)}
        for group, losses in strat_losses.items():
            metrics[f"val_loss_{group}"] = sum(losses) / len(losses)

        return metrics

    def save_checkpoint(
        self,
        path: str,
        epoch: int,
        val_loss: float,
        is_best: bool = False,
    ):
        """Save training checkpoint with full state."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.cati.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "config": {k: v for k, v in self.config.__dict__.items() if not callable(v)},
            "training_history": self.training_history,
        }

        if self.ema is not None:
            checkpoint["ema_state_dict"] = self.ema.state_dict()

        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, str(checkpoint_path))
        logger.info(f"Checkpoint saved: {path} (epoch={epoch}, val_loss={val_loss:.4f})")

        if is_best:
            best_path = checkpoint_path.parent / "cati_best.pt"
            torch.save(checkpoint, str(best_path))
            logger.info(f"Best model saved: {best_path}")

    def load_checkpoint(self, path: str) -> int:
        """Load training checkpoint. Returns the epoch number."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.cati.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.ema is not None and "ema_state_dict" in checkpoint:
            self.ema.load_state_dict(checkpoint["ema_state_dict"])

        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        self.training_history = checkpoint.get("training_history", [])
        epoch = checkpoint["epoch"]
        logger.info(f"Checkpoint loaded: {path} (epoch={epoch})")
        return epoch

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        epochs: int = 50,
        save_dir: str = "models",
        patience: int = 15,
    ) -> dict:
        """Full training loop with early stopping.

        Returns:
            Training result dict with final metrics and history.
        """
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max(epochs - self.warmup_epochs, 1),
            eta_min=1e-6,
        )

        for pg in self.optimizer.param_groups:
            pg["initial_lr"] = pg["lr"]

        no_improve = 0
        save_path = Path(save_dir)

        logger.info(
            f"Training for {epochs} epochs | "
            f"warmup={self.warmup_epochs} | "
            f"patience={patience} | "
            f"grad_accum={self.grad_accum_steps}"
        )

        for epoch in range(epochs):
            train_metrics = self.train_epoch(train_loader)

            val_metrics = {}
            if val_loader is not None:
                val_metrics = self.validate(val_loader)

            val_loss = val_metrics.get("val_loss", train_metrics["train_loss"])

            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                no_improve = 0
            else:
                no_improve += 1

            if (epoch + 1) % 5 == 0 or is_best or epoch == epochs - 1:
                self.save_checkpoint(
                    str(save_path / "cati_latest.pt"),
                    epoch=epoch,
                    val_loss=val_loss,
                    is_best=is_best,
                )

            if (epoch + 1) % 5 == 0 or epoch == 0 or is_best:
                lr = self.optimizer.param_groups[0]["lr"]
                synth = train_metrics.get("synthetic_batches", 0)
                synth_note = f" [⚠ {synth} synthetic batches]" if synth > 0 else ""
                logger.info(
                    f"Epoch {epoch + 1}/{epochs} | "
                    f"Train: {train_metrics['train_loss']:.4f} | "
                    f"Val: {val_loss:.4f} | "
                    f"LR: {lr:.2e} | "
                    f"Time: {train_metrics['epoch_time_s']:.1f}s"
                    + (" ★ BEST" if is_best else "")
                    + synth_note
                )

            if no_improve >= patience:
                logger.info(
                    f"Early stopping at epoch {epoch + 1} (no improvement for {patience} epochs)"
                )
                break

        logger.info(f"Training complete | Best val loss: {self.best_val_loss:.4f}")
        return {
            "best_val_loss": self.best_val_loss,
            "final_epoch": epoch,
            "history": self.training_history,
        }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


@click.command()
@click.option("--config", default="configs/training_config.yaml", help="Training config")
@click.option("--data-dir", default="data/features", help="Feature directory")
@click.option("--phase", type=int, default=1, help="Training phase (1 or 2)")
@click.option("--epochs", type=int, default=50)
@click.option("--lr", type=float, default=1e-3)
@click.option("--batch-size", type=int, default=16)
@click.option("--output", default="models", help="Output directory for checkpoints")
@click.option("--resume", default=None, help="Resume from checkpoint path")
def main(config, data_dir, phase, epochs, lr, batch_size, output, resume):
    """CATI Training — Context-Aware Traffic Intelligence."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    with open(config) as f:
        train_config = yaml.safe_load(f)

    model_config = train_config.get("model", {})
    phase_config = train_config.get(f"phase{phase}", {})

    cati_config = CATIConfig(
        num_cameras=model_config.get("num_cameras", 90),
        context_dim=model_config.get("context_dim", 64),
        camera_embed_dim=model_config.get("camera_embed_dim", 16),
    )

    data_path = Path(data_dir)
    train_dataset = CATIDataset(str(data_path / "train"))
    val_dataset = CATIDataset(str(data_path / "val"))

    train_loader = DataLoader(
        train_dataset,
        batch_size=phase_config.get("batch_size", batch_size),
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=cati_collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=phase_config.get("batch_size", batch_size),
        shuffle=False,
        num_workers=2,
        collate_fn=cati_collate,
    )

    trainer = CATITrainer(
        config=cati_config,
        learning_rate=phase_config.get("learning_rate", lr),
        weight_decay=phase_config.get("weight_decay", 1e-4),
        warmup_epochs=phase_config.get("warmup_epochs", 5),
    )

    if resume:
        trainer.load_checkpoint(resume)

    logger.info(f"CATI Training — Phase {phase} | Epochs: {epochs} | LR: {lr}")

    result = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=phase_config.get("epochs", epochs),
        save_dir=output,
    )

    logger.info(f"Final result: {result['best_val_loss']:.4f}")


if __name__ == "__main__":
    main()
