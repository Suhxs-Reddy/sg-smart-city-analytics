"""
CATI Phase 2 Training — End-to-End Fine-tuning with FiLM Conditioning

Subclasses ultralytics DetectionTrainer to inject per-image environmental
context into the YOLO backbone via FiLM hooks during training.

Architecture:
    1. Builds a filename→context lookup from feature extraction JSONs
    2. Registers FiLM hooks on backbone layers [4, 6, 9]
    3. Overrides preprocess_batch to set active context before each forward pass
    4. Both YOLO backbone and CATI context modules receive gradients

Usage (from Colab):
    from src.training.train_phase2 import CATIPhase2Trainer

    trainer = CATIPhase2Trainer(
        yolo_dataset_dir='/content/drive/.../yolo_dataset',
        feature_dir='/content/drive/.../features',
        cati_weights_path='/content/drive/.../models/cati_best.pt',
        model_variant='yolo11s',
        epochs=20,
        batch_size=8,
        lr=1e-4,
        device='cuda',
    )
    trainer.train()
"""

import json
import logging
from pathlib import Path
from typing import ClassVar

import torch
import torch.nn as nn

from src.models.cati_detector import CATIConfig, CATIDetector, EMAModel
from src.models.context_encoder import ContextEncoder

logger = logging.getLogger(__name__)


class ContextLookup:
    """Maps image filenames to environmental context tensors.

    Built from the feature extraction JSON files which store metadata
    alongside backbone features. Used by the Phase 2 trainer to inject
    per-image context into the FiLM hooks during training.

    Args:
        feature_dir: Root dir containing train/val/test subdirs with .json files.
        device: Torch device for context tensors.
    """

    def __init__(self, feature_dir: str, device: torch.device):
        self.device = device
        self._lookup: dict[str, dict] = {}
        self._build(Path(feature_dir))
        logger.info(f"ContextLookup: {len(self._lookup)} entries from {feature_dir}")

    def _build(self, root: Path):
        for jf in root.rglob("*.json"):
            meta = json.loads(jf.read_text())
            # Key by stem (e.g. "cam1001_20260405_080100") which matches image filename
            self._lookup[jf.stem] = meta

    def get(self, img_path: str) -> dict | None:
        """Return context dict for a given image path, or None if not found."""
        stem = Path(img_path).stem
        return self._lookup.get(stem)

    def to_tensors(self, meta: dict) -> dict[str, torch.Tensor]:
        """Convert metadata dict to context tensors (batch size 1)."""
        weather_id = ContextEncoder.weather_to_id(meta.get("weather_condition", "unknown"))
        resolution_id = ContextEncoder.resolution_to_id(
            meta.get("image_width", 1920), meta.get("image_height", 1080)
        )
        ctx: dict[str, torch.Tensor] = {
            "weather_id": torch.tensor([weather_id], dtype=torch.long, device=self.device),
            "temperature": torch.tensor([meta.get("temperature_celsius", 28.0)], dtype=torch.float32, device=self.device),
            "pm25": torch.tensor([meta.get("pm25_reading", 15.0)], dtype=torch.float32, device=self.device),
            "hour": torch.tensor([meta.get("hour", 12.0)], dtype=torch.float32, device=self.device),
            "camera_id": torch.tensor([meta.get("camera_idx", 0)], dtype=torch.long, device=self.device),
            "resolution_id": torch.tensor([resolution_id], dtype=torch.long, device=self.device),
        }
        lat = meta.get("camera_latitude")
        lon = meta.get("camera_longitude")
        if lat is not None and lon is not None:
            ctx["camera_lat"] = torch.tensor([float(lat)], dtype=torch.float32, device=self.device)
            ctx["camera_lon"] = torch.tensor([float(lon)], dtype=torch.float32, device=self.device)
        return ctx

    def batch_tensors(self, img_paths: list[str]) -> dict[str, torch.Tensor]:
        """Build batched context tensors for a list of image paths.

        Falls back to default (clear/28°C/camera 0) for images not in lookup.
        """
        default_meta: dict = {
            "weather_condition": "clear", "temperature_celsius": 28.0,
            "pm25_reading": 15.0, "hour": 12.0, "camera_idx": 0,
            "image_width": 1920, "image_height": 1080,
        }
        singles = [self.to_tensors(self.get(p) or default_meta) for p in img_paths]

        # Stack along batch dim for each key
        all_keys = singles[0].keys()
        return {k: torch.cat([s[k] for s in singles], dim=0) for k in all_keys}


class CATIPhase2Trainer:
    """End-to-end YOLO + CATI fine-tuning with FiLM conditioning.

    Registers FiLM hooks on the YOLO backbone and trains jointly using
    YOLO detection loss. Context is injected per batch from the metadata
    lookup built from feature extraction JSONs.

    Args:
        yolo_dataset_dir: Path to prepared YOLO dataset (with data.yaml).
        feature_dir: Path to feature extraction output (for context lookup).
        cati_weights_path: Phase 1 CATI checkpoint.
        model_variant: YOLO model variant ('yolo11s', 'yolo11m', etc.).
        epochs: Training epochs.
        batch_size: Batch size (reduce if OOM).
        lr: Learning rate for YOLO backbone (CATI modules use 10× this).
        device: Compute device.
        freeze_backbone_epochs: Epochs to keep backbone frozen (CATI only).
        save_dir: Directory for checkpoints.
    """

    HOOK_LAYERS: ClassVar[list[int]] = [4, 6, 9]  # P3, P4, P5

    def __init__(
        self,
        yolo_dataset_dir: str,
        feature_dir: str,
        cati_weights_path: str,
        model_variant: str = "yolo11s",
        epochs: int = 20,
        batch_size: int = 8,
        lr: float = 1e-4,
        device: str = "cuda",
        freeze_backbone_epochs: int = 3,
        save_dir: str = "models/phase2",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.yolo_dataset_dir = yolo_dataset_dir
        self.feature_dir = feature_dir
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.freeze_backbone_epochs = freeze_backbone_epochs
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Load YOLO
        from ultralytics import YOLO
        self.yolo = YOLO(f"{model_variant}.pt")
        logger.info(f"YOLO {model_variant} loaded")

        # Load CATI with Phase 1 weights
        config = CATIConfig(
            num_cameras=90, context_dim=64, use_gps_encoding=True, use_attention=True
        )
        self.cati = CATIDetector(config).to(self.device)
        if cati_weights_path and Path(cati_weights_path).exists():
            ckpt = torch.load(cati_weights_path, map_location=self.device, weights_only=False)
            self.cati.load_state_dict(ckpt["model_state_dict"])
            logger.info(f"Phase 1 CATI weights loaded from {cati_weights_path}")

        self.ema = EMAModel(self.cati, decay=0.9999)

        # Context lookup from feature JSONs
        self.ctx_lookup = ContextLookup(feature_dir, self.device)

        # State for hook communication
        self._active_ctx: dict[str, torch.Tensor] | None = None
        self._film_cache: list[tuple[torch.Tensor, torch.Tensor]] | None = None
        self._ctx_vec: torch.Tensor | None = None
        self._hooks: list = []

    def _register_hooks(self):
        """Register FiLM conditioning hooks on backbone layers."""
        backbone = self.yolo.model.model
        self._hooks = []
        self._film_cache = None

        for stage_idx, layer_idx in enumerate(self.HOOK_LAYERS):
            def _make_hook(s_idx: int):
                def hook(_module: nn.Module, _input: tuple, output: torch.Tensor) -> torch.Tensor:
                    ctx = self._active_ctx
                    if ctx is None:
                        return output

                    feat = output[0] if isinstance(output, tuple | list) else output
                    if not isinstance(feat, torch.Tensor):
                        return output

                    feat = feat.to(self.device)

                    # Compute FiLM params once (on P3), reuse for P4/P5
                    if s_idx == 0 or self._film_cache is None:
                        context_vec = self.cati.encode_context(**ctx)
                        self._film_cache = self.cati.get_film_params(context_vec)
                        self._ctx_vec = context_vec

                    gamma, beta = self._film_cache[s_idx]
                    return self.cati.film_layers[s_idx](feat, gamma, beta, self._ctx_vec)

                return hook

            handle = backbone[layer_idx].register_forward_hook(_make_hook(stage_idx))
            self._hooks.append(handle)

        logger.info(f"Registered {len(self._hooks)} FiLM hooks")

    def _remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def train(self):
        """Run Phase 2 end-to-end training."""
        data_yaml = str(Path(self.yolo_dataset_dir) / "data.yaml")
        if not Path(data_yaml).exists():
            raise FileNotFoundError(f"data.yaml not found at {data_yaml}. Run prepare_dataset.ipynb first.")

        self._register_hooks()

        # Build YOLO optimizer param groups: backbone at lr, CATI at 10×lr
        # ultralytics handles YOLO optimizer internally — we add CATI params separately
        cati_optimizer = torch.optim.AdamW(
            self.cati.parameters(), lr=self.lr * 10, weight_decay=1e-4
        )

        logger.info(
            f"Phase 2 training: {self.epochs} epochs | "
            f"batch={self.batch_size} | lr={self.lr} | "
            f"freeze_backbone={self.freeze_backbone_epochs} epochs"
        )

        # Subclass DetectionTrainer to override preprocess_batch where
        # the batch dict (with im_file paths) is guaranteed to be available.
        from ultralytics.models.yolo.detect import DetectionTrainer

        cati_trainer = self  # capture for closure

        class CATIDetectionTrainer(DetectionTrainer):
            def preprocess_batch(self, batch):
                batch = super().preprocess_batch(batch)
                img_paths = batch.get("im_file", [])
                if img_paths:
                    try:
                        cati_trainer._active_ctx = cati_trainer.ctx_lookup.batch_tensors(img_paths)
                        cati_trainer._film_cache = None
                    except Exception as e:
                        logger.warning(f"Context lookup failed: {e}")
                        cati_trainer._active_ctx = None
                else:
                    cati_trainer._active_ctx = None
                return batch

        def on_train_batch_end(trainer_obj):
            if cati_trainer._active_ctx is not None:
                cati_optimizer.step()
                cati_optimizer.zero_grad()
                cati_trainer.ema.update(cati_trainer.cati)

        def on_train_epoch_end(trainer_obj):
            epoch = trainer_obj.epoch
            # Only freeze backbone layers (0 → max hook layer); head must stay
            # trainable so the detection loss always has a grad_fn path.
            for i, layer in enumerate(self.yolo.model.model):
                if i <= max(self.HOOK_LAYERS):
                    for p in layer.parameters():
                        p.requires_grad_(epoch >= self.freeze_backbone_epochs)
            if (epoch + 1) % 5 == 0:
                ckpt_path = self.save_dir / f"cati_phase2_epoch{epoch+1}.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.cati.state_dict(),
                    "ema_state_dict": self.ema.state_dict(),
                    "optimizer_state_dict": cati_optimizer.state_dict(),
                }, str(ckpt_path))
                logger.info(f"CATI checkpoint saved: {ckpt_path}")

        def on_train_end(trainer_obj):
            self._remove_hooks()
            final_path = self.save_dir / "cati_phase2_final.pt"
            torch.save({
                "epoch": self.epochs,
                "model_state_dict": self.cati.state_dict(),
                "ema_state_dict": self.ema.state_dict(),
            }, str(final_path))
            logger.info(f"Phase 2 complete. CATI saved to {final_path}")

        self.yolo.add_callback("on_train_batch_end", on_train_batch_end)
        self.yolo.add_callback("on_train_epoch_end", on_train_epoch_end)
        self.yolo.add_callback("on_train_end", on_train_end)

        # Start YOLO training using the custom trainer subclass
        results = self.yolo.train(
            data=data_yaml,
            epochs=self.epochs,
            batch=self.batch_size,
            imgsz=640,
            lr0=self.lr,
            lrf=0.01,
            warmup_epochs=3,
            device=str(self.device),
            project=str(self.save_dir),
            name="yolo_cati",
            save=True,
            verbose=True,
            workers=0,
            trainer=CATIDetectionTrainer,
        )

        return results
