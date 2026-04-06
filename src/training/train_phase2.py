"""
CATI Phase 2 Training — End-to-End Fine-tuning with FiLM Conditioning

Subclasses ultralytics DetectionTrainer to inject per-image environmental
context into the YOLO backbone (and optionally neck) via FiLM hooks during
training.

Architecture:
    1. Builds a filename→context lookup from feature extraction JSONs
    2. Registers FiLM hooks on backbone layers [4, 6, 9]
       Optionally adds neck hooks on PAN output layers (set NECK_HOOK_LAYERS
       after running verify_layers() to confirm indices for your YOLO variant).
    3. Overrides preprocess_batch to set active context before each forward pass
    4. Both YOLO backbone and CATI context modules receive gradients
    5. Auxiliary context regularization loss keeps the context encoder grounded

Usage (from Colab):
    from src.training.train_phase2 import CATIPhase2Trainer

    # Step 1: verify neck layer indices (run once before training)
    CATIPhase2Trainer.verify_layers(yolo_model_path='yolo11s.pt')

    # Step 2: train
    trainer = CATIPhase2Trainer(
        yolo_dataset_dir='/content/drive/.../yolo_dataset',
        feature_dir='/content/drive/.../features',
        cati_weights_path='/content/drive/.../models/cati_best.pt',
        model_variant='yolo11s',
        epochs=20,
        batch_size=8,
        lr=1e-4,
        device='cuda',
        use_neck_film=True,   # enable after verify_layers confirms indices
    )
    trainer.train()
"""

import json
import logging
import math
from pathlib import Path
from typing import ClassVar

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.cati_detector import (
    YOLO11S_NECK_CHANNEL_DIMS,
    CATIConfig,
    CATIDetector,
    EMAModel,
)
from src.models.context_encoder import WEATHER_CONDITIONS, ContextEncoder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Auxiliary context regularization head (Phase 2 only)
# ---------------------------------------------------------------------------


class _ContextRegHead(nn.Module):
    """Predicts weather/hour/temperature directly from the context vector.

    Used as a lightweight auxiliary loss during Phase 2 to prevent the context
    encoder from drifting away from what it learned in Phase 1. Unlike the
    Phase 1 ContextPredictionHead (which requires backbone features), this head
    only needs the 64-dim context vector — so it can run as a second forward
    pass independently of YOLO's computation graph.

    Loss weight is 0.1× to keep it subordinate to the detection loss.
    """

    AUX_WEIGHT: float = 0.1

    def __init__(self, context_dim: int, num_weather_classes: int = len(WEATHER_CONDITIONS)):
        super().__init__()
        self.weather_head = nn.Linear(context_dim, num_weather_classes)
        self.hour_head = nn.Linear(context_dim, 2)   # sin/cos encoding
        self.temp_head = nn.Linear(context_dim, 1)   # normalized temperature

    def compute_loss(
        self,
        ctx_vec: torch.Tensor,
        ctx: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute weighted auxiliary loss from context vector + ground truth."""
        weather_loss = F.cross_entropy(self.weather_head(ctx_vec), ctx["weather_id"].long())

        hour = ctx["hour"].float()
        hour_target = torch.stack(
            [torch.sin(2 * math.pi * hour / 24), torch.cos(2 * math.pi * hour / 24)],
            dim=-1,
        )
        hour_loss = F.mse_loss(self.hour_head(ctx_vec), hour_target)

        temp_target = (ctx["temperature"].float() - 29.0) / 5.0
        temp_loss = F.mse_loss(self.temp_head(ctx_vec).squeeze(-1), temp_target)

        return self.AUX_WEIGHT * (weather_loss + hour_loss + temp_loss)


# ---------------------------------------------------------------------------
# Context lookup
# ---------------------------------------------------------------------------


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
        all_keys = singles[0].keys()
        return {k: torch.cat([s[k] for s in singles], dim=0) for k in all_keys}


# ---------------------------------------------------------------------------
# Phase 2 Trainer
# ---------------------------------------------------------------------------


class CATIPhase2Trainer:
    """End-to-end YOLO + CATI fine-tuning with FiLM conditioning.

    Registers FiLM hooks on the YOLO backbone (and optionally neck) and trains
    jointly using YOLO detection loss plus an auxiliary context regularization
    loss that keeps the context encoder grounded to its Phase 1 representation.

    Args:
        yolo_dataset_dir: Path to prepared YOLO dataset (with data.yaml).
        feature_dir: Path to feature extraction output (for context lookup).
        cati_weights_path: Phase 1 CATI checkpoint.
        model_variant: YOLO model variant ('yolo11s', 'yolo11m', etc.).
        epochs: Training epochs.
        batch_size: Batch size (reduce if OOM).
        lr: Learning rate for YOLO backbone (CATI modules use 10× this).
        device: Compute device.
        freeze_backbone_epochs: Epochs to keep backbone frozen (head + CATI only).
        save_dir: Directory for checkpoints.
        use_neck_film: Also condition neck PAN layers (run verify_layers() first).
        neck_hook_layers: Explicit neck layer indices (overrides class default).
    """

    # Backbone P3/P4/P5 — verified for YOLOv11s
    BACKBONE_HOOK_LAYERS: ClassVar[list[int]] = [4, 6, 9]

    # PAN neck outputs — run verify_layers() to confirm for your variant.
    # Approximate for YOLOv11s: P3@128ch, P4@256ch, P5@512ch.
    NECK_HOOK_LAYERS: ClassVar[list[int]] = [16, 19, 22]

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
        use_neck_film: bool = False,
        neck_hook_layers: list[int] | None = None,
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
        self.use_neck_film = use_neck_film
        self._neck_hook_layers = neck_hook_layers or self.NECK_HOOK_LAYERS

        # Combined hook layers: backbone first, then neck (if enabled)
        self._all_hook_layers = list(self.BACKBONE_HOOK_LAYERS)
        if use_neck_film:
            self._all_hook_layers += self._neck_hook_layers

        # Load YOLO
        from ultralytics import YOLO
        self.yolo = YOLO(f"{model_variant}.pt")
        logger.info(f"YOLO {model_variant} loaded")

        # Load CATI with Phase 1 weights
        neck_ch = list(YOLO11S_NECK_CHANNEL_DIMS) if use_neck_film else []
        config = CATIConfig(
            num_cameras=90, context_dim=64,
            use_gps_encoding=True, use_attention=True,
            neck_channels=neck_ch,
        )
        self.cati = CATIDetector(config).to(self.device)
        if cati_weights_path and Path(cati_weights_path).exists():
            ckpt = torch.load(cati_weights_path, map_location=self.device, weights_only=False)
            # strict=False so neck components (not in Phase 1 ckpt) init fresh
            self.cati.load_state_dict(ckpt["model_state_dict"], strict=False)
            logger.info(f"Phase 1 CATI weights loaded from {cati_weights_path}")

        self.ema = EMAModel(self.cati, decay=0.9999)

        # Auxiliary context regularization head (Phase 2 specific)
        self._ctx_reg_head = _ContextRegHead(
            context_dim=config.context_dim,
        ).to(self.device)

        # Context lookup from feature JSONs
        self.ctx_lookup = ContextLookup(feature_dir, self.device)

        # State for hook communication
        self._active_ctx: dict[str, torch.Tensor] | None = None
        self._film_cache: list[tuple[torch.Tensor, torch.Tensor]] | None = None
        self._neck_film_cache: list[tuple[torch.Tensor, torch.Tensor]] | None = None
        self._ctx_vec: torch.Tensor | None = None
        self._hooks: list = []

        neck_info = f" + neck FiLM layers {self._neck_hook_layers}" if use_neck_film else ""
        logger.info(
            f"CATIPhase2Trainer ready | backbone hooks={self.BACKBONE_HOOK_LAYERS}"
            f"{neck_info} | aux_loss_weight={_ContextRegHead.AUX_WEIGHT}"
        )

    @staticmethod
    def verify_layers(yolo_model_path: str = "yolo11s.pt", check_layers: list[int] | None = None):
        """Print output shapes for YOLO layers to identify neck hook targets.

        Run this once in Colab before training with use_neck_film=True.
        Look for the PAN output layers — typically three C3k2/C2f blocks
        producing [128, 256, 512] channels at [80x80, 40x40, 20x20].

        Args:
            yolo_model_path: Path or name of YOLO weights.
            check_layers: Specific layer indices to inspect (default: 10–25).
        """
        import torch
        from ultralytics import YOLO

        if check_layers is None:
            check_layers = list(range(10, 26))

        yolo = YOLO(yolo_model_path)
        model = yolo.model.model
        captured: dict[int, tuple] = {}
        hooks = []

        for idx in check_layers:
            if idx >= len(model):
                break
            def _make_hook(i):
                def hook(_, _in, output):
                    feat = output[0] if isinstance(output, tuple | list) else output
                    if isinstance(feat, torch.Tensor):
                        captured[i] = tuple(feat.shape)
                return hook
            hooks.append(model[idx].register_forward_hook(_make_hook(idx)))

        dummy = torch.zeros(1, 3, 640, 640)
        yolo.model(dummy)
        for h in hooks:
            h.remove()

        print(f"\nYOLO layer shapes ({yolo_model_path}) — neck starts at layer 10:")
        print(f"{'Layer':<8} {'Class':<20} {'Output shape'}")
        print("-" * 50)
        for idx in check_layers:
            if idx >= len(model):
                break
            shape = captured.get(idx, "no output captured")
            cls = type(model[idx]).__name__
            marker = " ← neck FiLM candidate" if (
                isinstance(shape, tuple) and len(shape) == 4
                and shape[2] in (80, 40, 20)
            ) else ""
            print(f"{idx:<8} {cls:<20} {shape}{marker}")

        print("\nLook for 3 layers with shapes (B, 128, 80, 80), (B, 256, 40, 40), (B, 512, 20, 20)")
        print("Set those layer indices as NECK_HOOK_LAYERS in the trainer.")
        return captured

    def _register_hooks(self):
        """Register FiLM conditioning hooks on backbone (and optionally neck) layers."""
        model = self.yolo.model.model
        self._hooks = []
        self._film_cache = None
        self._neck_film_cache = None
        num_backbone = len(self.BACKBONE_HOOK_LAYERS)

        for stage_idx, layer_idx in enumerate(self._all_hook_layers):
            is_neck_stage = stage_idx >= num_backbone
            neck_stage_idx = stage_idx - num_backbone  # index within neck stages

            def _make_hook(s_idx: int, is_neck: bool, n_idx: int):
                def hook(_module: nn.Module, _input: tuple, output: torch.Tensor) -> torch.Tensor:
                    ctx = self._active_ctx
                    if ctx is None:
                        return output

                    feat = output[0] if isinstance(output, tuple | list) else output
                    if not isinstance(feat, torch.Tensor):
                        return output

                    feat = feat.to(self.device)

                    if not is_neck:
                        # Backbone stage: compute context vec + FiLM params on first stage
                        if s_idx == 0 or self._film_cache is None:
                            context_vec = self.cati.encode_context(**ctx)
                            self._film_cache = self.cati.get_film_params(context_vec)
                            self._ctx_vec = context_vec
                            # Also pre-compute neck FiLM params if neck is enabled
                            if self.use_neck_film:
                                self._neck_film_cache = self.cati.get_neck_film_params(context_vec)
                        gamma, beta = self._film_cache[s_idx]
                        return self.cati.film_layers[s_idx](feat, gamma, beta, self._ctx_vec)
                    else:
                        # Neck stage: reuse context vec already computed by backbone hooks
                        if self._neck_film_cache is None or self._ctx_vec is None:
                            return output  # context not ready; pass through
                        gamma, beta = self._neck_film_cache[n_idx]
                        return self.cati.neck_film_layers[n_idx](feat, gamma, beta, self._ctx_vec)

                return hook

            handle = model[layer_idx].register_forward_hook(
                _make_hook(stage_idx, is_neck_stage, neck_stage_idx)
            )
            self._hooks.append(handle)

        logger.info(
            f"Registered {len(self._hooks)} FiLM hooks "
            f"(backbone={self.BACKBONE_HOOK_LAYERS}"
            f"{', neck=' + str(self._neck_hook_layers) if self.use_neck_film else ''})"
        )

    def _remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    @staticmethod
    def evaluate(
        yolo_weights_path: str,
        cati_weights_path: str,
        feature_dir: str,
        data_yaml: str,
        device: str = "cuda",
        use_neck_film: bool = False,
        neck_hook_layers: list[int] | None = None,
    ) -> object:
        """Evaluate CATI with FiLM hooks and per-image context active.

        Use this instead of YOLO.val() for a fair CATI evaluation.
        Plain YOLO.val() loads backbone weights only — FiLM hooks are never
        registered, so it measures fine-tuned backbone performance, not CATI.

        Args:
            yolo_weights_path: Path to Phase 2 YOLO best.pt.
            cati_weights_path: Path to CATI checkpoint (cati_phase2_final.pt).
            feature_dir: Feature dir with context JSONs for per-image lookup.
            data_yaml: Path to YOLO dataset data.yaml.
            device: Compute device.
            use_neck_film: Must match how Phase 2 was trained.
            neck_hook_layers: Neck layer indices (default NECK_HOOK_LAYERS).

        Returns:
            ultralytics val metrics object (metrics.box.map50, etc.)
        """
        from ultralytics import YOLO
        from ultralytics.models.yolo.detect import DetectionValidator

        _device = torch.device(device if torch.cuda.is_available() else "cpu")
        _neck_layers = neck_hook_layers or CATIPhase2Trainer.NECK_HOOK_LAYERS

        # Build CATI module with same config as training
        neck_ch = list(YOLO11S_NECK_CHANNEL_DIMS) if use_neck_film else []
        config = CATIConfig(
            num_cameras=90, context_dim=64,
            use_gps_encoding=True, use_attention=True,
            neck_channels=neck_ch,
            use_context_augmentation=False,  # no augmentation at eval
        )
        cati = CATIDetector(config).to(_device)
        if Path(cati_weights_path).exists():
            ckpt = torch.load(cati_weights_path, map_location=_device, weights_only=False)
            state = ckpt.get("model_state_dict", ckpt)
            missing, unexpected = cati.load_state_dict(state, strict=False)
            if missing:
                logger.warning(f"CATI eval: missing keys: {missing[:5]}")
            logger.info(f"CATI eval weights loaded from {cati_weights_path}")
        else:
            logger.warning(f"CATI weights not found at {cati_weights_path} — evaluating without conditioning")
        cati.eval()

        # Context lookup for per-image context injection
        ctx_lookup = ContextLookup(feature_dir, _device)

        # Shared state — hooks and validator communicate through this dict
        _state: dict = {
            "active_ctx": None,
            "film_cache": None,
            "neck_film_cache": None,
            "ctx_vec": None,
        }

        # Load YOLO and register FiLM hooks
        yolo = YOLO(yolo_weights_path)
        model = yolo.model.model

        all_layers = list(CATIPhase2Trainer.BACKBONE_HOOK_LAYERS)
        if use_neck_film:
            all_layers += _neck_layers
        num_backbone = len(CATIPhase2Trainer.BACKBONE_HOOK_LAYERS)
        hooks = []

        for stage_idx, layer_idx in enumerate(all_layers):
            is_neck = stage_idx >= num_backbone
            n_idx = stage_idx - num_backbone

            def _make_hook(s_idx: int, is_neck_stage: bool, neck_stage_idx: int):
                def hook(_mod: nn.Module, _in: tuple, output: torch.Tensor) -> torch.Tensor:
                    ctx = _state["active_ctx"]
                    if ctx is None:
                        return output
                    feat = output[0] if isinstance(output, tuple | list) else output
                    if not isinstance(feat, torch.Tensor):
                        return output
                    feat = feat.to(_device)

                    if not is_neck_stage:
                        if s_idx == 0 or _state["film_cache"] is None:
                            with torch.no_grad():
                                ctx_vec = cati.encode_context(**ctx)
                                _state["film_cache"] = cati.get_film_params(ctx_vec)
                                _state["ctx_vec"] = ctx_vec
                                if use_neck_film:
                                    _state["neck_film_cache"] = cati.get_neck_film_params(ctx_vec)
                        gamma, beta = _state["film_cache"][s_idx]
                        with torch.no_grad():
                            return cati.film_layers[s_idx](feat, gamma, beta, _state["ctx_vec"])
                    else:
                        if _state["neck_film_cache"] is None or _state["ctx_vec"] is None:
                            return output
                        gamma, beta = _state["neck_film_cache"][neck_stage_idx]
                        with torch.no_grad():
                            return cati.neck_film_layers[neck_stage_idx](
                                feat, gamma, beta, _state["ctx_vec"]
                            )
                return hook

            hooks.append(model[layer_idx].register_forward_hook(
                _make_hook(stage_idx, is_neck, n_idx)
            ))

        logger.info(f"CATI eval: {len(hooks)} FiLM hooks registered")

        # Subclass DetectionValidator to inject per-image context before each batch
        class CATIDetectionValidator(DetectionValidator):
            def preprocess_batch(self, batch):
                batch = super().preprocess_batch(batch)
                img_paths = batch.get("im_file", [])
                if img_paths:
                    try:
                        _state["active_ctx"] = ctx_lookup.batch_tensors(img_paths)
                        _state["film_cache"] = None
                        _state["neck_film_cache"] = None
                    except Exception as e:
                        logger.warning(f"Context lookup failed during eval: {e}")
                        _state["active_ctx"] = None
                else:
                    _state["active_ctx"] = None
                return batch

        try:
            metrics = yolo.val(
                data=data_yaml,
                imgsz=640,
                device=str(_device),
                verbose=True,
                validator=CATIDetectionValidator,
            )
        finally:
            for h in hooks:
                h.remove()
            _state["active_ctx"] = None

        return metrics

    def train(self):
        """Run Phase 2 end-to-end training."""
        data_yaml = str(Path(self.yolo_dataset_dir) / "data.yaml")
        if not Path(data_yaml).exists():
            raise FileNotFoundError(
                f"data.yaml not found at {data_yaml}. Run prepare_dataset.ipynb first."
            )

        self._register_hooks()

        # CATI optimizer: all CATI params at 10×lr + reg head at same rate
        cati_params = list(self.cati.parameters()) + list(self._ctx_reg_head.parameters())
        cati_optimizer = torch.optim.AdamW(cati_params, lr=self.lr * 10, weight_decay=1e-4)

        logger.info(
            f"Phase 2 training: {self.epochs} epochs | "
            f"batch={self.batch_size} | lr={self.lr} | "
            f"freeze_backbone={self.freeze_backbone_epochs} epochs | "
            f"neck_film={self.use_neck_film}"
        )

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
                        cati_trainer._neck_film_cache = None
                    except Exception as e:
                        logger.warning(f"Context lookup failed: {e}")
                        cati_trainer._active_ctx = None
                else:
                    cati_trainer._active_ctx = None
                return batch

        def on_train_batch_end(trainer_obj):
            ctx = cati_trainer._active_ctx
            if ctx is None:
                return

            # Auxiliary context regularization: re-encode context independently
            # of YOLO's graph (which is already freed) and compute a small loss
            # that keeps the encoder grounded to its Phase 1 representation.
            try:
                ctx_vec = cati_trainer.cati.encode_context(**ctx)
                aux_loss = cati_trainer._ctx_reg_head.compute_loss(ctx_vec, ctx)
                aux_loss.backward()
            except Exception as e:
                logger.warning(f"Aux loss failed: {e}")

            cati_optimizer.step()
            cati_optimizer.zero_grad()
            cati_trainer.ema.update(cati_trainer.cati)

        def on_train_epoch_end(trainer_obj):
            epoch = trainer_obj.epoch
            # Only freeze backbone layers; head must stay trainable for loss to flow
            max_backbone_layer = max(self.BACKBONE_HOOK_LAYERS)
            for i, layer in enumerate(self.yolo.model.model):
                if i <= max_backbone_layer:
                    for p in layer.parameters():
                        p.requires_grad_(epoch >= self.freeze_backbone_epochs)
            if (epoch + 1) % 5 == 0:
                ckpt_path = self.save_dir / f"cati_phase2_epoch{epoch+1}.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.cati.state_dict(),
                    "ema_state_dict": self.ema.state_dict(),
                    "optimizer_state_dict": cati_optimizer.state_dict(),
                    "ctx_reg_head_state_dict": self._ctx_reg_head.state_dict(),
                }, str(ckpt_path))
                logger.info(f"CATI checkpoint saved: {ckpt_path}")

        def on_train_end(trainer_obj):
            self._remove_hooks()
            final_path = self.save_dir / "cati_phase2_final.pt"
            torch.save({
                "epoch": self.epochs,
                "model_state_dict": self.cati.state_dict(),
                "ema_state_dict": self.ema.state_dict(),
                "ctx_reg_head_state_dict": self._ctx_reg_head.state_dict(),
            }, str(final_path))
            logger.info(f"Phase 2 complete. CATI saved to {final_path}")

        self.yolo.add_callback("on_train_batch_end", on_train_batch_end)
        self.yolo.add_callback("on_train_epoch_end", on_train_epoch_end)
        self.yolo.add_callback("on_train_end", on_train_end)

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
