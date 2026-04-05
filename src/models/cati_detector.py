"""
CATI — Context-Aware Traffic Intelligence Detector (Production Grade)

A novel detection architecture that conditions YOLOv11's feature extraction on
environmental metadata using Feature-wise Linear Modulation (FiLM) with
adaptive gating and channel/spatial attention.

Architecture overview:
    ┌─────────────────────────────────────────────────────────────┐
    │                    CATI Detector                             │
    │                                                             │
    │  1. YOLO backbone processes image → P3, P4, P5 features    │
    │  2. Forward hooks intercept features at each stage          │
    │  3. Context encoder processes metadata → context vector     │
    │  4. FiLM generator produces (γ, β) per stage               │
    │  5. AdaptiveFiLM applies gated conditioning + attention     │
    │  6. Modified features continue to detection head            │
    │  7. EMA model provides stable inference weights             │
    └─────────────────────────────────────────────────────────────┘

Key insight: At inference time on Singapore's LTA camera network, we know:
    1. Which camera is being processed (fixed viewpoint → learnable priors)
    2. Current weather conditions (from data.gov.sg API)
    3. Time of day (lighting conditions, rush hour patterns)
    4. Camera resolution (78 @ 1080p, 11 @ 320×240)
    5. Air quality / PM2.5 (affects visibility)
    6. Camera GPS location (spatial relationships)

Novel contributions:
    - First application of FiLM + adaptive gating to traffic detection
    - Per-camera GPS positional encoding captures spatial priors
    - Context-dependent gating learns when conditioning helps vs. hurts
    - CBAM post-attention selectively amplifies useful conditioning
    - ~300K parameter overhead on a 9.4M backbone

Reference:
    Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer"
    AAAI 2018 — adapted for environmental conditioning in traffic detection
"""

import copy
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import torch
import torch.nn as nn

from src.models.context_encoder import ContextEncoder
from src.models.film import AdaptiveFiLMLayer, FiLMGenerator

logger = logging.getLogger(__name__)

# YOLOv11s backbone channel dimensions at P3/P4/P5 (layers 4, 6, 9)
# Verified via forward hook: P3(layer4)=256, P4(layer6)=256, P5(layer9)=512
YOLO11S_CHANNEL_DIMS = [256, 256, 512]


@dataclass
class CATIConfig:
    """Configuration for the CATI detector.

    Attributes:
        num_cameras: Total cameras in Singapore's LTA network.
        context_dim: Dimension of the context embedding vector.
        camera_embed_dim: Dimension of per-camera learned embedding.
        weather_embed_dim: Dimension of weather condition embedding.
        backbone_channels: Channel dimensions at P3/P4/P5.
        num_classes: Traffic object classes.
        conf_threshold: Detection confidence threshold.
        iou_threshold: NMS IoU threshold.
        img_size: Input image size.
        use_gps_encoding: Enable camera GPS positional encoding.
        use_attention: Enable CBAM post-attention in FiLM layers.
        use_adaptive_gate: Enable context-dependent gating.
        use_context_augmentation: Augment context during training.
        ema_decay: EMA decay rate (0 = disabled).
    """

    num_cameras: int = 90
    context_dim: int = 64
    camera_embed_dim: int = 16
    weather_embed_dim: int = 16
    backbone_channels: list[int] = field(default_factory=lambda: list(YOLO11S_CHANNEL_DIMS))
    num_classes: int = 6
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    img_size: int = 640
    use_gps_encoding: bool = True
    use_attention: bool = True
    use_adaptive_gate: bool = True
    use_context_augmentation: bool = True
    ema_decay: float = 0.9999


class CATIDetector(nn.Module):
    """Context-Aware Traffic Intelligence detector module.

    This module contains ONLY the context conditioning pathway:
        - ContextEncoder: metadata → context vector
        - FiLMGenerator: context → (γ, β) parameters
        - AdaptiveFiLMLayer: applies gated FiLM conditioning + attention

    It does NOT contain the YOLO backbone — that's handled by
    CATIBackboneWrapper which hooks into ultralytics' YOLO.

    Args:
        config: CATIConfig with architecture hyperparameters.
    """

    def __init__(self, config: CATIConfig | None = None):
        super().__init__()
        self.config = config or CATIConfig()

        # Context encoding pathway
        self.context_encoder = ContextEncoder(
            num_cameras=self.config.num_cameras,
            camera_embed_dim=self.config.camera_embed_dim,
            weather_embed_dim=self.config.weather_embed_dim,
            context_dim=self.config.context_dim,
            use_gps_encoding=self.config.use_gps_encoding,
            use_augmentation=self.config.use_context_augmentation,
        )

        # FiLM parameter generation
        self.film_generator = FiLMGenerator(
            context_dim=self.config.context_dim,
            channel_dims=self.config.backbone_channels,
            use_spectral_norm=True,
        )

        # Adaptive FiLM layers (one per backbone stage)
        self.film_layers = nn.ModuleList(
            [
                AdaptiveFiLMLayer(
                    num_channels=dim,
                    context_dim=self.config.context_dim,
                    use_attention=self.config.use_attention,
                    se_reduction=16,
                )
                for dim in self.config.backbone_channels
            ]
        )

        logger.info(
            f"CATI initialized: {self.config.num_cameras} cameras, "
            f"context_dim={self.config.context_dim}, "
            f"FiLM stages={len(self.config.backbone_channels)}, "
            f"attention={self.config.use_attention}, "
            f"gps={self.config.use_gps_encoding}"
        )

    def encode_context(
        self,
        weather_id: torch.Tensor,
        temperature: torch.Tensor,
        pm25: torch.Tensor,
        hour: torch.Tensor,
        camera_id: torch.Tensor,
        resolution_id: torch.Tensor,
        camera_lat: torch.Tensor | None = None,
        camera_lon: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode environmental metadata into a dense context vector."""
        return self.context_encoder(
            weather_id, temperature, pm25, hour, camera_id, resolution_id,
            camera_lat, camera_lon,
        )

    def get_film_params(self, context: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Generate FiLM (γ, β) parameters from context embedding."""
        return self.film_generator(context)

    def apply_film(
        self,
        features: list[torch.Tensor],
        film_params: list[tuple[torch.Tensor, torch.Tensor]],
        context: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Apply adaptive FiLM conditioning to backbone feature maps."""
        modulated = []
        for feat, (gamma, beta), film_layer in zip(
            features, film_params, self.film_layers, strict=True
        ):
            modulated.append(film_layer(feat, gamma, beta, context))
        return modulated

    def forward(
        self,
        backbone_features: list[torch.Tensor],
        weather_id: torch.Tensor,
        temperature: torch.Tensor,
        pm25: torch.Tensor,
        hour: torch.Tensor,
        camera_id: torch.Tensor,
        resolution_id: torch.Tensor,
        camera_lat: torch.Tensor | None = None,
        camera_lon: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        """Full forward pass: encode context → generate FiLM → apply to features.

        Args:
            backbone_features: List of feature maps from YOLO backbone stages.
            weather_id, temperature, pm25, hour, camera_id, resolution_id:
                Environmental context tensors, each (B,).
            camera_lat, camera_lon: Optional GPS coordinates (B,).

        Returns:
            List of conditioned feature maps ready for the detection head.
        """
        context = self.encode_context(
            weather_id, temperature, pm25, hour, camera_id, resolution_id,
            camera_lat, camera_lon,
        )
        film_params = self.get_film_params(context)
        return self.apply_film(backbone_features, film_params, context)

    def count_parameters(self) -> dict:
        """Count parameters by component for reporting."""
        context_params = sum(p.numel() for p in self.context_encoder.parameters())
        film_gen_params = sum(p.numel() for p in self.film_generator.parameters())
        film_layer_params = sum(p.numel() for p in self.film_layers.parameters())
        total = context_params + film_gen_params + film_layer_params
        return {
            "context_encoder": context_params,
            "film_generator": film_gen_params,
            "film_layers": film_layer_params,
            "total_cati_overhead": total,
        }


class EMAModel:
    """Exponential Moving Average model for stable inference weights.

    Maintains a shadow copy of model parameters updated after each step:
        θ_ema = decay × θ_ema + (1 - decay) × θ_model

    Args:
        model: The model to track.
        decay: EMA decay rate (higher = slower updates, smoother).
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        """Update EMA parameters."""
        for s_param, m_param in zip(self.shadow.parameters(), model.parameters(), strict=False):
            s_param.data.mul_(self.decay).add_(m_param.data, alpha=1 - self.decay)

    def state_dict(self) -> dict:
        return self.shadow.state_dict()

    def load_state_dict(self, state_dict: dict):
        self.shadow.load_state_dict(state_dict)


class CATIBackboneWrapper:
    """End-to-end wrapper that hooks into YOLO backbone for FiLM conditioning.

    Uses PyTorch forward hooks to intercept feature maps at P3/P4/P5 stages
    of the YOLO backbone, apply FiLM conditioning, and replace features
    before they reach the detection head.

    Args:
        yolo_model_path: Path to YOLOv11 weights.
        config: CATIConfig.
        cati_weights_path: Path to trained CATI module weights.
        device: Compute device.
    """

    # YOLOv11s backbone layer indices for P3/P4/P5 — verified via verify_hooks()
    HOOK_LAYER_NAMES: ClassVar[dict[str, list[int]]] = {
        "yolo11s": [4, 6, 9],
        "yolo11m": [4, 6, 9],
        "yolo11l": [4, 6, 9],
    }

    def __init__(
        self,
        yolo_model_path: str = "yolo11s.pt",
        config: CATIConfig | None = None,
        cati_weights_path: str | None = None,
        device: str = "auto",
    ):
        self.config = config or CATIConfig()
        self.device = self._resolve_device(device)

        # Load YOLO backbone
        try:
            from ultralytics import YOLO

            self.yolo = YOLO(yolo_model_path)
            logger.info(f"YOLO loaded from {yolo_model_path}")
        except Exception as e:
            logger.warning(f"Failed to load YOLO: {e}")
            self.yolo = None

        # Initialize CATI module
        self.cati = CATIDetector(self.config).to(self.device)

        if cati_weights_path and Path(cati_weights_path).exists():
            checkpoint = torch.load(cati_weights_path, map_location=self.device, weights_only=False)
            state = checkpoint.get("model_state_dict", checkpoint)
            self.cati.load_state_dict(state)
            logger.info(f"CATI weights loaded from {cati_weights_path}")

        self.cati.eval()

        self.ema = None
        if self.config.ema_decay > 0:
            self.ema = EMAModel(self.cati, decay=self.config.ema_decay)

        params = self.cati.count_parameters()
        logger.info(
            f"CATI overhead: {params['total_cati_overhead']:,} params "
            f"(context={params['context_encoder']:,}, "
            f"film_gen={params['film_generator']:,}, "
            f"film_layers={params['film_layers']:,})"
        )

    def _resolve_device(self, device: str) -> torch.device:
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
        return torch.device("cpu") if device == "auto" else torch.device(device)

    def _build_context_tensors(
        self,
        camera_id: int,
        weather: str,
        temperature: float,
        pm25: float,
        hour: float,
        resolution: tuple[int, int],
        camera_lat: float | None,
        camera_lon: float | None,
    ) -> dict:
        """Convert scalar context values to device tensors (batch size 1)."""
        weather_id = ContextEncoder.weather_to_id(weather)
        resolution_id = ContextEncoder.resolution_to_id(*resolution)
        ctx = {
            "weather_id": torch.tensor([weather_id], dtype=torch.long, device=self.device),
            "temperature": torch.tensor([temperature], dtype=torch.float32, device=self.device),
            "pm25": torch.tensor([pm25], dtype=torch.float32, device=self.device),
            "hour": torch.tensor([hour], dtype=torch.float32, device=self.device),
            "camera_id": torch.tensor([camera_id], dtype=torch.long, device=self.device),
            "resolution_id": torch.tensor([resolution_id], dtype=torch.long, device=self.device),
        }
        if camera_lat is not None and camera_lon is not None:
            ctx["camera_lat"] = torch.tensor([camera_lat], dtype=torch.float32, device=self.device)
            ctx["camera_lon"] = torch.tensor([camera_lon], dtype=torch.float32, device=self.device)
        return ctx

    def register_film_hooks(
        self,
        model_variant: str = "yolo11s",
    ) -> list:
        """Register forward hooks that apply FiLM conditioning during YOLO forward pass.

        Hooks intercept P3/P4/P5 feature maps immediately after each backbone
        layer produces them, apply FiLM modulation in-place, and return the
        conditioned tensor so the rest of the network sees modulated features.

        The CATI context must be set via `self._active_context` before calling
        `yolo.predict()` — use `predict()` which handles this automatically.

        Args:
            model_variant: YOLO variant key for layer index lookup.

        Returns:
            List of hook handles (call handle.remove() to deregister).
        """
        if self.yolo is None:
            raise RuntimeError("YOLO model not loaded")

        layer_indices = self.HOOK_LAYER_NAMES.get(model_variant, [4, 6, 9])
        backbone = self.yolo.model.model
        handles = []

        # Pre-compute FiLM params once per image and cache them
        self._film_params_cache: list[tuple[torch.Tensor, torch.Tensor]] | None = None

        for stage_idx, layer_idx in enumerate(layer_indices):
            def _make_hook(s_idx: int):
                def hook(
                    _module: nn.Module,
                    _input: tuple,
                    output: torch.Tensor,
                ) -> torch.Tensor:
                    ctx = getattr(self, "_active_context", None)
                    if ctx is None:
                        return output

                    # Extract feature tensor (some layers return tuples)
                    feat = output[0] if isinstance(output, tuple | list) else output
                    if not isinstance(feat, torch.Tensor):
                        return output

                    # Move feature to CATI device for conditioning
                    feat = feat.to(self.device)

                    # Compute FiLM params on first stage, reuse for rest
                    if s_idx == 0 or self._film_params_cache is None:
                        with torch.no_grad():
                            context = self.cati.encode_context(**ctx)
                            self._film_params_cache = self.cati.get_film_params(context)
                        self._active_context_vec = context

                    gamma, beta = self._film_params_cache[s_idx]
                    with torch.no_grad():
                        modulated = self.cati.film_layers[s_idx](
                            feat, gamma, beta, self._active_context_vec
                        )

                    return modulated

                return hook

            handle = backbone[layer_idx].register_forward_hook(_make_hook(stage_idx))
            handles.append(handle)
            logger.debug(f"FiLM hook registered on backbone layer {layer_idx} (P{stage_idx + 3})")

        logger.info(f"Registered {len(handles)} FiLM hooks on {model_variant} backbone")
        return handles

    def predict(
        self,
        image_path: str,
        camera_id: int = 0,
        weather: str = "clear",
        temperature: float = 28.0,
        pm25: float = 15.0,
        hour: float = 12.0,
        resolution: tuple[int, int] = (1920, 1080),
        camera_lat: float | None = None,
        camera_lon: float | None = None,
        use_film: bool = True,
    ) -> dict:
        """Run CATI-enhanced detection on a single image.

        When use_film=True (default), FiLM conditioning is applied to the YOLO
        backbone via forward hooks before the detection head runs.

        Args:
            image_path: Path to input image.
            camera_id: LTA camera index.
            weather: Weather condition string (see WEATHER_CONDITIONS).
            temperature: Air temperature in Celsius.
            pm25: PM2.5 concentration in ug/m3.
            hour: Hour of day [0, 24).
            resolution: (width, height) of the source camera.
            camera_lat: Optional GPS latitude.
            camera_lon: Optional GPS longitude.
            use_film: Apply FiLM conditioning (requires trained CATI weights).

        Returns:
            Detection results dict with boxes, context, and metadata.
        """
        if self.yolo is None:
            raise RuntimeError("YOLO model not loaded")

        weather_id = ContextEncoder.weather_to_id(weather)
        resolution_id = ContextEncoder.resolution_to_id(*resolution)

        handles: list = []
        if use_film:
            self._active_context = self._build_context_tensors(
                camera_id, weather, temperature, pm25, hour,
                resolution, camera_lat, camera_lon,
            )
            self._film_params_cache = None
            handles = self.register_film_hooks()

        try:
            self.cati.eval()
            results = self.yolo.predict(
                image_path,
                conf=self.config.conf_threshold,
                iou=self.config.iou_threshold,
                imgsz=self.config.img_size,
                verbose=False,
            )
        finally:
            for h in handles:
                h.remove()
            self._active_context = None
            self._film_params_cache = None

        result = results[0]
        boxes = result.boxes

        detections = []
        if boxes is not None:
            for box in boxes:
                detections.append(
                    {
                        "bbox": box.xyxy[0].tolist(),
                        "confidence": float(box.conf[0]),
                        "class_id": int(box.cls[0]),
                    }
                )

        return {
            "camera_id": camera_id,
            "num_detections": len(detections),
            "detections": detections,
            "film_conditioning": use_film,
            "context": {
                "weather": weather,
                "weather_id": weather_id,
                "temperature": temperature,
                "pm25": pm25,
                "hour": hour,
                "resolution": resolution,
                "resolution_id": resolution_id,
                "camera_lat": camera_lat,
                "camera_lon": camera_lon,
            },
            "inference_device": str(self.device),
            "cati_params": self.cati.count_parameters(),
        }
