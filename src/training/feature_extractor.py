"""
CATI Feature Extractor — Extract & Cache YOLO Backbone Features

Bridges the gap between raw Singapore traffic images (collected to Google Drive)
and the CATI training pipeline. Extracts P3/P4/P5 feature maps from a frozen
YOLO backbone and saves them alongside environmental metadata for offline
FiLM training.

This enables Phase 1 training (context modules only) without loading the
full YOLO model into GPU memory during every training step — a critical
optimization for T4/Colab training where VRAM is limited.

Features are saved as FP16 tensors (.pt) alongside metadata (.json).
FP16 cuts storage from ~5.6MB/image to ~2.8MB/image (still ~28GB for 10K images,
so limit extraction to what fits in your Drive quota).

Data flow:
    Drive raw images → YOLO backbone (hooks) → .pt (FP16 features) + .json metadata
                                                ↓
                                      CATIDataset → CATITrainer

Usage:
    python -m src.training.feature_extractor \\
        --raw-dir data/raw \\
        --output-dir data/features \\
        --model yolo11s.pt \\
        --batch-size 16

    Or from Colab:
        from src.training.feature_extractor import FeatureExtractor
        extractor = FeatureExtractor("yolo11s.pt")
        extractor.extract_all("/content/drive/MyDrive/sg_smart_city/data/raw",
                              "/content/drive/MyDrive/sg_smart_city/data/features",
                              max_samples=5000)
"""

import json
import logging
from pathlib import Path
from typing import ClassVar

import click
import torch

logger = logging.getLogger(__name__)


class YOLOFeatureExtractor:
    """Extracts intermediate backbone features from YOLOv11 via forward hooks.

    Registers forward hooks on the backbone's P3/P4/P5 output layers to
    capture feature maps in-place without modifying the YOLO codebase.

    For YOLOv11s the backbone (model.model) is an nn.Sequential. Layers
    [4, 6, 9] correspond to the P3, P4, P5 output stages. If you switch to
    yolo11m/l the channel dims change but the layer indices stay the same.
    Verify with: `for i, m in enumerate(yolo.model.model): print(i, m.__class__.__name__)`

    Args:
        model_path: Path to YOLOv11 weights.
        device: Compute device.
        img_size: Input image size for YOLO preprocessing.
    """

    # Layer indices for backbone feature extraction (P3, P4, P5)
    # Verified for YOLOv11s — check CATI_HANDOVER.md if switching variants.
    BACKBONE_LAYERS: ClassVar[list[int]] = [4, 6, 9]

    def __init__(
        self,
        model_path: str = "yolo11s.pt",
        device: str = "auto",
        img_size: int = 640,
    ):
        self.device = self._resolve_device(device)
        self.img_size = img_size
        self._captured: dict[int, torch.Tensor] = {}
        self._hooks: list = []

        # Load YOLO
        from ultralytics import YOLO

        self.yolo = YOLO(model_path)
        self.yolo.model.eval()
        logger.info(f"YOLO loaded from {model_path} on {self.device}")

        # Register hooks immediately after loading
        self._register_backbone_hooks()

    def _resolve_device(self, device: str) -> torch.device:
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
        return torch.device("cpu") if device == "auto" else torch.device(device)

    def _register_backbone_hooks(self):
        """Attach forward hooks to P3/P4/P5 backbone layers."""
        try:
            backbone = self.yolo.model.model  # nn.Sequential of all YOLO layers
        except AttributeError as e:
            raise RuntimeError(
                "Cannot access yolo.model.model — ultralytics API may have changed. "
                "Try: list(yolo.model.named_modules()) to find the right attribute."
            ) from e

        for layer_idx in self.BACKBONE_LAYERS:
            def _make_hook(idx: int):
                def hook(module, _input, output):
                    # Some YOLO layers return tuples; we want the tensor output.
                    feat = output[0] if isinstance(output, (tuple, list)) else output
                    if isinstance(feat, torch.Tensor):
                        self._captured[idx] = feat.detach().cpu()

                return hook

            handle = backbone[layer_idx].register_forward_hook(_make_hook(layer_idx))
            self._hooks.append(handle)

        logger.info(
            f"Registered forward hooks on backbone layers {self.BACKBONE_LAYERS} "
            f"(P3, P4, P5)"
        )

    def remove_hooks(self):
        """Deregister all forward hooks (call when done extracting)."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def verify_hooks(self, img_size: int = 640) -> dict[int, tuple]:
        """Run a dummy forward pass and print captured feature shapes.

        Returns a dict of {layer_idx: shape} so you can confirm hook placement.
        """
        dummy = torch.zeros(1, 3, img_size, img_size)
        self._captured.clear()
        self.yolo.model(dummy)
        shapes = {idx: tuple(t.shape) for idx, t in self._captured.items()}
        for idx, shape in shapes.items():
            logger.info(f"  Layer {idx} (P{self.BACKBONE_LAYERS.index(idx) + 3}): {shape}")
        return shapes

    def extract_features(self, image_path: str) -> tuple[list[torch.Tensor], list[dict]]:
        """Extract P3/P4/P5 features and auto-label detections for one image.

        Args:
            image_path: Path to a single JPEG/PNG image.

        Returns:
            features: [P3_tensor(1,C,H,W), P4_tensor(1,C,H,W), P5_tensor(1,C,H,W)]
                      Each tensor is on CPU, FP32. The caller converts to FP16 for storage.
            detections: Auto-labeled bounding boxes from YOLO (pseudo-labels).

        Raises:
            RuntimeError: If hooks didn't fire (layer index mismatch).
        """
        self._captured.clear()

        results = self.yolo.predict(
            image_path,
            imgsz=self.img_size,
            verbose=False,
            save=False,
        )
        result = results[0]

        # Validate that hooks fired
        missing = [idx for idx in self.BACKBONE_LAYERS if idx not in self._captured]
        if missing:
            raise RuntimeError(
                f"Forward hooks did not capture features for layers {missing}. "
                f"Verify BACKBONE_LAYERS {self.BACKBONE_LAYERS} against your YOLO variant. "
                f"Run verify_hooks() to debug."
            )

        # Collect in P3→P4→P5 order
        features = [self._captured[idx] for idx in self.BACKBONE_LAYERS]

        # Auto-label detections (pseudo-labels for Phase 2)
        boxes = result.boxes
        detections = []
        if boxes is not None:
            for box in boxes:
                detections.append(
                    {
                        "bbox": box.xyxy[0].tolist(),
                        "confidence": float(box.conf[0]),
                        "class_id": int(box.cls[0]),
                        "bbox_norm": box.xywhn[0].tolist(),
                    }
                )

        return features, detections


class FeatureExtractor:
    """Full pipeline: raw images → cached FP16 feature tensors + metadata JSON.

    Each processed image produces two files in the output split directory:
        cam{id}_{timestamp}.json  — metadata + detections
        cam{id}_{timestamp}.pt   — FP16 backbone features [P3, P4, P5]

    The .pt file contains a list of 3 tensors (FP16, CPU).
    CATIDataset loads both files. If the .pt is missing (e.g., only metadata
    was written), the trainer falls back to synthetic features so training
    can proceed on partial data.

    Storage estimate (FP16):
        P3 (128, 80, 80) + P4 (256, 40, 40) + P5 (512, 20, 20) ≈ 2.8 MB/image
        → 1,000 images ≈ 2.8 GB on Drive
        → Set max_samples accordingly for your Drive quota.

    Args:
        model_path: Path to YOLOv11 weights.
        device: Compute device.
        img_size: Input image size.
    """

    def __init__(
        self,
        model_path: str = "yolo11s.pt",
        device: str = "auto",
        img_size: int = 640,
    ):
        self.extractor = YOLOFeatureExtractor(model_path, device, img_size)
        self.device = self.extractor.device

    def verify(self):
        """Sanity-check hook placement before running full extraction."""
        logger.info("Verifying hook placement with dummy forward pass...")
        shapes = self.extractor.verify_hooks()
        expected_num = len(YOLOFeatureExtractor.BACKBONE_LAYERS)
        if len(shapes) != expected_num:
            raise RuntimeError(
                f"Expected {expected_num} feature stages, got {len(shapes)}. "
                "Check BACKBONE_LAYERS in YOLOFeatureExtractor."
            )
        logger.info("Hook verification passed.")
        return shapes

    def extract_all(
        self,
        raw_dir: str,
        output_dir: str,
        max_samples: int | None = None,
    ) -> dict:
        """Extract features from all images in a directory tree.

        Expected raw_dir structure (from Singapore data collector):
            raw_dir/
              2026-03-09/
                1001/
                  metadata.jsonl
                  20260309_080100.jpg
                  20260309_080200.jpg
                1002/
                  ...

        Output structure:
            output_dir/
              train/
                cam1001_20260309_080100.json  (metadata + detections)
                cam1001_20260309_080100.pt    (FP16 backbone features)
              val/
                ...
              test/
                ...

        Returns:
            Summary statistics dict.
        """
        raw_path = Path(raw_dir)
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # Gather all image+metadata pairs
        pairs = self._gather_pairs(raw_path)
        if not pairs:
            logger.warning(f"No image+metadata pairs found in {raw_dir}")
            return {"train": 0, "val": 0, "test": 0}

        if max_samples:
            pairs = pairs[:max_samples]

        logger.info(f"Found {len(pairs)} image+metadata pairs in {raw_dir}")

        # Temporal sort for proper train/val/test split (avoids data leakage)
        pairs.sort(key=lambda p: p.get("timestamp", ""))

        # Temporal split: 70/15/15
        n = len(pairs)
        train_end = int(n * 0.70)
        val_end = int(n * 0.85)

        splits = {
            "train": pairs[:train_end],
            "val": pairs[train_end:val_end],
            "test": pairs[val_end:],
        }

        stats: dict[str, int] = {}
        for split_name, split_pairs in splits.items():
            split_dir = out_path / split_name
            split_dir.mkdir(parents=True, exist_ok=True)

            count = 0
            for pair in split_pairs:
                try:
                    self._process_pair(pair, split_dir)
                    count += 1
                    if count % 100 == 0:
                        logger.info(f"  [{split_name}] {count}/{len(split_pairs)} processed")
                except Exception as e:
                    logger.warning(f"Failed to process {pair.get('absolute_image_path', '?')}: {e}")

            stats[split_name] = count
            logger.info(f"  {split_name}: {count} samples written")

        logger.info(f"Feature extraction complete: {stats}")
        return stats

    def _gather_pairs(self, raw_path: Path) -> list[dict]:
        """Gather all image + metadata pairs from the raw data directory."""
        pairs = []

        for jsonl_file in raw_path.rglob("*.jsonl"):
            with open(jsonl_file) as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    # Find corresponding image file
                    img_name = Path(record.get("image_path", "")).name
                    img_path = jsonl_file.parent / img_name

                    if img_path.exists() and img_path.stat().st_size > 0:
                        record["absolute_image_path"] = str(img_path)
                        pairs.append(record)

        return pairs

    def _process_pair(self, pair: dict, output_dir: Path):
        """Process a single image+metadata pair.

        Saves:
            {name}.json — metadata enriched with pseudo-label detections
            {name}.pt   — list of 3 FP16 feature tensors [P3, P4, P5]
        """
        image_path = pair["absolute_image_path"]
        camera_id = pair.get("camera_id", "unknown")
        timestamp = (
            pair.get("timestamp", "unknown").replace(":", "").replace("-", "").replace("T", "_")
        )
        name = f"cam{camera_id}_{timestamp}"

        # Extract features and auto-label detections via hooked YOLO forward pass
        features, detections = self.extractor.extract_features(image_path)

        # Save FP16 feature tensors (saves ~50% storage vs FP32)
        feat_path = output_dir / f"{name}.pt"
        torch.save([f.half() for f in features], str(feat_path))

        # Save enriched metadata JSON
        meta = {
            "camera_id": pair.get("camera_id"),
            "camera_idx": pair.get("camera_idx", 0),
            "timestamp": pair.get("timestamp"),
            "weather_condition": pair.get("weather_main", "unknown"),
            "temperature_celsius": pair.get("temperature_celsius", 28.0),
            "pm25_reading": pair.get("pm25_reading", 15.0),
            "hour": pair.get("hour", 12.0),
            "image_width": pair.get("image_width", 1920),
            "image_height": pair.get("image_height", 1080),
            "camera_latitude": pair.get("latitude"),
            "camera_longitude": pair.get("longitude"),
            "num_detections": len(detections),
            "detections": detections,
            "source_image": image_path,
            "feature_file": f"{name}.pt",
            "feature_stages": len(features),
            "feature_shapes": [list(f.shape) for f in features],
        }

        with open(output_dir / f"{name}.json", "w") as f:
            json.dump(meta, f, indent=2)

    def cleanup(self):
        """Remove forward hooks after extraction is complete."""
        self.extractor.remove_hooks()


@click.command()
@click.option("--raw-dir", required=True, help="Path to raw data directory")
@click.option("--output-dir", default="data/features", help="Output directory for features")
@click.option("--model", default="yolo11s.pt", help="YOLO model path")
@click.option("--max-samples", type=int, default=None, help="Max samples to process")
@click.option("--img-size", type=int, default=640, help="Image input size")
@click.option("--verify/--no-verify", default=True, help="Run hook verification before extraction")
def main(raw_dir, output_dir, model, max_samples, img_size, verify):
    """Extract YOLO backbone features for CATI training."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    extractor = FeatureExtractor(model_path=model, img_size=img_size)

    if verify:
        extractor.verify()

    stats = extractor.extract_all(raw_dir, output_dir, max_samples)
    extractor.cleanup()
    logger.info(f"Done: {stats}")


if __name__ == "__main__":
    main()
