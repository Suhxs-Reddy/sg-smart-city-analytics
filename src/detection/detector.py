"""
Singapore Smart City — YOLOv11 Detection Pipeline

Wraps Ultralytics YOLOv11 for:
1. Inference on Singapore traffic camera images
2. Batch auto-labeling of collected datasets
3. Per-frame confidence and failure metadata extraction

Designed to run on Colab/Kaggle T4 GPU.
"""

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Traffic-relevant COCO class IDs
TRAFFIC_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    0: "person",
    1: "bicycle",
}


@dataclass
class DetectionResult:
    """Result from running detection on a single frame."""
    camera_id: str
    timestamp: str
    image_path: str
    image_width: int
    image_height: int

    # Detection stats
    num_detections: int = 0
    num_vehicles: int = 0
    num_persons: int = 0

    # Per-class counts
    class_counts: dict = field(default_factory=dict)

    # Confidence metrics
    mean_confidence: float = 0.0
    min_confidence: float = 0.0
    max_confidence: float = 0.0
    low_confidence_count: int = 0   # Detections below threshold

    # Image quality indicators
    mean_brightness: float = 0.0
    is_night: bool = False
    is_grayscale: bool = False

    # Failure flags (populated by FailureAnalyzer later)
    failure_flags: list = field(default_factory=list)

    # Raw detections (YOLO format: x_center, y_center, w, h, conf, class_id)
    detections: list = field(default_factory=list)

    # Processing metadata
    inference_time_ms: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class TrafficDetector:
    """YOLOv11-based vehicle detector for Singapore traffic cameras."""

    def __init__(
        self,
        model_path: str = "yolo11s.pt",
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        img_size: int = 640,
        device: str = "auto",
        low_confidence_floor: float = 0.15,
    ):
        """
        Args:
            model_path: Path to YOLO weights. Use 'yolo11s.pt' for pretrained
                        or path to fine-tuned weights.
            confidence_threshold: Minimum confidence for a detection to count.
            iou_threshold: NMS IoU threshold.
            img_size: Input image size for YOLO.
            device: 'auto', 'cuda', 'cpu', or device index.
            low_confidence_floor: Below this = flagged as low-confidence.
        """
        # Lazy import — only needed at runtime, not during testing
        from ultralytics import YOLO

        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.img_size = img_size
        self.device = device
        self.low_confidence_floor = low_confidence_floor

        # Night detection threshold (mean pixel intensity)
        self.night_brightness_threshold = 60

        logger.info(
            f"Detector initialized: model={model_path}, "
            f"conf={confidence_threshold}, img_size={img_size}"
        )

    def _compute_image_metrics(self, image_path: str) -> dict:
        """Compute image quality metrics for failure analysis."""
        try:
            img = Image.open(image_path).convert("RGB")
            arr = np.array(img)

            # Mean brightness (grayscale equivalent)
            gray = np.mean(arr, axis=2)
            mean_brightness = float(np.mean(gray))

            # Check if image is effectively grayscale
            # (low variance across color channels → likely grayscale/IR camera)
            channel_std = float(np.std(arr.std(axis=(0, 1))))
            is_grayscale = channel_std < 5.0

            return {
                "mean_brightness": round(mean_brightness, 1),
                "is_night": mean_brightness < self.night_brightness_threshold,
                "is_grayscale": is_grayscale,
                "width": img.width,
                "height": img.height,
            }
        except Exception as e:
            logger.warning(f"Failed to compute image metrics: {e}")
            return {
                "mean_brightness": 0.0,
                "is_night": False,
                "is_grayscale": False,
                "width": 0,
                "height": 0,
            }

    def detect(
        self,
        image_path: str,
        camera_id: str = "unknown",
        timestamp: str = "",
    ) -> DetectionResult:
        """Run detection on a single image.

        Args:
            image_path: Path to the image file.
            camera_id: Camera identifier for metadata.
            timestamp: ISO timestamp string.

        Returns:
            DetectionResult with all detection info and metadata.
        """
        # Compute image quality metrics
        img_metrics = self._compute_image_metrics(image_path)

        # Run YOLO inference
        start_time = time.time()
        results = self.model(
            image_path,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.img_size,
            device=self.device,
            verbose=False,
        )
        inference_time = (time.time() - start_time) * 1000  # ms

        # Parse results
        result = results[0]
        boxes = result.boxes

        detections = []
        class_counts = {}
        confidences = []
        low_conf_count = 0
        num_vehicles = 0
        num_persons = 0

        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy()

                # Convert to YOLO normalized format (x_center, y_center, w, h)
                x1, y1, x2, y2 = xyxy
                img_w = img_metrics["width"] or 1
                img_h = img_metrics["height"] or 1
                x_center = ((x1 + x2) / 2) / img_w
                y_center = ((y1 + y2) / 2) / img_h
                w = (x2 - x1) / img_w
                h = (y2 - y1) / img_h

                # Only track traffic-relevant classes
                class_name = TRAFFIC_CLASSES.get(cls_id)
                if class_name is None:
                    continue

                detections.append({
                    "class_id": cls_id,
                    "class_name": class_name,
                    "confidence": round(conf, 4),
                    "bbox_normalized": [
                        round(x_center, 6),
                        round(y_center, 6),
                        round(w, 6),
                        round(h, 6),
                    ],
                    "bbox_xyxy": [round(float(x), 1) for x in xyxy],
                })

                # Count classes
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
                confidences.append(conf)

                # Count vehicles vs persons
                if cls_id in (2, 3, 5, 7):  # car, motorcycle, bus, truck
                    num_vehicles += 1
                elif cls_id == 0:
                    num_persons += 1

                # Flag low confidence
                if conf < self.low_confidence_floor:
                    low_conf_count += 1

        return DetectionResult(
            camera_id=camera_id,
            timestamp=timestamp or datetime.now().isoformat(),
            image_path=str(image_path),
            image_width=img_metrics["width"],
            image_height=img_metrics["height"],
            num_detections=len(detections),
            num_vehicles=num_vehicles,
            num_persons=num_persons,
            class_counts=class_counts,
            mean_confidence=round(np.mean(confidences), 4) if confidences else 0.0,
            min_confidence=round(min(confidences), 4) if confidences else 0.0,
            max_confidence=round(max(confidences), 4) if confidences else 0.0,
            low_confidence_count=low_conf_count,
            mean_brightness=img_metrics["mean_brightness"],
            is_night=img_metrics["is_night"],
            is_grayscale=img_metrics["is_grayscale"],
            detections=detections,
            inference_time_ms=round(inference_time, 1),
        )

    def detect_batch(
        self,
        image_dir: str,
        camera_id: str = "unknown",
        output_jsonl: str | None = None,
        max_images: int | None = None,
    ) -> list[DetectionResult]:
        """Run detection on all images in a directory.

        Args:
            image_dir: Directory containing .jpg images.
            camera_id: Camera identifier.
            output_jsonl: If provided, write results to this JSONL file.
            max_images: Limit number of images processed.

        Returns:
            List of DetectionResult objects.
        """
        image_dir = Path(image_dir)
        image_files = sorted(image_dir.glob("*.jpg"))

        if max_images:
            image_files = image_files[:max_images]

        logger.info(f"Processing {len(image_files)} images from {image_dir}")

        results = []
        start_time = time.time()

        for i, img_path in enumerate(image_files):
            # Extract timestamp from filename (format: HH-MM-SS.jpg)
            timestamp_str = img_path.stem  # e.g. "15-00-00"

            result = self.detect(
                str(img_path),
                camera_id=camera_id,
                timestamp=timestamp_str,
            )
            results.append(result)

            if (i + 1) % 100 == 0:
                elapsed = time.time() - start_time
                fps = (i + 1) / elapsed
                logger.info(
                    f"Processed {i + 1}/{len(image_files)} images "
                    f"({fps:.1f} FPS)"
                )

        total_time = time.time() - start_time
        logger.info(
            f"Batch complete: {len(results)} images in {total_time:.1f}s "
            f"({len(results) / total_time:.1f} FPS avg)"
        )

        # Save results if output path provided
        if output_jsonl:
            output_path = Path(output_jsonl)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                for r in results:
                    f.write(json.dumps(r.to_dict()) + "\n")
            logger.info(f"Results saved to {output_jsonl}")

        return results


def generate_yolo_labels(
    detection_results: list[DetectionResult],
    output_dir: str,
    confidence_threshold: float = 0.3,
):
    """Generate YOLO-format label files from detection results.

    Creates .txt files alongside images with format:
        class_id x_center y_center width height

    This is the auto-labeling step for creating the Kaggle dataset.

    Args:
        detection_results: List of DetectionResult from detect_batch.
        output_dir: Directory to save .txt label files.
        confidence_threshold: Only include detections above this confidence.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Remap COCO classes to contiguous IDs for the traffic dataset
    class_remap = {
        2: 0,   # car → 0
        3: 1,   # motorcycle → 1
        5: 2,   # bus → 2
        7: 3,   # truck → 3
        0: 4,   # person → 4
        1: 5,   # bicycle → 5
    }

    total_labels = 0
    for result in detection_results:
        img_name = Path(result.image_path).stem
        label_path = output_dir / f"{img_name}.txt"

        lines = []
        for det in result.detections:
            if det["confidence"] >= confidence_threshold:
                new_cls = class_remap.get(det["class_id"])
                if new_cls is not None:
                    bbox = det["bbox_normalized"]
                    lines.append(
                        f"{new_cls} {bbox[0]:.6f} {bbox[1]:.6f} "
                        f"{bbox[2]:.6f} {bbox[3]:.6f}"
                    )

        with open(label_path, "w") as f:
            f.write("\n".join(lines))

        total_labels += len(lines)

    logger.info(
        f"Generated {total_labels} labels across "
        f"{len(detection_results)} images in {output_dir}"
    )
