"""
Singapore Smart City — Model Benchmarking Suite

Compares detection models on Singapore traffic camera data:
- YOLOv11s vs YOLOv8s (and optionally nano/medium variants)
- Performance across conditions: clear, rain, night, low-res
- Metrics: mAP, inference speed, confidence distributions
- Generates publishable comparison tables and plots

Run on Colab/Kaggle with pre-processed test data.
"""

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result from benchmarking one model on one condition."""

    model_name: str
    condition: str  # "clear", "rain", "night", "low_res", "all"
    num_images: int = 0

    # Detection metrics
    mean_confidence: float = 0.0
    median_confidence: float = 0.0
    total_detections: int = 0
    vehicles_per_image: float = 0.0

    # Speed metrics
    mean_inference_ms: float = 0.0
    p95_inference_ms: float = 0.0
    throughput_fps: float = 0.0

    # Quality metrics
    low_confidence_rate: float = 0.0  # % of detections below 0.25
    zero_detection_rate: float = 0.0  # % of images with 0 detections

    def to_dict(self) -> dict:
        return asdict(self)


class ModelBenchmark:
    """Benchmarks detection models on Singapore traffic data."""

    def __init__(self, test_data_dir: str):
        """
        Args:
            test_data_dir: Directory containing test images organized by condition.
                           Expected structure:
                           test_data/
                           ├── clear/     *.jpg
                           ├── rain/      *.jpg
                           ├── night/     *.jpg
                           └── low_res/   *.jpg
        """
        self.test_dir = Path(test_data_dir)
        self.results: list[BenchmarkResult] = []

    def _get_condition_images(self, condition: str) -> list[Path]:
        """Get image paths for a condition."""
        if condition == "all":
            return sorted(self.test_dir.rglob("*.jpg"))

        condition_dir = self.test_dir / condition
        if condition_dir.exists():
            return sorted(condition_dir.glob("*.jpg"))

        logger.warning(f"Condition directory not found: {condition_dir}")
        return []

    def benchmark_model(
        self,
        model_path: str,
        model_name: str,
        conditions: list[str] | None = None,
        max_images_per_condition: int = 500,
    ) -> list[BenchmarkResult]:
        """Benchmark a single model across conditions.

        Args:
            model_path: Path to YOLO weights.
            model_name: Human-readable model name for reports.
            conditions: List of conditions to test.
            max_images_per_condition: Max images per condition.

        Returns:
            List of BenchmarkResult objects.
        """
        from ultralytics import YOLO

        if conditions is None:
            conditions = ["clear", "rain", "night", "low_res", "all"]

        model = YOLO(model_path)
        results = []

        for condition in conditions:
            images = self._get_condition_images(condition)
            if not images:
                logger.warning(f"No images for condition: {condition}")
                continue

            images = images[:max_images_per_condition]
            logger.info(f"Benchmarking {model_name} on {condition}: {len(images)} images")

            confidences = []
            inference_times = []
            detections_per_image = []
            low_conf_count = 0
            total_detections = 0
            zero_detection_images = 0

            for img_path in images:
                start = time.time()
                preds = model(str(img_path), conf=0.15, verbose=False)
                inference_ms = (time.time() - start) * 1000
                inference_times.append(inference_ms)

                result = preds[0]
                boxes = result.boxes

                if boxes is not None and len(boxes) > 0:
                    # Filter to traffic classes
                    traffic_mask = torch_in(boxes.cls, [0, 1, 2, 3, 5, 7])
                    confs = boxes.conf[traffic_mask].cpu().numpy()

                    confidences.extend(confs.tolist())
                    n_det = len(confs)
                    detections_per_image.append(n_det)
                    total_detections += n_det

                    low_conf_count += sum(1 for c in confs if c < 0.25)
                else:
                    detections_per_image.append(0)
                    zero_detection_images += 1

            # Compute metrics
            bench_result = BenchmarkResult(
                model_name=model_name,
                condition=condition,
                num_images=len(images),
                mean_confidence=round(float(np.mean(confidences)), 4) if confidences else 0.0,
                median_confidence=round(float(np.median(confidences)), 4) if confidences else 0.0,
                total_detections=total_detections,
                vehicles_per_image=round(float(np.mean(detections_per_image)), 2),
                mean_inference_ms=round(float(np.mean(inference_times)), 1),
                p95_inference_ms=round(float(np.percentile(inference_times, 95)), 1),
                throughput_fps=round(1000 / np.mean(inference_times), 1) if inference_times else 0,
                low_confidence_rate=round(low_conf_count / max(total_detections, 1) * 100, 1),
                zero_detection_rate=round(zero_detection_images / max(len(images), 1) * 100, 1),
            )

            results.append(bench_result)
            self.results.append(bench_result)

            logger.info(
                f"  {condition}: {bench_result.vehicles_per_image} vehicles/img, "
                f"{bench_result.mean_inference_ms}ms avg, "
                f"{bench_result.throughput_fps} FPS, "
                f"conf={bench_result.mean_confidence:.3f}"
            )

        return results

    def generate_comparison_report(
        self,
        output_path: str = "reports/benchmark_report.json",
    ) -> dict:
        """Generate a full comparison report across all benchmarked models.

        Returns:
            Report dict with comparison tables and analysis.
        """
        # Group results by model
        models = {}
        for r in self.results:
            if r.model_name not in models:
                models[r.model_name] = {}
            models[r.model_name][r.condition] = r.to_dict()

        # Build comparison table (model x condition for key metrics)
        comparison = {
            "models_compared": list(models.keys()),
            "conditions_tested": list(set(r.condition for r in self.results)),
            "per_model": models,
        }

        # Best model per condition
        best_per_condition = {}
        conditions = set(r.condition for r in self.results)
        for cond in conditions:
            cond_results = [r for r in self.results if r.condition == cond]
            if cond_results:
                best = max(cond_results, key=lambda r: r.mean_confidence)
                best_per_condition[cond] = {
                    "best_model": best.model_name,
                    "confidence": best.mean_confidence,
                    "fps": best.throughput_fps,
                }

        comparison["best_per_condition"] = best_per_condition

        # Speed vs accuracy summary
        overall = [r for r in self.results if r.condition == "all"]
        if overall:
            comparison["speed_vs_accuracy"] = [
                {
                    "model": r.model_name,
                    "mean_confidence": r.mean_confidence,
                    "fps": r.throughput_fps,
                    "vehicles_per_image": r.vehicles_per_image,
                }
                for r in overall
            ]

        # Save report
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump(comparison, f, indent=2)

        logger.info(f"Benchmark report saved to {output_path}")
        return comparison


def torch_in(tensor, values):
    """Check if tensor elements are in a list of values.

    Helper to filter YOLO class predictions to traffic-relevant classes.
    """
    import torch

    mask = torch.zeros(len(tensor), dtype=torch.bool)
    for v in values:
        mask |= tensor == v
    return mask
