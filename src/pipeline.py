"""
Singapore Smart City — Pipeline Orchestrator

Chains all modules into a single end-to-end pipeline:
    Collect → Detect → Track → Analyze → Predict → Serve

Can be run as:
  - Full pipeline: python -m src.pipeline --mode full
  - Single stage: python -m src.pipeline --mode detect --input data/raw/2026-03-08

Each stage reads from and writes to well-defined directories,
enabling resumable, modular execution.

Designed for Colab/Azure execution.
"""

import json
import logging
import time
from pathlib import Path

import click

logger = logging.getLogger(__name__)


# =============================================================================
# Pipeline Configuration
# =============================================================================

DEFAULT_DIRS = {
    "raw": "data/raw",
    "detections": "data/processed/detections",
    "tracking": "data/processed/tracking",
    "analytics": "data/processed/analytics",
    "predictions": "data/processed/predictions",
    "models": "models",
    "reports": "reports",
    "dataset": "data/dataset",
}


class PipelineStage:
    """Base class for pipeline stages with standard I/O patterns."""

    def __init__(self, name: str, dirs: dict | None = None):
        self.name = name
        self.dirs = dirs or DEFAULT_DIRS
        self.start_time = None

    def _log_start(self, **kwargs):
        self.start_time = time.time()
        logger.info(f"\n{'=' * 60}")
        logger.info(f"  STAGE: {self.name}")
        for k, v in kwargs.items():
            logger.info(f"  {k}: {v}")
        logger.info(f"{'=' * 60}\n")

    def _log_end(self, results: dict):
        duration = time.time() - self.start_time
        logger.info(f"\n{'=' * 60}")
        logger.info(f"  STAGE COMPLETE: {self.name} ({duration:.1f}s)")
        for k, v in results.items():
            logger.info(f"  {k}: {v}")
        logger.info(f"{'=' * 60}\n")


# =============================================================================
# Stage 1: Detection
# =============================================================================


class DetectionStage(PipelineStage):
    """Run YOLOv11 detection on all collected images."""

    def __init__(self, dirs: dict | None = None):
        super().__init__("Detection", dirs)

    def run(
        self,
        input_dir: str | None = None,
        model_path: str = "models/yolo11s_traffic.pt",
        confidence: float = 0.25,
        max_images_per_camera: int | None = None,
    ) -> dict:
        """Run detection on collected data.

        Args:
            input_dir: Raw data directory (or use default).
            model_path: Path to YOLO weights.
            confidence: Detection confidence threshold.
            max_images_per_camera: Limit images per camera (for testing).

        Returns:
            Dict with detection summary.
        """
        from src.detection.detector import TrafficDetector

        raw_dir = Path(input_dir or self.dirs["raw"])
        output_dir = Path(self.dirs["detections"])
        output_dir.mkdir(parents=True, exist_ok=True)

        self._log_start(
            input=str(raw_dir),
            model=model_path,
            confidence=confidence,
        )

        detector = TrafficDetector(
            model_path=model_path,
            confidence_threshold=confidence,
        )

        # Find all camera directories
        camera_dirs = sorted(
            [d for d in raw_dir.rglob("*") if d.is_dir() and list(d.glob("*.jpg"))]
        )

        total_images = 0
        total_detections = 0

        for cam_dir in camera_dirs:
            camera_id = cam_dir.name
            results = detector.detect_batch(
                str(cam_dir),
                camera_id=camera_id,
                output_jsonl=str(output_dir / f"{camera_id}.jsonl"),
                max_images=max_images_per_camera,
            )
            total_images += len(results)
            total_detections += sum(r.num_detections for r in results)

        summary = {
            "cameras_processed": len(camera_dirs),
            "total_images": total_images,
            "total_detections": total_detections,
        }
        self._log_end(summary)
        return summary


# =============================================================================
# Stage 2: Tracking
# =============================================================================


class TrackingStage(PipelineStage):
    """Run BoT-SORT tracking on camera sequences."""

    def __init__(self, dirs: dict | None = None):
        super().__init__("Tracking", dirs)

    def run(
        self,
        input_dir: str | None = None,
        model_path: str = "models/yolo11s_traffic.pt",
        max_frames: int | None = None,
    ) -> dict:
        from src.tracking.tracker import VehicleTracker, estimate_congestion_score

        raw_dir = Path(input_dir or self.dirs["raw"])
        output_dir = Path(self.dirs["tracking"])
        output_dir.mkdir(parents=True, exist_ok=True)

        self._log_start(input=str(raw_dir), model=model_path)

        tracker = VehicleTracker(detector_model=model_path)

        camera_dirs = sorted(
            [d for d in raw_dir.rglob("*") if d.is_dir() and list(d.glob("*.jpg"))]
        )

        all_congestion = {}
        total_vehicles = 0

        for cam_dir in camera_dirs:
            camera_id = cam_dir.name
            result = tracker.track_image_sequence(
                str(cam_dir),
                camera_id=camera_id,
                output_path=str(output_dir / f"{camera_id}.json"),
                max_frames=max_frames,
            )
            congestion = estimate_congestion_score(result)
            all_congestion[camera_id] = congestion
            total_vehicles += result.total_unique_vehicles

        # Save congestion scores
        with open(output_dir / "congestion_scores.json", "w") as f:
            json.dump(all_congestion, f, indent=2)

        summary = {
            "cameras_tracked": len(camera_dirs),
            "total_unique_vehicles": total_vehicles,
        }
        self._log_end(summary)
        return summary


# =============================================================================
# Stage 3: Analytics (Failure + Drift)
# =============================================================================


class AnalyticsStage(PipelineStage):
    """Run failure analysis and drift monitoring."""

    def __init__(self, dirs: dict | None = None):
        super().__init__("Analytics", dirs)

    def run(self, config: dict | None = None) -> dict:
        from src.analytics.drift_monitor import DriftMonitor
        from src.analytics.failure_analyzer import FailureAnalyzer

        det_dir = Path(self.dirs["detections"])
        output_dir = Path(self.dirs["analytics"])
        output_dir.mkdir(parents=True, exist_ok=True)

        self._log_start(input=str(det_dir))

        analyzer = FailureAnalyzer(config)
        drift_monitor = DriftMonitor(config)

        # Load all detection results grouped by camera
        all_results = {}
        for jsonl_file in det_dir.glob("*.jsonl"):
            camera_id = jsonl_file.stem
            results = []
            with open(jsonl_file) as f:
                for line in f:
                    try:
                        results.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            if results:
                all_results[camera_id] = results

        # Run failure analysis
        fleet_report = analyzer.generate_fleet_report(
            all_results,
            output_path=str(output_dir / "fleet_report.json"),
        )

        # Run drift detection (use first 50% as baseline, check rest)
        all_flat = [r for results in all_results.values() for r in results]
        if len(all_flat) > 20:
            midpoint = len(all_flat) // 2
            drift_monitor.set_baseline(all_flat[:midpoint])
            alerts = drift_monitor.check_drift(all_flat[midpoint:])

            with open(output_dir / "drift_alerts.json", "w") as f:
                json.dump([a.to_dict() for a in alerts], f, indent=2)
        else:
            alerts = []

        summary = {
            "cameras_analyzed": len(all_results),
            "total_frames": len(all_flat),
            "mean_reliability": fleet_report.get("fleet_summary", {}).get(
                "mean_reliability", "N/A"
            ),
            "drift_alerts": len(alerts),
        }
        self._log_end(summary)
        return summary


# =============================================================================
# Stage 4: Auto-Labeling (for Kaggle dataset)
# =============================================================================


class LabelingStage(PipelineStage):
    """Auto-label collected images using fine-tuned YOLO."""

    def __init__(self, dirs: dict | None = None):
        super().__init__("Auto-Labeling", dirs)

    def run(
        self,
        model_path: str = "models/yolo11s_traffic.pt",
        confidence: float = 0.3,
    ) -> dict:
        from src.detection.detector import generate_yolo_labels

        det_dir = Path(self.dirs["detections"])
        label_dir = Path(self.dirs["dataset"]) / "labels"
        label_dir.mkdir(parents=True, exist_ok=True)

        self._log_start(model=model_path, confidence=confidence)

        # Load detection results
        all_results = []
        for jsonl_file in det_dir.glob("*.jsonl"):
            with open(jsonl_file) as f:
                for line in f:
                    try:
                        from src.detection.detector import DetectionResult

                        data = json.loads(line)
                        result = DetectionResult(
                            **{
                                k: v
                                for k, v in data.items()
                                if k in DetectionResult.__dataclass_fields__
                            }
                        )
                        all_results.append(result)
                    except (json.JSONDecodeError, TypeError):
                        continue

        generate_yolo_labels(all_results, str(label_dir), confidence)

        summary = {"total_labels_generated": len(all_results)}
        self._log_end(summary)
        return summary


# =============================================================================
# Stage 5: Dataset Formatting
# =============================================================================


class DatasetStage(PipelineStage):
    """Format collected data into a Kaggle-ready dataset."""

    def __init__(self, dirs: dict | None = None):
        super().__init__("Dataset Formatting", dirs)

    def run(self, labels_dir: str | None = None) -> dict:
        from src.ingestion.dataset_formatter import DatasetFormatter

        self._log_start(input=self.dirs["raw"])

        formatter = DatasetFormatter(
            raw_data_dir=self.dirs["raw"],
            output_dir=self.dirs["dataset"],
        )

        labels = labels_dir or str(Path(self.dirs["dataset"]) / "labels")
        stats = formatter.format_dataset(labels_dir=labels)

        self._log_end(stats)
        return stats


# =============================================================================
# Full Pipeline Runner
# =============================================================================


class SmartCityPipeline:
    """Full end-to-end pipeline orchestrator."""

    def __init__(self, config: dict | None = None, dirs: dict | None = None):
        self.config = config or {}
        self.dirs = dirs or DEFAULT_DIRS
        self.results = {}

        # Ensure all directories exist
        for d in self.dirs.values():
            Path(d).mkdir(parents=True, exist_ok=True)

    def run_full(
        self,
        model_path: str = "models/yolo11s_traffic.pt",
        max_images_per_camera: int | None = None,
    ) -> dict:
        """Run the complete pipeline: detect → track → analyze → label → format.

        Args:
            model_path: Path to fine-tuned YOLO weights.
            max_images_per_camera: Limit for testing.

        Returns:
            Combined results from all stages.
        """
        pipeline_start = time.time()

        logger.info("\n" + "🇸🇬" * 30)
        logger.info("  SINGAPORE SMART CITY — FULL PIPELINE")
        logger.info("🇸🇬" * 30 + "\n")

        # Stage 1: Detection
        det_stage = DetectionStage(self.dirs)
        self.results["detection"] = det_stage.run(
            model_path=model_path,
            max_images_per_camera=max_images_per_camera,
        )

        # Stage 2: Tracking
        track_stage = TrackingStage(self.dirs)
        self.results["tracking"] = track_stage.run(model_path=model_path)

        # Stage 3: Analytics
        analytics_stage = AnalyticsStage(self.dirs)
        self.results["analytics"] = analytics_stage.run(self.config)

        # Stage 4: Auto-labeling
        label_stage = LabelingStage(self.dirs)
        self.results["labeling"] = label_stage.run(model_path=model_path)

        # Stage 5: Dataset formatting
        dataset_stage = DatasetStage(self.dirs)
        self.results["dataset"] = dataset_stage.run()

        total_time = time.time() - pipeline_start

        logger.info("\n" + "=" * 60)
        logger.info("  PIPELINE COMPLETE")
        logger.info(f"  Total time: {total_time:.1f}s ({total_time / 60:.1f} min)")
        logger.info("=" * 60)

        # Save combined results
        results_path = Path(self.dirs["reports"]) / "pipeline_results.json"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(self.results, f, indent=2, default=str)

        return self.results


# =============================================================================
# CLI Entry Point
# =============================================================================


@click.command()
@click.option(
    "--mode",
    default="full",
    type=click.Choice(["full", "detect", "track", "analyze", "label", "dataset"]),
    help="Pipeline mode",
)
@click.option("--input", "input_dir", default=None, help="Input directory override")
@click.option("--model", default="models/yolo11s_traffic.pt", help="YOLO model path")
@click.option("--max-images", default=None, type=int, help="Limit images per camera")
@click.option("--config", default="configs/collection_config.yaml", help="Config path")
def main(mode, input_dir, model, max_images, config):
    """Singapore Smart City — Pipeline Runner

    Run the full pipeline or individual stages.

    Examples:
        # Full pipeline
        python -m src.pipeline --mode full --model models/yolo11s_traffic.pt

        # Detection only
        python -m src.pipeline --mode detect --input data/raw/2026-03-08

        # Analytics only (uses detection outputs)
        python -m src.pipeline --mode analyze
    """
    import yaml

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Load config
    cfg = {}
    config_path = Path(config)
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

    pipeline = SmartCityPipeline(config=cfg)

    if mode == "full":
        pipeline.run_full(model_path=model, max_images_per_camera=max_images)
    elif mode == "detect":
        DetectionStage().run(
            input_dir=input_dir, model_path=model, max_images_per_camera=max_images
        )
    elif mode == "track":
        TrackingStage().run(input_dir=input_dir, model_path=model)
    elif mode == "analyze":
        AnalyticsStage().run(config=cfg)
    elif mode == "label":
        LabelingStage().run(model_path=model)
    elif mode == "dataset":
        DatasetStage().run()


if __name__ == "__main__":
    main()
