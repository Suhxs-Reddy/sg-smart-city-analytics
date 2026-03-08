"""
Singapore Smart City — Detection Failure Analyzer

6-category failure taxonomy for traffic camera detection:
1. Low Confidence — detections below confidence floor
2. Weather Degradation — rain/fog causing low visibility
3. Low Resolution — cameras at 320×240 causing missed detections
4. Occlusion — overlapping vehicles hiding detections
5. Night Mode — low brightness causing feature loss
6. Camera Failure — zero detections when traffic expected

Generates per-camera reliability scorecards and failure reports.
"""

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Failure Categories
# =============================================================================

class FailureCategory:
    LOW_CONFIDENCE = "low_confidence"
    WEATHER_DEGRADATION = "weather_degradation"
    LOW_RESOLUTION = "low_resolution"
    OCCLUSION = "occlusion"
    NIGHT_MODE = "night_mode"
    CAMERA_FAILURE = "camera_failure"


@dataclass
class FailureReport:
    """Failure analysis for a single detection frame."""
    camera_id: str
    timestamp: str
    failure_flags: list = field(default_factory=list)
    failure_details: dict = field(default_factory=dict)
    reliability_score: float = 1.0  # 0 = unreliable, 1 = perfect

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CameraReliabilityCard:
    """Reliability scorecard for a single camera over a time period."""
    camera_id: str
    total_frames: int = 0
    resolution: str = ""

    # Failure counts per category
    failure_counts: dict = field(default_factory=dict)

    # Overall reliability
    overall_reliability: float = 1.0
    frames_with_failures: int = 0
    failure_rate: float = 0.0

    # Per-condition reliability
    reliability_by_weather: dict = field(default_factory=dict)
    reliability_by_time_of_day: dict = field(default_factory=dict)

    # Recommendations
    issues: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# =============================================================================
# Failure Analyzer
# =============================================================================

class FailureAnalyzer:
    """Analyzes detection results for failure patterns."""

    def __init__(self, config: dict = None):
        """
        Args:
            config: Failure threshold configuration. Uses defaults if None.
        """
        thresholds = (config or {}).get("failure_thresholds", {})

        self.low_confidence_floor = thresholds.get("low_confidence_floor", 0.15)
        self.low_confidence_ceiling = thresholds.get("low_confidence_ceiling", 0.25)
        self.occlusion_iou = thresholds.get("occlusion_iou", 0.7)
        self.night_brightness = thresholds.get("night_brightness", 60)
        self.min_expected_vehicles = thresholds.get("min_expected_vehicles", 1)

        # Weather conditions that typically cause degradation
        self.adverse_weather = {
            "Thundery Showers", "Heavy Thundery Showers",
            "Showers", "Heavy Showers", "Light Showers",
            "Heavy Rain", "Moderate Rain", "Light Rain",
        }

    def _compute_iou(self, box1: list, box2: list) -> float:
        """Compute IoU between two [x1, y1, x2, y2] boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def analyze_frame(self, detection_result: dict) -> FailureReport:
        """Analyze a single frame's detection result for failures.

        Args:
            detection_result: Dictionary from DetectionResult.to_dict()

        Returns:
            FailureReport with categorized failures.
        """
        failures = []
        details = {}

        camera_id = detection_result.get("camera_id", "unknown")
        timestamp = detection_result.get("timestamp", "")
        detections = detection_result.get("detections", [])
        mean_conf = detection_result.get("mean_confidence", 0)
        mean_brightness = detection_result.get("mean_brightness", 128)
        weather = detection_result.get("weather_condition", "unknown")
        img_width = detection_result.get("image_width", 1920)
        num_detections = detection_result.get("num_detections", 0)
        low_conf_count = detection_result.get("low_confidence_count", 0)

        # --- 1. Low Confidence ---
        if low_conf_count > 0 or (mean_conf > 0 and mean_conf < self.low_confidence_ceiling):
            failures.append(FailureCategory.LOW_CONFIDENCE)
            details["low_confidence"] = {
                "mean_confidence": mean_conf,
                "low_conf_detections": low_conf_count,
                "total_detections": num_detections,
            }

        # --- 2. Weather Degradation ---
        if weather in self.adverse_weather:
            # Weather is bad — check if confidence is also lower than usual
            failures.append(FailureCategory.WEATHER_DEGRADATION)
            details["weather_degradation"] = {
                "weather_condition": weather,
                "mean_confidence": mean_conf,
            }

        # --- 3. Low Resolution ---
        if img_width <= 320:
            failures.append(FailureCategory.LOW_RESOLUTION)
            details["low_resolution"] = {
                "image_width": img_width,
                "image_height": detection_result.get("image_height", 0),
            }

        # --- 4. Occlusion ---
        occlusion_count = 0
        for i, det_a in enumerate(detections):
            for j, det_b in enumerate(detections):
                if i >= j:
                    continue
                bbox_a = det_a.get("bbox_xyxy", [0, 0, 0, 0])
                bbox_b = det_b.get("bbox_xyxy", [0, 0, 0, 0])
                iou = self._compute_iou(bbox_a, bbox_b)
                if iou > self.occlusion_iou:
                    occlusion_count += 1

        if occlusion_count > 0:
            failures.append(FailureCategory.OCCLUSION)
            details["occlusion"] = {
                "occluded_pairs": occlusion_count,
                "iou_threshold": self.occlusion_iou,
            }

        # --- 5. Night Mode ---
        if mean_brightness < self.night_brightness:
            failures.append(FailureCategory.NIGHT_MODE)
            details["night_mode"] = {
                "mean_brightness": mean_brightness,
                "threshold": self.night_brightness,
            }

        # --- 6. Camera Failure ---
        # Zero detections + not night + not bad weather = likely camera issue
        if num_detections == 0:
            is_expected_empty = (
                mean_brightness < self.night_brightness or
                weather in self.adverse_weather
            )
            if not is_expected_empty:
                failures.append(FailureCategory.CAMERA_FAILURE)
                details["camera_failure"] = {
                    "reason": "zero_detections_in_normal_conditions",
                    "mean_brightness": mean_brightness,
                    "weather": weather,
                }

        # Compute reliability score
        # Each failure type reduces reliability
        penalty_map = {
            FailureCategory.LOW_CONFIDENCE: 0.15,
            FailureCategory.WEATHER_DEGRADATION: 0.10,
            FailureCategory.LOW_RESOLUTION: 0.20,
            FailureCategory.OCCLUSION: 0.10,
            FailureCategory.NIGHT_MODE: 0.15,
            FailureCategory.CAMERA_FAILURE: 0.50,
        }

        reliability = 1.0
        for f in failures:
            reliability -= penalty_map.get(f, 0.1)
        reliability = max(0.0, round(reliability, 3))

        return FailureReport(
            camera_id=camera_id,
            timestamp=timestamp,
            failure_flags=failures,
            failure_details=details,
            reliability_score=reliability,
        )

    def generate_camera_scorecard(
        self,
        detection_results: list[dict],
        camera_id: str,
    ) -> CameraReliabilityCard:
        """Generate a reliability scorecard for a camera over many frames.

        Args:
            detection_results: List of detection result dicts for this camera.
            camera_id: Camera identifier.

        Returns:
            CameraReliabilityCard with aggregated reliability metrics.
        """
        if not detection_results:
            return CameraReliabilityCard(camera_id=camera_id)

        # Analyze each frame
        reports = [self.analyze_frame(r) for r in detection_results]

        # Count failures by category
        failure_counts = defaultdict(int)
        frames_with_any_failure = 0
        reliability_scores = []

        # Group by weather
        weather_reliability = defaultdict(list)
        # Group by time of day (rough buckets)
        time_reliability = defaultdict(list)

        for report, det in zip(reports, detection_results):
            reliability_scores.append(report.reliability_score)

            if report.failure_flags:
                frames_with_any_failure += 1
                for flag in report.failure_flags:
                    failure_counts[flag] += 1

            # Weather grouping
            weather = det.get("weather_condition", "unknown")
            weather_reliability[weather].append(report.reliability_score)

            # Time grouping
            ts = det.get("timestamp", "")
            try:
                hour = int(ts.split("T")[1][:2]) if "T" in ts else -1
            except (IndexError, ValueError):
                hour = -1

            if 6 <= hour < 12:
                period = "morning"
            elif 12 <= hour < 18:
                period = "afternoon"
            elif 18 <= hour < 22:
                period = "evening"
            elif hour >= 0:
                period = "night"
            else:
                period = "unknown"
            time_reliability[period].append(report.reliability_score)

        # Resolution
        first = detection_results[0]
        resolution = f"{first.get('image_width', '?')}x{first.get('image_height', '?')}"

        # Identify top issues
        issues = []
        total = len(reports)
        for cat, count in sorted(failure_counts.items(), key=lambda x: -x[1]):
            rate = count / total * 100
            if rate > 10:
                issues.append(f"{cat}: {rate:.0f}% of frames ({count}/{total})")

        return CameraReliabilityCard(
            camera_id=camera_id,
            total_frames=total,
            resolution=resolution,
            failure_counts=dict(failure_counts),
            overall_reliability=round(float(np.mean(reliability_scores)), 3),
            frames_with_failures=frames_with_any_failure,
            failure_rate=round(frames_with_any_failure / total * 100, 1),
            reliability_by_weather={
                k: round(float(np.mean(v)), 3)
                for k, v in weather_reliability.items()
            },
            reliability_by_time_of_day={
                k: round(float(np.mean(v)), 3)
                for k, v in time_reliability.items()
            },
            issues=issues,
        )

    def generate_fleet_report(
        self,
        all_detection_results: dict[str, list[dict]],
        output_path: Optional[str] = None,
    ) -> dict:
        """Generate a report across all cameras.

        Args:
            all_detection_results: Dict of camera_id → list of detection results.
            output_path: If provided, save report to this JSON file.

        Returns:
            Fleet-level report with per-camera scorecards.
        """
        scorecards = {}
        for camera_id, results in all_detection_results.items():
            scorecards[camera_id] = self.generate_camera_scorecard(
                results, camera_id
            )

        # Fleet summary
        reliabilities = [sc.overall_reliability for sc in scorecards.values()]
        failure_rates = [sc.failure_rate for sc in scorecards.values()]

        # Worst cameras
        worst_cameras = sorted(
            scorecards.values(),
            key=lambda x: x.overall_reliability,
        )[:10]

        report = {
            "fleet_summary": {
                "total_cameras": len(scorecards),
                "mean_reliability": round(float(np.mean(reliabilities)), 3),
                "median_reliability": round(float(np.median(reliabilities)), 3),
                "worst_camera_reliability": round(float(min(reliabilities)), 3) if reliabilities else 0,
                "mean_failure_rate_pct": round(float(np.mean(failure_rates)), 1),
            },
            "worst_cameras": [
                {
                    "camera_id": sc.camera_id,
                    "reliability": sc.overall_reliability,
                    "resolution": sc.resolution,
                    "issues": sc.issues,
                }
                for sc in worst_cameras
            ],
            "per_camera_scorecards": {
                cam_id: sc.to_dict() for cam_id, sc in scorecards.items()
            },
        }

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(report, f, indent=2)
            logger.info(f"Fleet report saved to {output_path}")

        return report
