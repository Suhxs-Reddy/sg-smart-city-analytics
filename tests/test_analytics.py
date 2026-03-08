"""
Tests for the failure analyzer and drift monitor.

Covers:
- 6-category failure detection
- IoU computation
- Reliability scoring
- Camera scorecard generation
- PSI computation
- KS test
- Drift alert generation
"""

from collections import defaultdict

import numpy as np
import pytest

from src.analytics.failure_analyzer import (
    FailureAnalyzer,
    FailureCategory,
    FailureReport,
    CameraReliabilityCard,
)
from src.analytics.drift_monitor import (
    DriftMonitor,
    compute_psi,
    compute_ks_test,
    DriftAlert,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def analyzer():
    return FailureAnalyzer()


@pytest.fixture
def drift_monitor():
    return DriftMonitor()


@pytest.fixture
def normal_detection():
    """A normal, healthy detection result."""
    return {
        "camera_id": "1701",
        "timestamp": "2026-03-08T15:00:00+08:00",
        "num_detections": 8,
        "num_vehicles": 7,
        "mean_confidence": 0.75,
        "mean_brightness": 140,
        "image_width": 1920,
        "image_height": 1080,
        "weather_condition": "Fair",
        "low_confidence_count": 0,
        "detections": [
            {"class_id": 2, "confidence": 0.85, "bbox_xyxy": [100, 200, 250, 350]},
            {"class_id": 2, "confidence": 0.72, "bbox_xyxy": [400, 200, 550, 350]},
        ],
    }


@pytest.fixture
def night_detection():
    """A night-time detection with low brightness."""
    return {
        "camera_id": "2701",
        "timestamp": "2026-03-08T23:00:00+08:00",
        "num_detections": 2,
        "num_vehicles": 2,
        "mean_confidence": 0.38,
        "mean_brightness": 35,
        "image_width": 1920,
        "image_height": 1080,
        "weather_condition": "Fair",
        "low_confidence_count": 1,
        "detections": [],
    }


@pytest.fixture
def rain_detection():
    """Detection during heavy rain."""
    return {
        "camera_id": "1701",
        "timestamp": "2026-03-08T14:00:00+08:00",
        "num_detections": 3,
        "num_vehicles": 3,
        "mean_confidence": 0.55,
        "mean_brightness": 95,
        "image_width": 1920,
        "image_height": 1080,
        "weather_condition": "Heavy Thundery Showers",
        "low_confidence_count": 0,
        "detections": [],
    }


@pytest.fixture
def camera_failure_detection():
    """Zero detections in normal conditions — likely camera issue."""
    return {
        "camera_id": "3801",
        "timestamp": "2026-03-08T12:00:00+08:00",
        "num_detections": 0,
        "num_vehicles": 0,
        "mean_confidence": 0,
        "mean_brightness": 150,
        "image_width": 1920,
        "image_height": 1080,
        "weather_condition": "Fair",
        "low_confidence_count": 0,
        "detections": [],
    }


@pytest.fixture
def low_res_detection():
    """Detection from a 320×240 camera."""
    return {
        "camera_id": "1001",
        "timestamp": "2026-03-08T10:00:00+08:00",
        "num_detections": 2,
        "num_vehicles": 2,
        "mean_confidence": 0.52,
        "mean_brightness": 130,
        "image_width": 320,
        "image_height": 240,
        "weather_condition": "Fair",
        "low_confidence_count": 0,
        "detections": [],
    }


# =============================================================================
# Failure Analyzer Tests
# =============================================================================

class TestFailureCategories:
    def test_normal_frame_no_failures(self, analyzer, normal_detection):
        report = analyzer.analyze_frame(normal_detection)
        assert report.failure_flags == []
        assert report.reliability_score == 1.0

    def test_night_mode_detected(self, analyzer, night_detection):
        report = analyzer.analyze_frame(night_detection)
        assert FailureCategory.NIGHT_MODE in report.failure_flags
        assert report.reliability_score < 1.0

    def test_weather_degradation(self, analyzer, rain_detection):
        report = analyzer.analyze_frame(rain_detection)
        assert FailureCategory.WEATHER_DEGRADATION in report.failure_flags

    def test_camera_failure(self, analyzer, camera_failure_detection):
        report = analyzer.analyze_frame(camera_failure_detection)
        assert FailureCategory.CAMERA_FAILURE in report.failure_flags
        assert report.reliability_score <= 0.5

    def test_low_resolution(self, analyzer, low_res_detection):
        report = analyzer.analyze_frame(low_res_detection)
        assert FailureCategory.LOW_RESOLUTION in report.failure_flags

    def test_occlusion_detection(self, analyzer):
        """Overlapping bounding boxes should flag occlusion."""
        detection = {
            "camera_id": "test",
            "timestamp": "",
            "num_detections": 2,
            "mean_confidence": 0.8,
            "mean_brightness": 140,
            "image_width": 1920,
            "weather_condition": "Fair",
            "low_confidence_count": 0,
            "detections": [
                {"class_id": 2, "confidence": 0.9, "bbox_xyxy": [100, 200, 300, 400]},
                {"class_id": 2, "confidence": 0.85, "bbox_xyxy": [110, 210, 310, 410]},
            ],
        }
        report = analyzer.analyze_frame(detection)
        assert FailureCategory.OCCLUSION in report.failure_flags

    def test_low_confidence_flagging(self, analyzer):
        detection = {
            "camera_id": "test",
            "timestamp": "",
            "num_detections": 3,
            "mean_confidence": 0.18,
            "mean_brightness": 140,
            "image_width": 1920,
            "weather_condition": "Fair",
            "low_confidence_count": 2,
            "detections": [],
        }
        report = analyzer.analyze_frame(detection)
        assert FailureCategory.LOW_CONFIDENCE in report.failure_flags

    def test_multiple_failures_stack(self, analyzer):
        """Night + rain should both be flagged."""
        detection = {
            "camera_id": "test",
            "timestamp": "",
            "num_detections": 1,
            "mean_confidence": 0.2,
            "mean_brightness": 30,
            "image_width": 1920,
            "weather_condition": "Heavy Rain",
            "low_confidence_count": 1,
            "detections": [],
        }
        report = analyzer.analyze_frame(detection)
        assert FailureCategory.NIGHT_MODE in report.failure_flags
        assert FailureCategory.WEATHER_DEGRADATION in report.failure_flags
        assert FailureCategory.LOW_CONFIDENCE in report.failure_flags
        assert report.reliability_score < 0.7


class TestIoUComputation:
    def test_perfect_overlap(self, analyzer):
        iou = analyzer._compute_iou([0, 0, 10, 10], [0, 0, 10, 10])
        assert iou == 1.0

    def test_no_overlap(self, analyzer):
        iou = analyzer._compute_iou([0, 0, 10, 10], [20, 20, 30, 30])
        assert iou == 0.0

    def test_partial_overlap(self, analyzer):
        iou = analyzer._compute_iou([0, 0, 10, 10], [5, 5, 15, 15])
        assert 0.1 < iou < 0.3  # ~14.3%

    def test_contained_box(self, analyzer):
        iou = analyzer._compute_iou([0, 0, 20, 20], [5, 5, 15, 15])
        assert 0.2 < iou < 0.3


class TestCameraScorecard:
    def test_scorecard_generation(self, analyzer, normal_detection, night_detection):
        results = [normal_detection] * 8 + [night_detection] * 2
        card = analyzer.generate_camera_scorecard(results, "1701")

        assert card.camera_id == "1701"
        assert card.total_frames == 10
        assert card.overall_reliability > 0.8
        assert card.failure_rate < 30  # Less than 30% frames have failures

    def test_empty_results(self, analyzer):
        card = analyzer.generate_camera_scorecard([], "1701")
        assert card.total_frames == 0


# =============================================================================
# Drift Monitor Tests
# =============================================================================

class TestPSI:
    def test_identical_distributions(self):
        baseline = np.random.normal(128, 20, 1000)
        psi = compute_psi(baseline, baseline)
        assert psi < 0.05  # Should be near zero

    def test_shifted_distribution(self):
        baseline = np.random.normal(128, 20, 1000)
        shifted = np.random.normal(80, 20, 1000)  # Major shift
        psi = compute_psi(baseline, shifted)
        assert psi > 0.2  # Should detect significant drift

    def test_empty_arrays(self):
        assert compute_psi(np.array([]), np.array([1, 2, 3])) == 0.0

    def test_constant_values(self):
        psi = compute_psi(np.array([5, 5, 5]), np.array([5, 5, 5]))
        assert psi == 0.0


class TestKSTest:
    def test_same_distribution(self):
        data = np.random.normal(0.7, 0.1, 100)
        result = compute_ks_test(data, data)
        assert result["drift_detected"] is False

    def test_different_distribution(self):
        baseline = np.random.normal(0.7, 0.1, 200)
        shifted = np.random.normal(0.4, 0.1, 200)
        result = compute_ks_test(baseline, shifted)
        assert result["drift_detected"] is True
        assert result["p_value"] < 0.05

    def test_small_sample(self):
        result = compute_ks_test(np.array([1, 2]), np.array([3, 4]))
        assert result["drift_detected"] is False  # Too few samples


class TestDriftMonitor:
    def test_baseline_setting(self, drift_monitor):
        baseline_data = [
            {"mean_brightness": 130 + np.random.normal(0, 5),
             "mean_confidence": 0.7 + np.random.normal(0, 0.05),
             "num_vehicles": 5}
            for _ in range(100)
        ]
        drift_monitor.set_baseline(baseline_data)

        assert drift_monitor.baseline_brightness is not None
        assert len(drift_monitor.baseline_brightness) == 100

    def test_no_drift_on_similar_data(self, drift_monitor):
        np.random.seed(42)
        baseline = [
            {"mean_brightness": 130 + np.random.normal(0, 5),
             "mean_confidence": 0.7 + np.random.normal(0, 0.05),
             "num_vehicles": 5}
            for _ in range(200)
        ]
        drift_monitor.set_baseline(baseline)

        current = [
            {"mean_brightness": 130 + np.random.normal(0, 5),
             "mean_confidence": 0.7 + np.random.normal(0, 0.05),
             "num_vehicles": 5}
            for _ in range(50)
        ]
        alerts = drift_monitor.check_drift(current)
        # Should have few or no alerts on similar data
        data_alerts = [a for a in alerts if a.drift_type == "data"]
        assert len(data_alerts) == 0

    def test_brightness_drift_detected(self, drift_monitor):
        np.random.seed(42)
        baseline = [
            {"mean_brightness": 130, "mean_confidence": 0.7, "num_vehicles": 5}
            for _ in range(200)
        ]
        drift_monitor.set_baseline(baseline)

        # Sudden drop in brightness (night shift or camera degradation)
        dark_data = [
            {"mean_brightness": 40, "mean_confidence": 0.4, "num_vehicles": 2}
            for _ in range(50)
        ]
        alerts = drift_monitor.check_drift(dark_data)
        data_alerts = [a for a in alerts if a.drift_type == "data"]
        assert len(data_alerts) > 0

    def test_drift_summary(self, drift_monitor):
        summary = drift_monitor.get_drift_summary()
        assert "total_alerts" in summary
        assert "by_type" in summary
        assert "by_severity" in summary
