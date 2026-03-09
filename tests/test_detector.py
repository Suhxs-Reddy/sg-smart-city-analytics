"""
Tests for the YOLOv11 detection pipeline.

Tests cover:
- Image quality metric computation
- DetectionResult serialization
- Batch processing path generation
- YOLO label generation from detection results
"""

import json

import pytest

from src.detection.detector import (
    TRAFFIC_CLASSES,
    DetectionResult,
    generate_yolo_labels,
)

# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_detection_result():
    """A realistic detection result for testing."""
    return DetectionResult(
        camera_id="1701",
        timestamp="2026-03-08T15:00:00+08:00",
        image_path="/data/raw/2026-03-08/1701/15-00-00.jpg",
        image_width=1920,
        image_height=1080,
        num_detections=5,
        num_vehicles=4,
        num_persons=1,
        class_counts={"car": 3, "truck": 1, "person": 1},
        mean_confidence=0.72,
        min_confidence=0.45,
        max_confidence=0.91,
        low_confidence_count=0,
        mean_brightness=142.5,
        is_night=False,
        is_grayscale=False,
        detections=[
            {
                "class_id": 2, "class_name": "car", "confidence": 0.91,
                "bbox_normalized": [0.3, 0.5, 0.1, 0.08],
                "bbox_xyxy": [480.0, 496.0, 672.0, 582.4],
            },
            {
                "class_id": 2, "class_name": "car", "confidence": 0.75,
                "bbox_normalized": [0.6, 0.4, 0.08, 0.07],
                "bbox_xyxy": [1075.2, 388.8, 1228.8, 464.4],
            },
            {
                "class_id": 2, "class_name": "car", "confidence": 0.68,
                "bbox_normalized": [0.15, 0.6, 0.09, 0.07],
                "bbox_xyxy": [201.6, 610.2, 374.4, 685.8],
            },
            {
                "class_id": 7, "class_name": "truck", "confidence": 0.82,
                "bbox_normalized": [0.5, 0.3, 0.15, 0.12],
                "bbox_xyxy": [816.0, 259.2, 1104.0, 388.8],
            },
            {
                "class_id": 0, "class_name": "person", "confidence": 0.45,
                "bbox_normalized": [0.8, 0.7, 0.03, 0.1],
                "bbox_xyxy": [1507.2, 702.0, 1564.8, 810.0],
            },
        ],
        inference_time_ms=6.2,
    )


@pytest.fixture
def night_detection_result():
    """Detection result from a night image."""
    return DetectionResult(
        camera_id="2701",
        timestamp="2026-03-08T23:00:00+08:00",
        image_path="/data/raw/2026-03-08/2701/23-00-00.jpg",
        image_width=1920,
        image_height=1080,
        num_detections=1,
        num_vehicles=1,
        class_counts={"car": 1},
        mean_confidence=0.35,
        min_confidence=0.35,
        max_confidence=0.35,
        mean_brightness=38.2,
        is_night=True,
        detections=[
            {
                "class_id": 2, "class_name": "car", "confidence": 0.35,
                "bbox_normalized": [0.5, 0.5, 0.1, 0.08],
                "bbox_xyxy": [864.0, 496.8, 1056.0, 583.2],
            },
        ],
        inference_time_ms=5.8,
    )


# =============================================================================
# DetectionResult Tests
# =============================================================================

class TestDetectionResult:
    def test_serialization(self, sample_detection_result):
        d = sample_detection_result.to_dict()
        assert d["camera_id"] == "1701"
        assert d["num_vehicles"] == 4
        assert d["num_persons"] == 1
        assert len(d["detections"]) == 5
        assert isinstance(d["class_counts"], dict)

    def test_json_roundtrip(self, sample_detection_result):
        d = sample_detection_result.to_dict()
        json_str = json.dumps(d)
        restored = json.loads(json_str)
        assert restored["camera_id"] == "1701"
        assert restored["mean_confidence"] == 0.72

    def test_night_flags(self, night_detection_result):
        assert night_detection_result.is_night is True
        assert night_detection_result.mean_brightness < 60

    def test_empty_result(self):
        result = DetectionResult(
            camera_id="test",
            timestamp="",
            image_path="",
            image_width=0,
            image_height=0,
        )
        assert result.num_detections == 0
        assert result.mean_confidence == 0.0
        assert result.detections == []


# =============================================================================
# YOLO Label Generation Tests
# =============================================================================

class TestYoloLabelGeneration:
    def test_basic_generation(self, sample_detection_result, tmp_path):
        """Test that labels are generated in correct YOLO format."""
        generate_yolo_labels(
            [sample_detection_result],
            str(tmp_path / "labels"),
            confidence_threshold=0.3,
        )

        label_file = tmp_path / "labels" / "15-00-00.txt"
        assert label_file.exists()

        lines = label_file.read_text().strip().split("\n")
        # 4 detections above 0.3 confidence (person at 0.45 passes too)
        assert len(lines) == 5

        # Check format: class_id x_center y_center width height
        parts = lines[0].split()
        assert len(parts) == 5
        assert int(parts[0]) in range(6)  # Valid remapped class
        for val in parts[1:]:
            assert 0 <= float(val) <= 1  # Normalized coords

    def test_confidence_filtering(self, sample_detection_result, tmp_path):
        """Higher threshold should filter out low-conf detections."""
        generate_yolo_labels(
            [sample_detection_result],
            str(tmp_path / "labels"),
            confidence_threshold=0.7,
        )

        label_file = tmp_path / "labels" / "15-00-00.txt"
        lines = label_file.read_text().strip().split("\n")
        # Only detections >= 0.7: car(0.91), car(0.75), truck(0.82) = 3
        assert len(lines) == 3

    def test_empty_image_generates_empty_file(self, tmp_path):
        """Image with no detections should produce empty label file."""
        empty_result = DetectionResult(
            camera_id="test",
            timestamp="",
            image_path="/fake/image.jpg",
            image_width=1920,
            image_height=1080,
        )
        generate_yolo_labels([empty_result], str(tmp_path / "labels"))

        label_file = tmp_path / "labels" / "image.txt"
        assert label_file.exists()
        assert label_file.read_text().strip() == ""

    def test_class_remapping(self, sample_detection_result, tmp_path):
        """Verify COCO classes are remapped to contiguous IDs."""
        generate_yolo_labels(
            [sample_detection_result],
            str(tmp_path / "labels"),
            confidence_threshold=0.0,
        )

        label_file = tmp_path / "labels" / "15-00-00.txt"
        lines = label_file.read_text().strip().split("\n")

        class_ids = set()
        for line in lines:
            class_ids.add(int(line.split()[0]))

        # car→0, truck→3, person→4
        assert 0 in class_ids   # car
        assert 3 in class_ids   # truck
        assert 4 in class_ids   # person


class TestTrafficClasses:
    def test_expected_classes(self):
        assert TRAFFIC_CLASSES[2] == "car"
        assert TRAFFIC_CLASSES[5] == "bus"
        assert TRAFFIC_CLASSES[7] == "truck"
        assert TRAFFIC_CLASSES[0] == "person"
        assert len(TRAFFIC_CLASSES) == 6
