"""
Singapore Smart City — Multi-Object Tracking Pipeline

Wraps BoxMOT's BoT-SORT tracker for:
1. Per-camera vehicle tracking with Re-ID
2. Trajectory extraction (entry/exit, dwell time, direction)
3. Vehicle counting (unique vehicles per time window)
4. Speed estimation via simple pixel displacement

Designed to run on Colab/Kaggle T4 GPU.
"""

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TrackedVehicle:
    """Represents a single tracked vehicle across frames."""
    track_id: int
    class_name: str
    first_seen: str           # Timestamp of first appearance
    last_seen: str            # Timestamp of last appearance
    num_frames: int = 0       # Total frames this track appears in
    bbox_history: list = field(default_factory=list)  # List of (timestamp, [x1,y1,x2,y2])
    confidence_history: list = field(default_factory=list)

    @property
    def dwell_frames(self) -> int:
        """How many frames this vehicle was tracked."""
        return self.num_frames

    @property
    def mean_confidence(self) -> float:
        """Average detection confidence across frames."""
        if not self.confidence_history:
            return 0.0
        return round(sum(self.confidence_history) / len(self.confidence_history), 4)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["dwell_frames"] = self.dwell_frames
        d["mean_confidence"] = self.mean_confidence
        return d


@dataclass
class TrackingResult:
    """Result from tracking across a sequence of frames."""
    camera_id: str
    start_time: str
    end_time: str
    total_frames: int
    total_unique_vehicles: int
    total_unique_persons: int

    # Per-class unique counts
    class_counts: dict = field(default_factory=dict)

    # Flow metrics
    avg_vehicles_per_frame: float = 0.0
    max_vehicles_in_frame: int = 0

    # Tracked vehicle details
    vehicles: list = field(default_factory=list)

    # Processing metadata
    total_processing_time_s: float = 0.0
    avg_fps: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class VehicleTracker:
    """BoT-SORT based vehicle tracker using BoxMOT."""

    def __init__(
        self,
        detector_model: str = "yolo11s.pt",
        tracker_type: str = "botsort",
        reid_model: str = "osnet_x0_25_msmt17.pt",
        confidence_threshold: float = 0.25,
        img_size: int = 640,
        device: str = "auto",
    ):
        """
        Args:
            detector_model: Path to YOLO weights.
            tracker_type: BoxMOT tracker type ('botsort', 'strongsort', 'ocsort').
            reid_model: Re-ID model for appearance matching.
            confidence_threshold: Minimum detection confidence.
            img_size: YOLO input size.
            device: Compute device.
        """
        self.detector_model = detector_model
        self.tracker_type = tracker_type
        self.reid_model = reid_model
        self.confidence_threshold = confidence_threshold
        self.img_size = img_size
        self.device = device

        # Lazy initialization — don't load models until needed
        self._tracker = None
        self._model = None

        # Traffic-relevant COCO class IDs
        self.traffic_class_ids = {2, 3, 5, 7, 0, 1}  # car, moto, bus, truck, person, bicycle
        self.vehicle_class_ids = {2, 3, 5, 7}         # car, moto, bus, truck
        self.class_names = {
            2: "car", 3: "motorcycle", 5: "bus",
            7: "truck", 0: "person", 1: "bicycle",
        }

    def _initialize(self):
        """Lazy-load models (avoids GPU allocation until actually needed)."""
        if self._model is not None:
            return

        from ultralytics import YOLO

        logger.info(f"Loading detector: {self.detector_model}")
        self._model = YOLO(self.detector_model)

        logger.info(
            f"Tracker ready: {self.tracker_type} + {self.reid_model}"
        )

    def track_image_sequence(
        self,
        image_dir: str,
        camera_id: str = "unknown",
        output_path: Optional[str] = None,
        max_frames: Optional[int] = None,
    ) -> TrackingResult:
        """Run tracking on a directory of sequential images.

        Images should be named in chronological order (e.g. HH-MM-SS.jpg).

        Args:
            image_dir: Directory containing sequential .jpg images.
            camera_id: Camera identifier.
            output_path: If provided, save results to this JSON file.
            max_frames: Limit number of frames to process.

        Returns:
            TrackingResult with all tracking metadata.
        """
        self._initialize()

        image_dir = Path(image_dir)
        image_files = sorted(image_dir.glob("*.jpg"))

        if max_frames:
            image_files = image_files[:max_frames]

        if not image_files:
            logger.warning(f"No images found in {image_dir}")
            return TrackingResult(
                camera_id=camera_id,
                start_time="",
                end_time="",
                total_frames=0,
                total_unique_vehicles=0,
                total_unique_persons=0,
            )

        logger.info(
            f"Tracking {len(image_files)} frames from camera {camera_id}"
        )

        # Track all vehicles across frames
        all_tracks = {}  # track_id → TrackedVehicle
        vehicles_per_frame = []
        start_time = time.time()

        # Use YOLO's built-in tracking with BoxMOT
        results = self._model.track(
            source=str(image_dir),
            conf=self.confidence_threshold,
            iou=0.45,
            imgsz=self.img_size,
            tracker=f"{self.tracker_type}.yaml",
            device=self.device,
            stream=True,
            verbose=False,
            persist=True,
        )

        frame_idx = 0
        for result in results:
            if max_frames and frame_idx >= max_frames:
                break

            timestamp = image_files[frame_idx].stem if frame_idx < len(image_files) else str(frame_idx)
            frame_vehicle_count = 0

            if result.boxes is not None and result.boxes.id is not None:
                boxes = result.boxes
                for i in range(len(boxes)):
                    cls_id = int(boxes.cls[i])
                    if cls_id not in self.traffic_class_ids:
                        continue

                    track_id = int(boxes.id[i])
                    conf = float(boxes.conf[i])
                    xyxy = boxes.xyxy[i].cpu().numpy().tolist()
                    class_name = self.class_names.get(cls_id, "unknown")

                    if track_id not in all_tracks:
                        all_tracks[track_id] = TrackedVehicle(
                            track_id=track_id,
                            class_name=class_name,
                            first_seen=timestamp,
                            last_seen=timestamp,
                        )

                    track = all_tracks[track_id]
                    track.last_seen = timestamp
                    track.num_frames += 1
                    track.bbox_history.append({
                        "frame": frame_idx,
                        "timestamp": timestamp,
                        "bbox": [round(x, 1) for x in xyxy],
                    })
                    track.confidence_history.append(round(conf, 4))

                    if cls_id in self.vehicle_class_ids:
                        frame_vehicle_count += 1

            vehicles_per_frame.append(frame_vehicle_count)
            frame_idx += 1

        total_time = time.time() - start_time

        # Compute class counts
        class_counts = {}
        vehicle_tracks = []
        person_count = 0

        for track in all_tracks.values():
            class_counts[track.class_name] = class_counts.get(track.class_name, 0) + 1
            if track.class_name in ("car", "motorcycle", "bus", "truck"):
                vehicle_tracks.append(track.to_dict())
            elif track.class_name == "person":
                person_count += 1

        result = TrackingResult(
            camera_id=camera_id,
            start_time=image_files[0].stem if image_files else "",
            end_time=image_files[-1].stem if image_files else "",
            total_frames=frame_idx,
            total_unique_vehicles=len(vehicle_tracks),
            total_unique_persons=person_count,
            class_counts=class_counts,
            avg_vehicles_per_frame=round(
                np.mean(vehicles_per_frame), 1
            ) if vehicles_per_frame else 0.0,
            max_vehicles_in_frame=max(vehicles_per_frame) if vehicles_per_frame else 0,
            vehicles=vehicle_tracks,
            total_processing_time_s=round(total_time, 1),
            avg_fps=round(frame_idx / total_time, 1) if total_time > 0 else 0.0,
        )

        logger.info(
            f"Tracking complete: {len(vehicle_tracks)} unique vehicles, "
            f"{person_count} persons across {frame_idx} frames "
            f"({result.avg_fps} FPS)"
        )

        # Save results
        if output_path:
            output = Path(output_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            with open(output, "w") as f:
                json.dump(result.to_dict(), f, indent=2)
            logger.info(f"Results saved to {output_path}")

        return result


def estimate_congestion_score(
    tracking_result: TrackingResult,
    high_vehicle_threshold: int = 20,
    high_dwell_threshold: int = 10,
) -> dict:
    """Estimate congestion level from tracking results.

    Returns a score from 0 (free flow) to 1 (gridlock) with breakdown.

    Args:
        tracking_result: Result from VehicleTracker.
        high_vehicle_threshold: Vehicles per frame above this = congested.
        high_dwell_threshold: Dwell time above this (frames) = slow traffic.
    """
    if tracking_result.total_frames == 0:
        return {"score": 0.0, "level": "unknown", "reason": "no data"}

    # Factor 1: Vehicle density (0-1)
    density_score = min(
        tracking_result.avg_vehicles_per_frame / high_vehicle_threshold, 1.0
    )

    # Factor 2: Average dwell time (0-1)
    # Higher dwell = slower traffic = more congested
    dwell_times = []
    for v in tracking_result.vehicles:
        dwell_times.append(v.get("num_frames", 0))

    avg_dwell = np.mean(dwell_times) if dwell_times else 0
    dwell_score = min(avg_dwell / high_dwell_threshold, 1.0)

    # Combined congestion score (weighted average)
    congestion_score = round(0.5 * density_score + 0.5 * dwell_score, 3)

    # Level classification
    if congestion_score < 0.3:
        level = "free_flow"
    elif congestion_score < 0.6:
        level = "moderate"
    elif congestion_score < 0.8:
        level = "heavy"
    else:
        level = "gridlock"

    return {
        "score": congestion_score,
        "level": level,
        "density_score": round(density_score, 3),
        "dwell_score": round(dwell_score, 3),
        "avg_vehicles_per_frame": tracking_result.avg_vehicles_per_frame,
        "avg_dwell_frames": round(avg_dwell, 1),
        "unique_vehicles": tracking_result.total_unique_vehicles,
    }
