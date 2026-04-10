"""
Traffic Analytics — Per-Camera Metrics

Computes traffic state from CATI detection outputs:
  - Vehicle count (per class and total)
  - Occupancy ratio  = Σ(bbox areas) / ROI area  [proxy for density]
  - Level of Service (LOS A-F) from HCM thresholds
  - Heavy vehicle ratio (buses + trucks / total)
  - Congestion score (0-1, fused occupancy + HV ratio + weather penalty)

Sits directly on top of CATI detector output and feeds into:
  speed_estimator.py (inter-camera speed via re-ID)
  server.py          (REST API)

Usage:
    from src.analytics.traffic_analytics import TrafficAnalytics, Detection

    analytics = TrafficAnalytics()
    detections = [Detection(cls="car", bbox=(x1,y1,x2,y2), conf=0.8), ...]
    state = analytics.compute(detections, frame_wh=(1920,1080),
                              weather="Thundery Showers", camera_id="1001")
    print(state.los, state.congestion_score)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# YOLO class names as labelled in our dataset
VEHICLE_CLASSES   = {"car", "motorcycle", "bus", "truck", "bicycle"}
HEAVY_CLASSES     = {"bus", "truck"}

# HCM Level of Service thresholds (occupancy-based for camera surveillance)
# Source: Highway Capacity Manual 2010, adapted for image-based occupancy
_LOS_THRESHOLDS = [
    (0.12, "A"),
    (0.20, "B"),
    (0.33, "C"),
    (0.45, "D"),
    (0.55, "E"),
    (1.00, "F"),
]

# Weather congestion penalty — adverse weather degrades effective capacity
_WEATHER_PENALTY: dict[str, float] = {
    "thundery showers":        0.15,
    "heavy thundery showers":  0.20,
    "moderate rain":           0.10,
    "heavy rain":              0.15,
    "showers":                 0.08,
    "hazy":                    0.05,
    "windy":                   0.03,
    "fair":                    0.00,
    "partly cloudy":           0.00,
    "cloudy":                  0.02,
    "unknown":                 0.00,
}


# ---------------------------------------------------------------------------
# Input / Output types
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    """Single detection from CATI detector."""
    cls: str                        # class name: "car", "bus", etc.
    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2) in pixels
    conf: float


class LOS(str, Enum):
    A = "A"   # Free flow
    B = "B"   # Reasonably free flow
    C = "C"   # Stable flow
    D = "D"   # Approaching unstable
    E = "E"   # Unstable / at capacity
    F = "F"   # Forced flow / breakdown


@dataclass
class TrafficState:
    """Full traffic state for one camera at one timestamp."""
    camera_id: str
    timestamp: str                  # ISO 8601

    # Counts
    total_vehicles: int
    count_by_class: dict[str, int]  # {"car": N, "bus": N, ...}

    # Core metrics
    occupancy: float                # 0.0 – 1.0
    los: LOS
    heavy_vehicle_ratio: float      # 0.0 – 1.0
    congestion_score: float         # 0.0 – 1.0  (higher = worse)

    # Context
    weather: str
    weather_penalty: float
    frame_width: int
    frame_height: int
    roi_area_px: int

    # Road info (filled in by pipeline if camera_map is wired up)
    road: str = "Unknown"
    region: str = "Unknown"
    area: str = "Unknown"

    @property
    def los_label(self) -> str:
        labels = {
            LOS.A: "Free flow",
            LOS.B: "Reasonably free flow",
            LOS.C: "Stable flow",
            LOS.D: "Approaching unstable",
            LOS.E: "Unstable / at capacity",
            LOS.F: "Forced flow / breakdown",
        }
        return labels[self.los]

    def to_dict(self) -> dict:
        return {
            "camera_id":           self.camera_id,
            "timestamp":           self.timestamp,
            "road":                self.road,
            "region":              self.region,
            "area":                self.area,
            "total_vehicles":      self.total_vehicles,
            "count_by_class":      self.count_by_class,
            "occupancy":           round(self.occupancy, 4),
            "los":                 self.los.value,
            "los_label":           self.los_label,
            "heavy_vehicle_ratio": round(self.heavy_vehicle_ratio, 4),
            "congestion_score":    round(self.congestion_score, 4),
            "weather":             self.weather,
        }


# ---------------------------------------------------------------------------
# Analytics engine
# ---------------------------------------------------------------------------

class TrafficAnalytics:
    """
    Computes per-camera traffic metrics from CATI detection outputs.

    Designed to be stateless per call — pass detections + frame size,
    get back a TrafficState. The pipeline (inference.py) is responsible
    for aggregating states over time.
    """

    def compute(
        self,
        detections: list[Detection],
        frame_wh: tuple[int, int],
        camera_id: str = "",
        timestamp: str = "",
        weather: str = "unknown",
        road: str = "Unknown",
        region: str = "Unknown",
        area: str = "Unknown",
    ) -> TrafficState:
        """
        Compute traffic state from detections for one frame.

        Args:
            detections: All detections from CATI for this frame.
            frame_wh:   (width, height) of the frame in pixels.
            camera_id:  LTA camera ID string.
            timestamp:  ISO 8601 timestamp of the frame.
            weather:    Current weather condition string.
            road/region/area: From CameraMap.lookup().

        Returns:
            TrafficState with all computed metrics.
        """
        w, h = frame_wh
        roi_area = w * h

        # Filter to vehicle detections only
        vehicles = [d for d in detections if d.cls in VEHICLE_CLASSES]

        # Count by class
        count_by_class: dict[str, int] = {}
        for d in vehicles:
            count_by_class[d.cls] = count_by_class.get(d.cls, 0) + 1

        total = len(vehicles)

        # Occupancy = sum of bbox areas / frame area
        bbox_area_sum = sum(
            (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1])
            for d in vehicles
        )
        occupancy = min(bbox_area_sum / roi_area, 1.0) if roi_area > 0 else 0.0

        # Level of Service
        los = self._los_from_occupancy(occupancy)

        # Heavy vehicle ratio
        heavy_count = sum(count_by_class.get(c, 0) for c in HEAVY_CLASSES)
        hv_ratio = heavy_count / total if total > 0 else 0.0

        # Weather penalty
        weather_key = weather.lower().strip()
        penalty = _WEATHER_PENALTY.get(weather_key, 0.0)
        # Partial match fallback
        if weather_key not in _WEATHER_PENALTY:
            for key, val in _WEATHER_PENALTY.items():
                if key in weather_key:
                    penalty = val
                    break

        # Congestion score (0-1):
        #   60% occupancy weight + 20% HV weight + 20% weather penalty
        # HV raises effective occupancy since heavy vehicles take more road space
        congestion = min(
            0.60 * occupancy
            + 0.20 * hv_ratio * occupancy   # HV effect scales with base density
            + 0.20 * penalty,
            1.0,
        )

        return TrafficState(
            camera_id=camera_id,
            timestamp=timestamp,
            total_vehicles=total,
            count_by_class=count_by_class,
            occupancy=occupancy,
            los=los,
            heavy_vehicle_ratio=hv_ratio,
            congestion_score=congestion,
            weather=weather,
            weather_penalty=penalty,
            frame_width=w,
            frame_height=h,
            roi_area_px=roi_area,
            road=road,
            region=region,
            area=area,
        )

    # ------------------------------------------------------------------

    def _los_from_occupancy(self, occupancy: float) -> LOS:
        for threshold, grade in _LOS_THRESHOLDS:
            if occupancy <= threshold:
                return LOS(grade)
        return LOS.F

    def network_summary(self, states: list[TrafficState]) -> dict:
        """
        Aggregate multiple camera states into a network-level summary.
        Called by the API to give a city-wide overview.
        """
        if not states:
            return {}

        by_region: dict[str, list[TrafficState]] = {}
        by_road: dict[str, list[TrafficState]] = {}
        for s in states:
            by_region.setdefault(s.region, []).append(s)
            by_road.setdefault(s.road, []).append(s)

        def avg(vals):
            return round(sum(vals) / len(vals), 4) if vals else 0.0

        return {
            "total_cameras": len(states),
            "total_vehicles": sum(s.total_vehicles for s in states),
            "avg_occupancy": avg([s.occupancy for s in states]),
            "avg_congestion": avg([s.congestion_score for s in states]),
            "worst_camera": max(states, key=lambda s: s.congestion_score).camera_id,
            "by_region": {
                region: {
                    "cameras": len(ss),
                    "avg_congestion": avg([s.congestion_score for s in ss]),
                    "avg_occupancy": avg([s.occupancy for s in ss]),
                    "total_vehicles": sum(s.total_vehicles for s in ss),
                }
                for region, ss in by_region.items()
            },
            "by_road": {
                road: {
                    "cameras": len(ss),
                    "avg_congestion": avg([s.congestion_score for s in ss]),
                    "avg_los": max(ss, key=lambda s: s.congestion_score).los.value,
                    "total_vehicles": sum(s.total_vehicles for s in ss),
                }
                for road, ss in by_road.items()
            },
        }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    analytics = TrafficAnalytics()

    # Simulate a busy CTE frame
    fake_detections = (
        [Detection("car", (100, 200, 300, 350), 0.9)] * 12
        + [Detection("bus", (400, 150, 700, 420), 0.85)] * 2
        + [Detection("motorcycle", (50, 300, 120, 390), 0.7)] * 4
        + [Detection("truck", (800, 100, 1100, 380), 0.88)] * 1
    )

    state = analytics.compute(
        detections=fake_detections,
        frame_wh=(1920, 1080),
        camera_id="1001",
        timestamp="2026-04-09T08:30:00+08:00",
        weather="Thundery Showers",
        road="CTE",
        region="Central",
        area="Novena",
    )

    print(json.dumps(state.to_dict(), indent=2))
    print(f"\nLOS: {state.los.value} — {state.los_label}")
    print(f"Congestion score: {state.congestion_score:.3f}")
