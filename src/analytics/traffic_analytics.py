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

# Passenger Car Equivalent (PCE) factors — HCM 6th Ed. Chapter 12 Table 12-9.
# PCE accounts for the disproportionate road space and headway consumed by
# heavy vehicles relative to a standard passenger car.
# Applied to bbox area before computing occupancy so LOS reflects effective
# road consumption, not just pixel count.
#   bus:        2.5 — takes 2.5× the effective lane space of a car
#   truck:      2.0
#   motorcycle: 0.5 — smaller footprint, gaps between lanes
#   bicycle:    0.5
#   car:        1.0 (baseline)
_PCE: dict[str, float] = {
    "car":        1.0,
    "bus":        2.5,
    "truck":      2.0,
    "motorcycle": 0.5,
    "bicycle":    0.5,
    "person":     0.3,   # pedestrian (rare on expressways; minimal weight)
}

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

# HCM 6th Edition Chapter 12 — density-based LOS for basic freeway segments.
# Units: passenger cars per km per lane (pc/km/ln).
# Used when speed_kmh is available from SpeedEstimator.
_LOS_DENSITY_THRESHOLDS = [
    (7.0,  "A"),   # free flow, density ≤ 7  pc/km/ln
    (11.0, "B"),   # ≤ 11
    (16.0, "C"),   # ≤ 16
    (22.0, "D"),   # ≤ 22
    (28.0, "E"),   # ≤ 28  (approaching capacity)
    (1e9,  "F"),   # > 28  (forced/breakdown flow)
]

# Singapore expressway speed limits (km/h) — same as SpeedEstimator
_SPEED_LIMITS: dict[str, int] = {
    "CTE": 80, "AYE": 80, "MCE": 80, "KJE": 80, "KPE": 80, "NSC": 80,
    "PIE": 90, "ECP": 90, "TPE": 90, "BKE": 90, "SLE": 90,
}

# Singapore expressway lane counts per road direction.
# Source: LTA Road Design Guidelines and OneMap road metadata.
# MCE is dual 2-lane; CTE/PIE are 4-lane at most stretches.
# Used in HCM density LOS calculation: density (pc/km) / lanes → pc/km/ln.
_ROAD_LANES: dict[str, int] = {
    "CTE": 4,   # 3-4 lanes; use 4 (conservative)
    "PIE": 4,   # up to 4 lanes on main carriageway
    "AYE": 3,
    "ECP": 3,
    "TPE": 3,
    "BKE": 3,
    "KJE": 3,
    "KPE": 3,
    "SLE": 3,
    "MCE": 2,   # 2-lane tunnel
    "NSC": 2,
}

# LTA EMAS peak hour schedule — Singapore.
# AM peak: 07:00–09:30, PM peak: 17:30–20:00 (weekday).
# Shoulder periods: ±1h around peak.
# During peak hours, the same occupancy represents higher operational stress
# because capacity is fully utilised and incident recovery is slower.
_PEAK_HOUR_RANGES = [
    (7,  9,  1.30),   # AM peak core  (+30% congestion sensitivity)
    (6,  10, 1.15),   # AM shoulder   (+15%)
    (17, 20, 1.30),   # PM peak core
    (16, 21, 1.15),   # PM shoulder
]


def _peak_multiplier(hour: int) -> float:
    """
    LTA EMAS peak-hour congestion multiplier.
    Returns 1.0 during off-peak (midnight–6am, 10am–4pm).
    """
    best = 1.0
    for start, end, mult in _PEAK_HOUR_RANGES:
        if start <= hour < end:
            best = max(best, mult)
    return best

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

    # Speed context (from SpeedEstimator when available, else 0.0)
    speed_kmh:    float = 0.0   # smoothed inter-camera speed
    speed_limit:  int   = 80    # road speed limit
    speed_ratio:  float = 0.0   # speed_kmh / speed_limit (0 when unknown)
    los_method:   str   = "occupancy"  # "occupancy" or "density"
    peak_hour:    float = 1.0   # LTA EMAS peak-hour multiplier applied

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
            "los_method":          self.los_method,
            "heavy_vehicle_ratio": round(self.heavy_vehicle_ratio, 4),
            "congestion_score":    round(self.congestion_score, 4),
            "weather":             self.weather,
            "speed_kmh":           round(self.speed_kmh, 1),
            "speed_limit":         self.speed_limit,
            "speed_ratio":         round(self.speed_ratio, 3),
            "peak_hour_mult":      round(self.peak_hour, 2),
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
        speed_kmh: float = 0.0,
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
            speed_kmh:  Kalman-smoothed inter-camera speed from SpeedEstimator.
                        When > 0, enables HCM density-based LOS and the speed
                        deficit congestion term. Defaults to 0.0 (not available).

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

        # PCE-weighted occupancy (HCM 6th Ed. Chapter 12, Table 12-9).
        # Each vehicle's bbox area is multiplied by its Passenger Car Equivalent
        # before summing, so a bus correctly contributes 2.5× more effective road
        # space than a car of the same pixel area. This fixes LOS underestimation
        # on frames with heavy goods vehicles common on AYE/PIE during peak hours.
        bbox_area_sum = sum(
            (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]) * _PCE.get(d.cls, 1.0)
            for d in vehicles
        )
        occupancy = min(bbox_area_sum / roi_area, 1.0) if roi_area > 0 else 0.0

        # Speed context
        limit = _SPEED_LIMITS.get(road, 80)
        speed_ratio = (speed_kmh / limit) if speed_kmh > 0 and limit > 0 else 0.0

        # ── Level of Service ──────────────────────────────────────────────
        # Primary: HCM 6th Ed. Chapter 12 density-based LOS when speed known.
        # Fallback: occupancy-based LOS (HCM 2010 camera adaptation).
        if speed_kmh > 0 and total > 0:
            los, los_method = self._los_from_density(total, speed_ratio, roi_area, road=road), "density"
        else:
            los, los_method = self._los_from_occupancy(occupancy), "occupancy"

        # Heavy vehicle ratio
        heavy_count = sum(count_by_class.get(c, 0) for c in HEAVY_CLASSES)
        hv_ratio = heavy_count / total if total > 0 else 0.0

        # Weather penalty
        weather_key = weather.lower().strip()
        penalty = _WEATHER_PENALTY.get(weather_key, 0.0)
        if weather_key not in _WEATHER_PENALTY:
            for key, val in _WEATHER_PENALTY.items():
                if key in weather_key:
                    penalty = val
                    break

        # ── Peak hour multiplier (LTA EMAS) ──────────────────────────────
        # Parse hour from timestamp (ISO 8601). Falls back to 12 if unavailable.
        try:
            from datetime import datetime as _dt
            _hour = _dt.fromisoformat(timestamp).hour if timestamp else 12
        except (ValueError, AttributeError):
            _hour = 12
        peak_mult = _peak_multiplier(_hour)

        # ── Congestion score (0-1) ────────────────────────────────────────
        # Composite (Lomax et al. 1997; BPR volume-delay; LTA EMAS):
        #   50% occupancy      — base spatial density proxy
        #   20% HV effect      — heavy vehicles amplify effective occupancy
        #   15% weather        — adverse weather degrades effective capacity
        #   15% speed deficit  — slow-moving traffic (incident detection at night)
        #
        # The raw score is then multiplied by the LTA peak-hour factor so
        # the same occupancy is classified more severely during AM/PM peak.
        # Clamped to [0, 1] after scaling.
        speed_deficit = max(0.0, 1.0 - speed_ratio) if speed_kmh > 0 else 0.0
        raw_congestion = (
            0.50 * occupancy
            + 0.20 * hv_ratio * occupancy
            + 0.15 * penalty
            + 0.15 * speed_deficit
        )
        congestion = min(raw_congestion * peak_mult, 1.0)

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
            speed_kmh=speed_kmh,
            speed_limit=limit,
            speed_ratio=round(speed_ratio, 3),
            los_method=los_method,
            peak_hour=peak_mult,
        )

    # ------------------------------------------------------------------

    def _los_from_occupancy(self, occupancy: float) -> LOS:
        for threshold, grade in _LOS_THRESHOLDS:
            if occupancy <= threshold:
                return LOS(grade)
        return LOS.F

    def _los_from_density(
        self, total_vehicles: int, speed_ratio: float, roi_area_px: int,
        road: str = "Unknown",
    ) -> LOS:
        """
        HCM 6th Edition Chapter 12 density-based LOS for basic freeway segments.

        Density proxy (pc/km/ln) = vehicles / (speed_ratio × frame_road_km × lanes)

        Frame road coverage:
          ~40m for 1920×1080 cameras (standard LTA HD camera at 30m height).
          ~15m for 320×240 cameras (older LTA cameras, narrower FOV).
          Approximated from roi_area_px: smaller frame → shorter road coverage.

        Lane count is looked up from _ROAD_LANES using the Singapore expressway
        road code (MCE=2, CTE/PIE=4, most others=3). Falls back to 3 if unknown.

        (HCM 6th Ed. Table 12-6, Basic Freeway Segment LOS criteria)
        """
        # Frame road coverage scales with frame resolution
        # Full-HD (≥1M px): ~40m; legacy (≤0.1M px): ~15m; interpolated in between
        HD_AREA_PX   = 1920 * 1080     # 2,073,600
        LEGACY_AREA  = 320  * 240      #    76,800
        HD_ROAD_KM   = 0.040
        LEGACY_ROAD_KM = 0.015
        frac = min(max((roi_area_px - LEGACY_AREA) / (HD_AREA_PX - LEGACY_AREA), 0.0), 1.0)
        frame_road_km = LEGACY_ROAD_KM + frac * (HD_ROAD_KM - LEGACY_ROAD_KM)

        lanes = _ROAD_LANES.get(road, 3)

        # Protect against zero/degenerate speed
        effective_speed_ratio = max(speed_ratio, 0.05)
        density = total_vehicles / max(effective_speed_ratio * frame_road_km * lanes, 1e-6)

        for threshold, grade in _LOS_DENSITY_THRESHOLDS:
            if density <= threshold:
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
