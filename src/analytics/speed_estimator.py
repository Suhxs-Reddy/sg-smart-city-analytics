"""
Inter-Camera Speed Estimator

Computes vehicle speed from cross-camera re-ID matches using real GPS
distances between adjacent LTA cameras on the same expressway.

Formula:
    speed (km/h) = edge_distance_km / travel_time_hours

Where:
    edge_distance_km  — Haversine distance between camera A and B
                        (from CameraNetwork edges, derived from LTA GPS coords)
    travel_time_hours — time between vehicle exit at camera A and
                        entry at camera B (from ReIDMatch timestamps)

Singapore expressway context:
    - Legal speed limits: 80 km/h (CTE/AYE/MCE), 90 km/h (PIE/ECP/TPE/BKE)
    - Typical free-flow: 70-85 km/h
    - Congested: < 40 km/h
    - Readings outside [10, 130] km/h are flagged as outliers

The speed readings feed directly into TrafficState and network_summary()
to produce per-road speed profiles alongside the occupancy/LOS metrics.

Integrates with:
    camera_network.py   (edge distances between adjacent cameras)
    vehicle_reid.py     (ReIDMatch with timestamps)
    traffic_analytics.py (SpeedReading merged into TrafficState)
    inference.py        (called after each re-ID match)

Usage:
    from src.analytics.speed_estimator import SpeedEstimator

    estimator = SpeedEstimator()
    reading = estimator.estimate(match)   # ReIDMatch → SpeedReading or None
    print(reading.speed_kmh, reading.road, reading.congestion_band)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Singapore speed limit table per road
# ---------------------------------------------------------------------------

_SPEED_LIMITS: dict[str, int] = {
    "CTE": 80,
    "AYE": 80,
    "MCE": 80,
    "KJE": 80,
    "KPE": 80,
    "NSC": 80,
    "PIE": 90,
    "ECP": 90,
    "TPE": 90,
    "BKE": 90,
    "SLE": 90,
}

# Plausible speed range — readings outside are flagged as outliers
# (camera clock drift, wrong re-ID match, or vehicle stopped on shoulder)
MIN_PLAUSIBLE_KMH = 10.0
MAX_PLAUSIBLE_KMH = 130.0

# Congestion band thresholds (fraction of speed limit)
# Aligned with LTA's EMAS (Expressway Monitoring and Advisory System)
_BAND_THRESHOLDS = [
    (0.85, "free_flow"),       # > 85% of speed limit
    (0.60, "light"),           # 60-85%
    (0.35, "moderate"),        # 35-60%
    (0.00, "heavy"),           # < 35%
]


# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------

@dataclass
class SpeedReading:
    """Speed measurement derived from one cross-camera re-ID match."""

    # Camera pair
    camera_from:  str    # upstream camera (vehicle exited here)
    camera_to:    str    # downstream camera (vehicle entered here)
    road:         str
    region:       str

    # Timestamps
    timestamp_from: str  # ISO 8601 — exit time at camera_from
    timestamp_to:   str  # ISO 8601 — entry time at camera_to

    # Measurement
    travel_time_s:  float   # seconds between cameras
    distance_km:    float   # GPS distance between cameras
    speed_kmh:      float   # computed speed

    # Context
    speed_limit:    int     # legal limit for this road (km/h)
    congestion_band: str    # free_flow / light / moderate / heavy
    is_outlier:     bool    # True if speed outside plausible range
    similarity:     float   # re-ID cosine similarity (confidence proxy)
    vehicle_cls:    str

    @property
    def speed_ratio(self) -> float:
        """Speed as fraction of speed limit (1.0 = at limit)."""
        return self.speed_kmh / self.speed_limit if self.speed_limit > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "camera_from":     self.camera_from,
            "camera_to":       self.camera_to,
            "road":            self.road,
            "region":          self.region,
            "timestamp_from":  self.timestamp_from,
            "timestamp_to":    self.timestamp_to,
            "travel_time_s":   round(self.travel_time_s, 1),
            "distance_km":     round(self.distance_km, 3),
            "speed_kmh":       round(self.speed_kmh, 1),
            "speed_limit":     self.speed_limit,
            "congestion_band": self.congestion_band,
            "speed_ratio":     round(self.speed_ratio, 3),
            "is_outlier":      self.is_outlier,
            "similarity":      self.similarity,
            "vehicle_cls":     self.vehicle_cls,
        }


# ---------------------------------------------------------------------------
# Speed estimator
# ---------------------------------------------------------------------------

class SpeedEstimator:
    """
    Computes inter-camera vehicle speed from re-ID matches.

    Maintains a rolling buffer of recent speed readings per road segment
    for smoothed per-road speed profiles.

    Args:
        buffer_size: Number of recent readings to keep per road segment
                     for rolling average speed computation.
    """

    def __init__(self, buffer_size: int = 20):
        self.buffer_size = buffer_size
        self._network = None
        # road_segment → list of recent SpeedReadings (rolling buffer)
        self._buffer: dict[str, list[SpeedReading]] = {}

    def _get_network(self):
        if self._network is None:
            from src.analytics.camera_network import CameraNetwork
            self._network = CameraNetwork()
        return self._network

    def estimate(self, match) -> Optional[SpeedReading]:
        """
        Compute speed from a ReIDMatch.

        Args:
            match: ReIDMatch from vehicle_reid.ReIDGallery.query()
                   Fields used: gallery_camera (from), query_camera (to),
                   gallery_timestamp (exit), query_timestamp (entry),
                   similarity, cls.

        Returns:
            SpeedReading, or None if cameras are not adjacent / timestamps invalid.
        """
        net = self._get_network()

        cam_from = match.gallery_camera
        cam_to   = match.query_camera

        # Look up the edge between these two cameras
        edge = self._find_edge(net, cam_from, cam_to)
        if edge is None:
            logger.debug(f"No edge between {cam_from} and {cam_to} — skipping")
            return None

        # Parse timestamps
        travel_time_s = self._travel_time_seconds(
            match.gallery_timestamp, match.query_timestamp
        )
        if travel_time_s is None or travel_time_s <= 0:
            logger.warning(f"Invalid travel time between {cam_from}→{cam_to}")
            return None

        # Compute speed
        speed_kmh = edge.distance_km / (travel_time_s / 3600.0)
        is_outlier = not (MIN_PLAUSIBLE_KMH <= speed_kmh <= MAX_PLAUSIBLE_KMH)

        road   = edge.road
        region = edge.cam_a.region
        limit  = _SPEED_LIMITS.get(road, 80)
        band   = self._congestion_band(speed_kmh, limit)

        reading = SpeedReading(
            camera_from=cam_from,
            camera_to=cam_to,
            road=road,
            region=region,
            timestamp_from=match.gallery_timestamp,
            timestamp_to=match.query_timestamp,
            travel_time_s=round(travel_time_s, 1),
            distance_km=edge.distance_km,
            speed_kmh=round(speed_kmh, 1),
            speed_limit=limit,
            congestion_band=band,
            is_outlier=is_outlier,
            similarity=match.similarity,
            vehicle_cls=match.cls,
        )

        if not is_outlier:
            seg_key = f"{cam_from}→{cam_to}"
            buf = self._buffer.setdefault(seg_key, [])
            buf.append(reading)
            if len(buf) > self.buffer_size:
                buf.pop(0)

        return reading

    def average_speed(self, road: str | None = None) -> dict[str, float]:
        """
        Rolling average speed (km/h) per road segment.

        Args:
            road: Filter to a specific road (e.g. "CTE"), or None for all.

        Returns:
            Dict mapping "camA→camB" segment key to average speed (km/h).
        """
        result = {}
        for seg, readings in self._buffer.items():
            if not readings:
                continue
            if road and readings[0].road != road:
                continue
            avg = sum(r.speed_kmh for r in readings) / len(readings)
            result[seg] = round(avg, 1)
        return result

    def road_speed_profile(self) -> dict[str, dict]:
        """
        Per-road speed summary — avg speed, band, and number of readings.
        Used by network_summary() and the REST API.
        """
        by_road: dict[str, list[SpeedReading]] = {}
        for readings in self._buffer.values():
            for r in readings:
                by_road.setdefault(r.road, []).append(r)

        profile = {}
        for road, readings in by_road.items():
            speeds = [r.speed_kmh for r in readings]
            avg    = sum(speeds) / len(speeds)
            limit  = _SPEED_LIMITS.get(road, 80)
            profile[road] = {
                "avg_speed_kmh":   round(avg, 1),
                "speed_limit":     limit,
                "speed_ratio":     round(avg / limit, 3),
                "congestion_band": self._congestion_band(avg, limit),
                "num_readings":    len(readings),
            }
        return profile

    # ------------------------------------------------------------------

    def _find_edge(self, net, cam_from: str, cam_to: str):
        """Find the CameraEdge between cam_from and cam_to (order-insensitive)."""
        for edge in net.edges:
            a, b = edge.cam_a.camera_id, edge.cam_b.camera_id
            if (a == cam_from and b == cam_to) or (a == cam_to and b == cam_from):
                return edge
        return None

    @staticmethod
    def _travel_time_seconds(ts_from: str, ts_to: str) -> Optional[float]:
        try:
            t0 = datetime.fromisoformat(ts_from)
            t1 = datetime.fromisoformat(ts_to)
            return (t1 - t0).total_seconds()
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _congestion_band(speed_kmh: float, limit: int) -> str:
        ratio = speed_kmh / limit if limit > 0 else 0.0
        for threshold, band in _BAND_THRESHOLDS:
            if ratio >= threshold:
                return band
        return "heavy"


# ---------------------------------------------------------------------------
# Smoke test — full pipeline: re-ID match → speed reading
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    from src.tracking.vehicle_reid import ReIDMatch

    estimator = SpeedEstimator()

    # Simulate: vehicle exits cam 1004 (CTE) at 08:30:00
    #           same vehicle enters cam 1002 (CTE, adjacent — 0.39 km apart)
    net = estimator._get_network()
    edge = estimator._find_edge(net, "1004", "1002")
    if edge:
        print(f"Edge 1004→1002: {edge.distance_km:.3f} km on {edge.road}")
        travel_s = 22   # 22s travel at ~63 km/h
        expected_speed = edge.distance_km / (travel_s / 3600)
        print(f"Expected speed ({travel_s}s travel): {expected_speed:.1f} km/h\n")
    else:
        print("Edge 1004→1002 not found in network")

    match = ReIDMatch(
        query_camera="1002",
        gallery_camera="1004",
        query_track_id=3,
        gallery_track_id=7,
        similarity=0.976,
        query_timestamp="2026-04-10T08:30:22+08:00",
        gallery_timestamp="2026-04-10T08:30:00+08:00",
        cls="car",
    )

    reading = estimator.estimate(match)
    if reading:
        print(json.dumps(reading.to_dict(), indent=2))
    else:
        print("No reading — cameras not adjacent or invalid timestamps")

    # Add a few more readings to test road_speed_profile
    for delta_s, cam_a, cam_b in [
        (95,  "1002", "1003"),
        (110, "1003", "1703"),
        (75,  "1701", "1702"),
    ]:
        m = ReIDMatch(
            query_camera=cam_b,
            gallery_camera=cam_a,
            query_track_id=1,
            gallery_track_id=1,
            similarity=0.91,
            query_timestamp=f"2026-04-10T08:32:{delta_s % 60:02d}+08:00",
            gallery_timestamp="2026-04-10T08:30:00+08:00",
            cls="car",
        )
        # Manually set timestamps to encode travel time
        from datetime import timedelta
        base = datetime.fromisoformat("2026-04-10T08:30:00+08:00")
        m.gallery_timestamp = base.isoformat()
        m.query_timestamp   = (base + timedelta(seconds=delta_s)).isoformat()
        estimator.estimate(m)

    print("\nRoad speed profile:")
    profile = estimator.road_speed_profile()
    for road, info in profile.items():
        print(f"  {road}: {info['avg_speed_kmh']} km/h  [{info['congestion_band']}]  "
              f"(limit {info['speed_limit']} km/h, {info['num_readings']} readings)")
