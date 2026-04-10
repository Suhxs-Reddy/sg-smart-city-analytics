"""
ByteTrack — Multi-Object Tracker (from scratch)

Full implementation of ByteTrack (Zhang et al., 2022) for vehicle tracking:
  - Kalman filter with constant-velocity motion model
  - Hungarian algorithm (optimal assignment via scipy)
  - Two-step association: high-confidence → low-confidence detections
  - Track state machine: Tentative → Confirmed → Lost → Removed
  - Appearance embedding slots for cross-camera re-ID (vehicle_reid.py)

Reference:
  ByteTrack: Multi-Object Tracking by Associating Every Detection Box
  Zhang et al., ECCV 2022. https://arxiv.org/abs/2110.06864

Integrates with:
  traffic_analytics.py  (Detection dataclass as input)
  vehicle_reid.py       (fills track.embedding for cross-camera matching)
  inference.py          (called per-frame in the main pipeline)

Usage:
    from src.tracking.tracker import ByteTracker, Detection

    tracker = ByteTracker()
    tracks = tracker.update(detections, frame_id=1)
    for t in tracks:
        print(t.track_id, t.cls, t.bbox_tlbr, t.embedding)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

from scipy.optimize import linear_sum_assignment


# ---------------------------------------------------------------------------
# Track state
# ---------------------------------------------------------------------------

class TrackState(IntEnum):
    Tentative = 1   # Newly created — not yet confirmed
    Confirmed = 2   # Seen enough consecutive frames
    Lost      = 3   # Missed recently — kept alive via Kalman prediction
    Removed   = 4   # Expired — purged from active set


# ---------------------------------------------------------------------------
# Kalman Filter
# State vector: [cx, cy, a, h, vcx, vcy, va, vh]
#   cx, cy = bounding box center
#   a      = aspect ratio (w/h)
#   h      = height
#   v*     = velocities (constant-velocity model)
# Observation: [cx, cy, a, h]
# ---------------------------------------------------------------------------

class KalmanFilter:
    """
    Kalman filter for bounding box tracking.
    State: [cx, cy, a, h, vcx, vcy, va, vh]
    Observation: [cx, cy, a, h]
    """

    ndim = 4        # observation dimensions
    dt   = 1.0      # time step (one frame)

    def __init__(self):
        # State transition matrix (constant velocity)
        self.F = np.eye(2 * self.ndim, dtype=np.float32)
        for i in range(self.ndim):
            self.F[i, self.ndim + i] = self.dt

        # Observation matrix
        self.H = np.eye(self.ndim, 2 * self.ndim, dtype=np.float32)

        # Process noise weights — position uncertainty grows with object size
        self._std_weight_pos = 1.0 / 20.0
        self._std_weight_vel = 1.0 / 160.0

    def initiate(self, bbox: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Create initial state from a measurement [cx, cy, a, h].
        Returns (mean, covariance).
        """
        mean_pos = bbox.copy()
        mean_vel = np.zeros_like(mean_pos)
        mean = np.concatenate([mean_pos, mean_vel])

        std = [
            2 * self._std_weight_pos * bbox[3],
            2 * self._std_weight_pos * bbox[3],
            1e-2,
            2 * self._std_weight_pos * bbox[3],
            10 * self._std_weight_vel * bbox[3],
            10 * self._std_weight_vel * bbox[3],
            1e-5,
            10 * self._std_weight_vel * bbox[3],
        ]
        covariance = np.diag(np.square(std, dtype=np.float32))
        return mean, covariance

    def predict(
        self, mean: np.ndarray, covariance: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        std_pos = [
            self._std_weight_pos * mean[3],
            self._std_weight_pos * mean[3],
            1e-2,
            self._std_weight_pos * mean[3],
        ]
        std_vel = [
            self._std_weight_vel * mean[3],
            self._std_weight_vel * mean[3],
            1e-5,
            self._std_weight_vel * mean[3],
        ]
        Q = np.diag(np.square(std_pos + std_vel, dtype=np.float32))

        mean = self.F @ mean
        covariance = self.F @ covariance @ self.F.T + Q
        return mean, covariance

    def update(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurement: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        std = [
            self._std_weight_pos * mean[3],
            self._std_weight_pos * mean[3],
            1e-1,
            self._std_weight_pos * mean[3],
        ]
        R = np.diag(np.square(std, dtype=np.float32))

        S = self.H @ covariance @ self.H.T + R
        K = covariance @ self.H.T @ np.linalg.inv(S)
        innovation = measurement - self.H @ mean

        mean = mean + K @ innovation
        covariance = (np.eye(2 * self.ndim) - K @ self.H) @ covariance
        return mean, covariance

    def project(
        self, mean: np.ndarray, covariance: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        std = [
            self._std_weight_pos * mean[3],
            self._std_weight_pos * mean[3],
            1e-1,
            self._std_weight_pos * mean[3],
        ]
        R = np.diag(np.square(std, dtype=np.float32))
        projected_mean = self.H @ mean
        projected_cov  = self.H @ covariance @ self.H.T + R
        return projected_mean, projected_cov


# ---------------------------------------------------------------------------
# Detection (input to tracker)
# Compatible with traffic_analytics.Detection — mirrors its fields.
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    cls:  str
    bbox: tuple[float, float, float, float]   # (x1, y1, x2, y2) pixels
    conf: float
    embedding: Optional[np.ndarray] = None    # from vehicle_reid.py if available

    def to_tlwh(self) -> np.ndarray:
        x1, y1, x2, y2 = self.bbox
        return np.array([x1, y1, x2 - x1, y2 - y1], dtype=np.float32)

    def to_cxcyah(self) -> np.ndarray:
        """Center-x, center-y, aspect ratio, height."""
        x1, y1, x2, y2 = self.bbox
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w / 2
        cy = y1 + h / 2
        a  = w / max(h, 1e-6)
        return np.array([cx, cy, a, h], dtype=np.float32)


# ---------------------------------------------------------------------------
# Track
# ---------------------------------------------------------------------------

_track_counter = 0

@dataclass
class Track:
    """Single vehicle track with Kalman state and metadata."""

    track_id:  int
    cls:       str
    state:     TrackState

    # Kalman state
    mean:       np.ndarray
    covariance: np.ndarray

    # Lifecycle counters
    hits:           int = 1    # consecutive matched frames
    age:            int = 1    # total frames since creation
    time_since_update: int = 0

    # Confidence
    score: float = 0.0

    # Appearance embedding (for cross-camera re-ID)
    embedding: Optional[np.ndarray] = None

    @property
    def bbox_tlbr(self) -> tuple[float, float, float, float]:
        """Return (x1, y1, x2, y2) from Kalman state."""
        cx, cy, a, h = self.mean[:4]
        w  = a * h
        x1 = cx - w / 2
        y1 = cy - h / 2
        return float(x1), float(y1), float(x1 + w), float(y1 + h)

    @property
    def bbox_cxcyah(self) -> np.ndarray:
        return self.mean[:4].copy()

    def predict(self, kf: KalmanFilter):
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, kf: KalmanFilter, det: Detection):
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, det.to_cxcyah()
        )
        self.hits += 1
        self.time_since_update = 0
        self.score = det.conf
        self.cls   = det.cls
        if det.embedding is not None:
            # Exponential moving average of appearance embedding
            alpha = 0.9
            if self.embedding is None:
                self.embedding = det.embedding.copy()
            else:
                self.embedding = alpha * self.embedding + (1 - alpha) * det.embedding
                self.embedding /= np.linalg.norm(self.embedding) + 1e-6

    def mark_missed(self):
        if self.state == TrackState.Tentative:
            self.state = TrackState.Removed
        elif self.time_since_update > 0:
            self.state = TrackState.Lost


# ---------------------------------------------------------------------------
# IoU utilities
# ---------------------------------------------------------------------------

def _iou_matrix(
    tracks: list[Track], detections: list[Detection]
) -> np.ndarray:
    """Compute IoU cost matrix [n_tracks × n_dets]. Cost = 1 - IoU."""
    n, m = len(tracks), len(detections)
    cost = np.ones((n, m), dtype=np.float32)
    for i, t in enumerate(tracks):
        tx1, ty1, tx2, ty2 = t.bbox_tlbr
        for j, d in enumerate(detections):
            dx1, dy1, dx2, dy2 = d.bbox
            ix1 = max(tx1, dx1); iy1 = max(ty1, dy1)
            ix2 = min(tx2, dx2); iy2 = min(ty2, dy2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            union = (
                (tx2 - tx1) * (ty2 - ty1)
                + (dx2 - dx1) * (dy2 - dy1)
                - inter
            )
            cost[i, j] = 1.0 - (inter / union if union > 0 else 0.0)
    return cost


def _hungarian(cost: np.ndarray, thresh: float):
    """
    Run Hungarian algorithm on cost matrix.
    Returns matched (row, col) pairs where cost < thresh,
    unmatched row indices, unmatched col indices.
    """
    if cost.size == 0:
        return [], list(range(cost.shape[0])), list(range(cost.shape[1]))

    rows, cols = linear_sum_assignment(cost)
    matched, unmatched_r, unmatched_c = [], [], []

    matched_set_r, matched_set_c = set(), set()
    for r, c in zip(rows, cols):
        if cost[r, c] < thresh:
            matched.append((r, c))
            matched_set_r.add(r)
            matched_set_c.add(c)

    unmatched_r = [i for i in range(cost.shape[0]) if i not in matched_set_r]
    unmatched_c = [j for j in range(cost.shape[1]) if j not in matched_set_c]
    return matched, unmatched_r, unmatched_c


# ---------------------------------------------------------------------------
# ByteTracker
# ---------------------------------------------------------------------------

class ByteTracker:
    """
    ByteTrack multi-object tracker.

    Two-step association:
      Step 1 — high-confidence detections matched to confirmed + lost tracks
      Step 2 — low-confidence detections matched to remaining unmatched tracks

    Args:
        track_thresh:     Detection confidence threshold for high-confidence pool.
        track_buffer:     Frames to keep a Lost track alive before removing.
        match_thresh:     IoU cost threshold for a valid match (1 - IoU < thresh).
        min_hits:         Consecutive hits before Tentative → Confirmed.
        low_conf_thresh:  Minimum confidence for low-confidence pool.
    """

    def __init__(
        self,
        track_thresh:    float = 0.45,
        track_buffer:    int   = 30,
        match_thresh:    float = 0.80,
        min_hits:        int   = 3,
        low_conf_thresh: float = 0.10,
    ):
        self.track_thresh    = track_thresh
        self.track_buffer    = track_buffer
        self.match_thresh    = match_thresh
        self.min_hits        = min_hits
        self.low_conf_thresh = low_conf_thresh

        self._kf: KalmanFilter = KalmanFilter()
        self._tracks: list[Track] = []
        self._frame_id: int = 0
        self._next_id: int  = 1

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, detections: list[Detection], frame_id: int | None = None) -> list[Track]:
        """
        Process one frame of detections.

        Args:
            detections: All detections for this frame (any confidence).
            frame_id:   Optional frame counter for logging.

        Returns:
            List of currently active (Confirmed) tracks.
        """
        self._frame_id = frame_id if frame_id is not None else self._frame_id + 1

        # Split detections into high / low confidence pools
        high_dets = [d for d in detections if d.conf >= self.track_thresh]
        low_dets  = [d for d in detections if self.low_conf_thresh <= d.conf < self.track_thresh]

        # Predict all existing tracks forward one step
        for t in self._tracks:
            t.predict(self._kf)

        # Active tracks = Confirmed + Lost (both eligible for matching)
        confirmed = [t for t in self._tracks if t.state == TrackState.Confirmed]
        lost      = [t for t in self._tracks if t.state == TrackState.Lost]
        tentative = [t for t in self._tracks if t.state == TrackState.Tentative]

        # ── Step 1: match high-confidence dets → confirmed + lost tracks ──
        all_active = confirmed + lost
        cost1 = _iou_matrix(all_active, high_dets)
        matched1, unmatched_tracks1, unmatched_dets1 = _hungarian(cost1, self.match_thresh)

        for ti, di in matched1:
            all_active[ti].update(self._kf, high_dets[di])
            if all_active[ti].state == TrackState.Lost:
                all_active[ti].state = TrackState.Confirmed

        unmatched_active = [all_active[i] for i in unmatched_tracks1]

        # ── Step 2: match low-confidence dets → remaining unmatched tracks ──
        cost2 = _iou_matrix(unmatched_active, low_dets)
        matched2, unmatched_tracks2, _ = _hungarian(cost2, self.match_thresh)

        for ti, di in matched2:
            unmatched_active[ti].update(self._kf, low_dets[di])
            if unmatched_active[ti].state == TrackState.Lost:
                unmatched_active[ti].state = TrackState.Confirmed

        # Mark truly unmatched tracks as missed
        for i in unmatched_tracks2:
            unmatched_active[i].mark_missed()

        # ── Step 3: match remaining high-conf dets → tentative tracks ──
        remaining_high = [high_dets[i] for i in unmatched_dets1]
        cost3 = _iou_matrix(tentative, remaining_high)
        matched3, unmatched_tent, unmatched_new = _hungarian(cost3, self.match_thresh)

        for ti, di in matched3:
            tentative[ti].update(self._kf, remaining_high[di])

        for i in unmatched_tent:
            tentative[i].mark_missed()

        # ── Promote tentative → confirmed ──
        for t in tentative:
            if t.hits >= self.min_hits and t.state == TrackState.Tentative:
                t.state = TrackState.Confirmed

        # ── Initialize new tracks for unmatched high-conf dets ──
        for i in unmatched_new:
            det = remaining_high[i]
            mean, cov = self._kf.initiate(det.to_cxcyah())
            new_track = Track(
                track_id=self._next_id,
                cls=det.cls,
                state=TrackState.Tentative,
                mean=mean,
                covariance=cov,
                score=det.conf,
                embedding=det.embedding.copy() if det.embedding is not None else None,
            )
            self._next_id += 1
            self._tracks.append(new_track)

        # ── Remove expired lost tracks ──
        self._tracks = [
            t for t in self._tracks
            if not (
                t.state == TrackState.Removed
                or (t.state == TrackState.Lost and t.time_since_update > self.track_buffer)
            )
        ]

        return [t for t in self._tracks if t.state == TrackState.Confirmed]

    def reset(self):
        """Reset tracker state between camera sequences."""
        self._tracks = []
        self._frame_id = 0
        self._next_id  = 1

    @property
    def active_tracks(self) -> list[Track]:
        return [t for t in self._tracks if t.state == TrackState.Confirmed]


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random
    random.seed(42)
    np.random.seed(42)

    tracker = ByteTracker()

    print("Simulating 10 frames with 3 vehicles...")
    print(f"{'Frame':<6} {'Active tracks':<14} {'Track IDs'}")
    print("-" * 40)

    # Simulate 3 vehicles moving across the frame
    vehicles = [
        {"x": 100.0, "y": 200.0, "w": 80.0, "h": 50.0, "vx": 5.0, "vy": 0.0, "cls": "car"},
        {"x": 400.0, "y": 300.0, "w": 120.0, "h": 70.0, "vx": 8.0, "vy": 1.0, "cls": "bus"},
        {"x": 700.0, "y": 250.0, "w": 60.0, "h": 40.0, "vx": 6.0, "vy": -1.0, "cls": "motorcycle"},
    ]

    for frame in range(1, 11):
        dets = []
        for v in vehicles:
            # Add slight noise to simulate imperfect detection
            nx = v["x"] + random.gauss(0, 2)
            ny = v["y"] + random.gauss(0, 2)
            dets.append(Detection(
                cls=v["cls"],
                bbox=(nx, ny, nx + v["w"], ny + v["h"]),
                conf=random.uniform(0.6, 0.95),
            ))
            v["x"] += v["vx"]
            v["y"] += v["vy"]

        active = tracker.update(dets, frame_id=frame)
        ids = [f"{t.track_id}({t.cls[:3]})" for t in active]
        print(f"{frame:<6} {len(active):<14} {', '.join(ids) if ids else '—'}")
