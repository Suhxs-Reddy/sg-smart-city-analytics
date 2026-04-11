"""
ByteTrack — Singapore Expressway Multi-Object Tracker (from scratch)

Full ByteTrack (Zhang et al., ECCV 2022) implementation tailored to the
Singapore LTA expressway camera network:

  Core tracker:
  - Kalman filter with constant-velocity motion model
  - Hungarian algorithm (scipy) for optimal assignment
  - Two-step association: high-conf → low-conf detections
  - Track state machine: Tentative → Confirmed → Lost → Removed
  - EMA appearance embedding slots for cross-camera re-ID

  Singapore-specific extensions:
  - Direction-constrained Kalman: process noise is asymmetric per road axis.
    N-S roads (CTE, BKE, KJE) strongly resist horizontal drift;
    E-W roads (PIE, AYE, ECP, TPE) resist vertical drift.
    This matches the physical constraint that expressway vehicles
    can only move along the road, not across it.
  - Entry/exit zone tagging: each track records which frame edge it
    entered and exited from (LEFT/RIGHT for E-W roads, TOP/BOTTOM for
    N-S roads). Used by speed_estimator.py to timestamp inter-camera
    vehicle handoffs.
  - SingaporeTracker: high-level wrapper that looks up road direction
    from CameraNetwork by camera_id and configures ByteTracker
    automatically.

References:
  ByteTrack: Zhang et al., ECCV 2022. https://arxiv.org/abs/2110.06864
  LTA camera network: data.gov.sg/v1/transport/traffic-images

Integrates with:
  camera_network.py     (road direction lookup per camera_id)
  traffic_analytics.py  (Detection dataclass as input)
  vehicle_reid.py       (fills track.embedding for cross-camera matching)
  speed_estimator.py    (consumes track.entry_edge / exit_edge + timestamps)
  inference.py          (called per-frame in the main pipeline)

Usage:
    from src.tracking.tracker import SingaporeTracker, Detection

    tracker = SingaporeTracker(camera_id="1001")   # auto-configures for CTE
    tracks = tracker.update(detections, frame_id=1, frame_wh=(1920, 1080))
    for t in tracks:
        print(t.track_id, t.cls, t.entry_edge, t.exit_edge, t.embedding)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, auto

import numpy as np
from scipy.optimize import linear_sum_assignment

# ---------------------------------------------------------------------------
# Road motion axis — which pixel axis vehicles move along
# ---------------------------------------------------------------------------


class RoadAxis(IntEnum):
    HORIZONTAL = auto()  # E-W roads: PIE, AYE, ECP, TPE, SLE, MCE
    VERTICAL = auto()  # N-S roads: CTE, BKE, KJE, KPE, NSC
    UNKNOWN = auto()  # Fallback — no constraint applied


_ROAD_AXIS: dict[str, RoadAxis] = {
    "PIE": RoadAxis.HORIZONTAL,
    "AYE": RoadAxis.HORIZONTAL,
    "ECP": RoadAxis.HORIZONTAL,
    "TPE": RoadAxis.HORIZONTAL,
    "SLE": RoadAxis.HORIZONTAL,
    "MCE": RoadAxis.HORIZONTAL,
    "CTE": RoadAxis.VERTICAL,
    "BKE": RoadAxis.VERTICAL,
    "KJE": RoadAxis.VERTICAL,
    "KPE": RoadAxis.VERTICAL,
    "NSC": RoadAxis.VERTICAL,
}

# Frame edge zones — a track is "at an edge" if its bbox overlaps
# within EDGE_ZONE_FRACTION of the frame width/height
EDGE_ZONE_FRACTION = 0.12


class FrameEdge(IntEnum):
    LEFT = auto()
    RIGHT = auto()
    TOP = auto()
    BOTTOM = auto()
    INTERIOR = auto()  # Not near any edge

    def __str__(self):
        return self.name


def _detect_edge(
    bbox: tuple[float, float, float, float],
    frame_wh: tuple[int, int],
    axis: RoadAxis,
) -> FrameEdge:
    """Return which frame edge a bbox is touching, constrained to road axis."""
    x1, y1, x2, y2 = bbox
    W, H = frame_wh
    z = EDGE_ZONE_FRACTION

    if axis == RoadAxis.HORIZONTAL:
        if x1 < W * z:
            return FrameEdge.LEFT
        if x2 > W * (1 - z):
            return FrameEdge.RIGHT
    elif axis == RoadAxis.VERTICAL:
        if y1 < H * z:
            return FrameEdge.TOP
        if y2 > H * (1 - z):
            return FrameEdge.BOTTOM
    else:
        # No axis constraint — check all edges
        if x1 < W * z:
            return FrameEdge.LEFT
        if x2 > W * (1 - z):
            return FrameEdge.RIGHT
        if y1 < H * z:
            return FrameEdge.TOP
        if y2 > H * (1 - z):
            return FrameEdge.BOTTOM
    return FrameEdge.INTERIOR


# ---------------------------------------------------------------------------
# Track state
# ---------------------------------------------------------------------------


class TrackState(IntEnum):
    Tentative = 1  # Newly created — not yet confirmed
    Confirmed = 2  # Seen enough consecutive frames
    Lost = 3  # Missed recently — kept alive via Kalman prediction
    Removed = 4  # Expired — purged from active set


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
    Direction-constrained Kalman filter for Singapore expressway tracking.
    State: [cx, cy, a, h, vcx, vcy, va, vh]
    Observation: [cx, cy, a, h]

    Two Singapore-specific extensions beyond vanilla ByteTrack:

    1. Direction-constrained Q (road axis):
       Process noise matrix Q is asymmetric per road axis.
       - HORIZONTAL roads (PIE/AYE/ECP): vcy noise reduced 10× (vehicles
         physically cannot move far perpendicular to the road in pixel space).
       - VERTICAL roads (CTE/BKE/KJE): vcx noise reduced 10×.

    2. NSA (Noise Scale Adaptive) measurement noise (StrongSORT, Du et al. 2022):
       Measurement noise R is scaled by (1 - conf)^2 so high-confidence
       detections dominate the Kalman update (gain K → 1) while low-confidence
       detections (partial occlusion under gantries) defer to the Kalman
       prediction. This reduces jitter on clean frames and prevents a single
       noisy reading from corrupting the track state.
    """

    ndim = 4  # observation dimensions
    dt = 1.0  # time step (one frame)

    # How much to suppress perpendicular velocity noise (higher = stronger constraint)
    PERP_SUPPRESSION = 10.0

    def __init__(self, axis: RoadAxis = RoadAxis.UNKNOWN):
        self.axis = axis

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

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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

        # Direction constraint: suppress perpendicular velocity noise.
        # Index 4=vcx, 5=vcy in the [pos(4), vel(4)] layout.
        if self.axis == RoadAxis.HORIZONTAL:
            std_vel[1] /= self.PERP_SUPPRESSION  # suppress vcy
        elif self.axis == RoadAxis.VERTICAL:
            std_vel[0] /= self.PERP_SUPPRESSION  # suppress vcx

        Q = np.diag(np.square(std_pos + std_vel, dtype=np.float32))

        mean = self.F @ mean
        covariance = self.F @ covariance @ self.F.T + Q
        return mean, covariance

    def update(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurement: np.ndarray,
        confidence: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        NSA Kalman update (StrongSORT, Du et al. 2022).
        R is scaled by (1 - confidence)^2 — high-confidence detections reduce
        measurement noise so the filter trusts the detection over the prior.
        Floor at 0.05 prevents numerical instability at conf ≈ 1.0.
        """
        std = [
            self._std_weight_pos * mean[3],
            self._std_weight_pos * mean[3],
            1e-1,
            self._std_weight_pos * mean[3],
        ]
        # NSA: confidence-adaptive measurement noise
        nsa_scale = max((1.0 - confidence) ** 2, 0.05)
        R = np.diag(np.square([s * nsa_scale for s in std], dtype=np.float32))

        S = self.H @ covariance @ self.H.T + R
        K = covariance @ self.H.T @ np.linalg.inv(S)
        innovation = measurement - self.H @ mean

        mean = mean + K @ innovation
        covariance = (np.eye(2 * self.ndim) - K @ self.H) @ covariance
        return mean, covariance

    def project(self, mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        std = [
            self._std_weight_pos * mean[3],
            self._std_weight_pos * mean[3],
            1e-1,
            self._std_weight_pos * mean[3],
        ]
        R = np.diag(np.square(std, dtype=np.float32))
        projected_mean = self.H @ mean
        projected_cov = self.H @ covariance @ self.H.T + R
        return projected_mean, projected_cov


# ---------------------------------------------------------------------------
# Detection (input to tracker)
# Compatible with traffic_analytics.Detection — mirrors its fields.
# ---------------------------------------------------------------------------


@dataclass
class Detection:
    cls: str
    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2) pixels
    conf: float
    embedding: np.ndarray | None = None  # from vehicle_reid.py if available

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
        a = w / max(h, 1e-6)
        return np.array([cx, cy, a, h], dtype=np.float32)


# ---------------------------------------------------------------------------
# Track
# ---------------------------------------------------------------------------

_track_counter = 0


@dataclass
class Track:
    """Single vehicle track with Kalman state, edge events, and re-ID embedding."""

    track_id: int
    cls: str
    state: TrackState

    # Kalman state
    mean: np.ndarray
    covariance: np.ndarray

    # Lifecycle counters
    hits: int = 1
    age: int = 1
    time_since_update: int = 0

    # Confidence
    score: float = 0.0

    # Appearance embedding (updated as EMA each frame, consumed by vehicle_reid.py)
    embedding: np.ndarray | None = None

    # Entry / exit zone events — set by SingaporeTracker on first/last detection.
    # Used by speed_estimator.py to timestamp cross-camera handoffs.
    entry_edge: FrameEdge = FrameEdge.INTERIOR
    exit_edge: FrameEdge = FrameEdge.INTERIOR
    entry_frame_id: int = 0
    exit_frame_id: int = 0
    entry_timestamp: str = ""  # ISO 8601 — set by pipeline
    exit_timestamp: str = ""  # ISO 8601 — set by pipeline

    @property
    def bbox_tlbr(self) -> tuple[float, float, float, float]:
        """Return (x1, y1, x2, y2) from Kalman state."""
        cx, cy, a, h = self.mean[:4]
        w = a * h
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
        # NSA Kalman: pass detection confidence so R is scaled adaptively
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, det.to_cxcyah(), confidence=det.conf
        )
        self.hits += 1
        self.time_since_update = 0
        self.score = det.conf
        self.cls = det.cls
        if det.embedding is not None:
            # Confidence-weighted EMA (StrongSORT / BoT-SORT):
            # High-confidence frame → smaller alpha → more weight on new embedding.
            # Low-confidence frame → larger alpha → preserve accumulated gallery.
            # alpha = max(0.7, 1.0 - 0.3 * conf): range [0.7, 1.0)
            alpha = max(0.70, 1.0 - 0.30 * det.conf)
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


def _giou_matrix(tracks: list[Track], detections: list[Detection]) -> np.ndarray:
    """
    Generalised IoU cost matrix [n_tracks × n_dets]. Cost = 1 - GIoU.

    GIoU (Rezatofighi et al., CVPR 2019):
        GIoU = IoU - (C_area - union) / C_area
    where C_area is the area of the smallest enclosing box of the two
    rectangles. GIoU ∈ (-1, 1] and is strictly negative when boxes do not
    overlap, providing a continuous proximity signal even at zero IoU.
    This prevents the Hungarian algorithm from treating all non-overlapping
    pairs as equally bad — critical for fast-moving expressway vehicles where
    40-60px/frame displacement can reduce IoU to zero between frames.
    """
    n, m = len(tracks), len(detections)
    cost = np.ones((n, m), dtype=np.float32)
    for i, t in enumerate(tracks):
        tx1, ty1, tx2, ty2 = t.bbox_tlbr
        t_area = max(0, tx2 - tx1) * max(0, ty2 - ty1)
        for j, d in enumerate(detections):
            dx1, dy1, dx2, dy2 = d.bbox
            d_area = max(0, dx2 - dx1) * max(0, dy2 - dy1)

            # Intersection
            ix1, iy1 = max(tx1, dx1), max(ty1, dy1)
            ix2, iy2 = min(tx2, dx2), min(ty2, dy2)
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            union = t_area + d_area - inter

            iou = inter / union if union > 1e-6 else 0.0

            # Smallest enclosing box
            cx1, cy1 = min(tx1, dx1), min(ty1, dy1)
            cx2, cy2 = max(tx2, dx2), max(ty2, dy2)
            c_area = max(0.0, cx2 - cx1) * max(0.0, cy2 - cy1)

            giou = iou - (c_area - union) / c_area if c_area > 1e-6 else iou
            cost[i, j] = 1.0 - giou  # GIoU cost ∈ [0, 2]

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
    for r, c in zip(rows, cols, strict=False):
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
        axis:             Road motion axis for direction-constrained Kalman filter.
    """

    def __init__(
        self,
        track_thresh: float = 0.45,
        track_buffer: int = 30,
        match_thresh: float = 0.80,
        min_hits: int = 2,
        low_conf_thresh: float = 0.10,
        axis: RoadAxis = RoadAxis.UNKNOWN,
    ):
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.min_hits = min_hits
        self.low_conf_thresh = low_conf_thresh

        self._kf: KalmanFilter = KalmanFilter(axis=axis)
        self._tracks: list[Track] = []
        self._frame_id: int = 0
        self._next_id: int = 1

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
        low_dets = [d for d in detections if self.low_conf_thresh <= d.conf < self.track_thresh]

        # Predict all existing tracks forward one step
        for t in self._tracks:
            t.predict(self._kf)

        # Active tracks = Confirmed + Lost (both eligible for matching)
        confirmed = [t for t in self._tracks if t.state == TrackState.Confirmed]
        lost = [t for t in self._tracks if t.state == TrackState.Lost]
        tentative = [t for t in self._tracks if t.state == TrackState.Tentative]

        # ── Step 1: match high-confidence dets → confirmed + lost tracks ──
        all_active = confirmed + lost
        cost1 = _giou_matrix(all_active, high_dets)
        matched1, unmatched_tracks1, unmatched_dets1 = _hungarian(cost1, self.match_thresh)

        for ti, di in matched1:
            all_active[ti].update(self._kf, high_dets[di])
            if all_active[ti].state == TrackState.Lost:
                all_active[ti].state = TrackState.Confirmed

        unmatched_active = [all_active[i] for i in unmatched_tracks1]

        # ── Step 2: match low-confidence dets → remaining unmatched tracks ──
        cost2 = _giou_matrix(unmatched_active, low_dets)
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
        cost3 = _giou_matrix(tentative, remaining_high)
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
            t
            for t in self._tracks
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
        self._next_id = 1

    @property
    def active_tracks(self) -> list[Track]:
        return [t for t in self._tracks if t.state == TrackState.Confirmed]


# ---------------------------------------------------------------------------
# Singapore-aware tracker — wraps ByteTracker with camera context
# ---------------------------------------------------------------------------


class SingaporeTracker:
    """
    High-level tracker tailored to the Singapore LTA expressway network.

    Automatically configures ByteTracker with the correct road axis and
    handles entry/exit zone tagging for inter-camera speed estimation.

    Args:
        camera_id:  LTA camera ID (e.g. "1001"). Used to look up road/axis
                    from CameraNetwork.
        **kwargs:   Forwarded to ByteTracker (track_thresh, match_thresh, etc.)

    Example:
        tracker = SingaporeTracker(camera_id="4701")  # PIE → HORIZONTAL axis
        for frame_id, (dets, timestamp) in enumerate(frames):
            tracks = tracker.update(dets, frame_id, frame_wh=(1920, 1080),
                                    timestamp=timestamp)
            # tracks[i].entry_edge, .exit_edge are set automatically
    """

    def __init__(self, camera_id: str = "", **kwargs):
        self.camera_id = camera_id
        self.road = "Unknown"
        self.axis = RoadAxis.UNKNOWN

        if camera_id:
            self._resolve_camera(camera_id)

        self._tracker = ByteTracker(axis=self.axis, **kwargs)

    def _resolve_camera(self, camera_id: str):
        """Look up road and axis from CameraNetwork."""
        try:
            from src.analytics.camera_network import CameraNetwork

            net = CameraNetwork()
            node = net.nodes.get(camera_id)
            if node:
                self.road = node.road
                self.axis = _ROAD_AXIS.get(node.road, RoadAxis.UNKNOWN)
        except Exception:
            pass  # Graceful degradation — no axis constraint applied

    def update(
        self,
        detections: list[Detection],
        frame_id: int,
        frame_wh: tuple[int, int] = (1920, 1080),
        timestamp: str = "",
    ) -> list[Track]:
        """
        Process one frame. Extends ByteTracker.update() with:
          - Entry edge tagging on first confirmation
          - Exit edge tagging when a track is about to be removed
        """
        tracks = self._tracker.update(detections, frame_id)

        for t in tracks:
            bbox = t.bbox_tlbr
            edge = _detect_edge(bbox, frame_wh, self.axis)

            # Tag entry edge on first confirmed frame
            if t.hits == self._tracker.min_hits:
                t.entry_edge = edge
                t.entry_frame_id = frame_id
                t.entry_timestamp = timestamp

            # Continuously update exit edge — last observed value persists
            if edge != FrameEdge.INTERIOR:
                t.exit_edge = edge
                t.exit_frame_id = frame_id
                t.exit_timestamp = timestamp

        return tracks

    def reset(self):
        self._tracker.reset()

    @property
    def road_info(self) -> dict:
        return {"camera_id": self.camera_id, "road": self.road, "axis": self.axis.name}


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random

    random.seed(42)
    np.random.seed(42)

    # Test on camera 4701 (PIE — HORIZONTAL axis)
    tracker = SingaporeTracker(camera_id="4701")
    print(f"Camera: {tracker.camera_id}  Road: {tracker.road}  Axis: {tracker.axis.name}")
    print()
    print("Simulating 10 frames — 3 vehicles moving left→right (PIE eastbound):")
    print(f"{'Frame':<6} {'Tracks':<8} {'IDs & edges'}")
    print("-" * 60)

    W, H = 1920, 1080
    # Vehicles move ~40-60px/frame (realistic for expressway cameras at 640→1920 scale)
    # vx must be < bbox width to maintain IoU > 0 between consecutive frames
    vehicles = [
        {"x": 50.0, "y": 400.0, "w": 120.0, "h": 70.0, "vx": 50.0, "vy": 0.5, "cls": "car"},
        {"x": 250.0, "y": 500.0, "w": 150.0, "h": 90.0, "vx": 40.0, "vy": -0.5, "cls": "bus"},
        {"x": 100.0, "y": 350.0, "w": 80.0, "h": 50.0, "vx": 60.0, "vy": 0.3, "cls": "motorcycle"},
    ]

    for frame in range(1, 11):
        dets = []
        for v in vehicles:
            if 0 < v["x"] < W:
                nx = v["x"] + random.gauss(0, 3)
                ny = v["y"] + random.gauss(0, 2)
                dets.append(
                    Detection(
                        cls=v["cls"],
                        bbox=(nx, ny, nx + v["w"], ny + v["h"]),
                        conf=random.uniform(0.6, 0.95),
                    )
                )
            v["x"] += v["vx"]
            v["y"] += v["vy"]

        active = tracker.update(
            dets, frame_id=frame, frame_wh=(W, H), timestamp=f"2026-04-09T08:{frame:02d}:00+08:00"
        )
        info = [
            f"{t.track_id}({t.cls[:3]}) entry={t.entry_edge} exit={t.exit_edge}" for t in active
        ]
        print(f"{frame:<6} {len(active):<8} {', '.join(info) if info else '—'}")
