"""
CATI End-to-End Inference Pipeline

Wires together every component into a single callable pipeline:

    Image + camera_id + metadata
         ↓
    CATIDetector      — FiLM-conditioned YOLOv11 detection
         ↓
    SingaporeTracker  — Direction-constrained ByteTrack per camera
         ↓
    VehicleReID       — OSNet-x0.25 appearance embeddings
         ↓
    TrafficAnalytics  — Occupancy, LOS, congestion score
         ↓
    ReIDGallery       — Cross-camera vehicle matching
         ↓
    SpeedEstimator    — Inter-camera speed from GPS edge distances
         ↓
    FrameResult       — Full per-camera + network state

CATIPipeline is stateful across frames and cameras:
  - One SingaporeTracker instance per camera (preserves Kalman state)
  - One shared ReIDGallery (cross-camera matching)
  - One shared SpeedEstimator (rolling speed buffer per road segment)

Usage (Colab / server):
    from src.inference import CATIPipeline

    pipeline = CATIPipeline(
        yolo_weights="/content/drive/.../phase2/yolo_cati6/weights/best.pt",
        cati_weights="/content/drive/.../phase2/cati_phase2_final.pt",
        feature_dir="/content/drive/.../features",
        device="cuda",
    )

    # Process one frame
    import cv2
    img = cv2.imread("frame.jpg")
    result = pipeline.process_frame(
        image_bgr=img,
        camera_id="1001",
        timestamp="2026-04-10T08:30:00+08:00",
        weather="Thundery Showers",
        temperature=28.5,
    )
    print(result.traffic_state.los, result.speed_readings)

    # Draw annotated frame
    annotated = pipeline.draw(img, result)
    cv2.imwrite("annotated.jpg", annotated)

    # Network-wide summary
    summary = pipeline.network_state()
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class FrameResult:
    """Full analytics result for one camera frame."""
    camera_id:    str
    timestamp:    str
    road:         str
    region:       str
    area:         str

    # Detection
    detections:   list[dict]          # raw detection dicts
    num_vehicles: int

    # Tracking
    active_tracks: list               # confirmed Track objects
    num_active_tracks: int

    # Analytics
    traffic_state: object             # TrafficState

    # Speed readings produced this frame (from re-ID matches)
    speed_readings: list              # SpeedReading objects

    # Timing
    inference_ms: float
    pipeline_ms:  float

    def to_dict(self) -> dict:
        return {
            "camera_id":         self.camera_id,
            "timestamp":         self.timestamp,
            "road":              self.road,
            "region":            self.region,
            "area":              self.area,
            "num_vehicles":      self.num_vehicles,
            "num_active_tracks": self.num_active_tracks,
            "traffic_state":     self.traffic_state.to_dict(),
            "speed_readings":    [s.to_dict() for s in self.speed_readings],
            "inference_ms":      round(self.inference_ms, 1),
            "pipeline_ms":       round(self.pipeline_ms, 1),
        }


# ---------------------------------------------------------------------------
# CATI detector wrapper (YOLO + FiLM hooks)
# ---------------------------------------------------------------------------

class _CATIInferenceDetector:
    """
    Loads Phase 2 YOLO weights + CATI weights, registers FiLM hooks,
    and runs conditioned inference per frame.

    FiLM hooks fire on backbone layers [4, 6, 9] and neck layers [16, 19, 22]
    (same as training). Context is encoded once per frame and injected via
    the hooks into each YOLO forward pass.
    """

    BACKBONE_HOOK_LAYERS = [4, 6, 9]
    NECK_HOOK_LAYERS     = [16, 19, 22]

    def __init__(
        self,
        yolo_weights: str,
        cati_weights: str,
        feature_dir:  str,
        device:       str = "cuda",
        conf:         float = 0.25,
        use_neck_film: bool = True,
    ):
        import torch
        from ultralytics import YOLO
        from src.models.cati_detector import CATIConfig, CATIDetector, YOLO11S_NECK_CHANNEL_DIMS
        from src.training.train_phase2 import ContextLookup

        self.device   = torch.device(device if torch.cuda.is_available() else "cpu")
        self.conf     = conf
        self._handles = []
        self._ctx_vec = None

        # Load YOLO
        self._yolo = YOLO(yolo_weights)
        self._yolo.to(self.device)

        # Load CATI
        cfg = CATIConfig(
            neck_channels=YOLO11S_NECK_CHANNEL_DIMS if use_neck_film else [],
        )
        self._cati = CATIDetector(cfg)
        ckpt = torch.load(cati_weights, map_location=self.device)
        state = ckpt.get("model_state_dict", ckpt)
        self._cati.load_state_dict(state, strict=False)
        self._cati.to(self.device).eval()

        # Context lookup (image stem → metadata tensors)
        self._ctx_lookup = ContextLookup(feature_dir, self.device)

        # Register FiLM hooks
        hook_layers = self.BACKBONE_HOOK_LAYERS + (self.NECK_HOOK_LAYERS if use_neck_film else [])
        self._register_hooks(hook_layers, use_neck_film)

        logger.info(
            f"CATIInferenceDetector ready | "
            f"YOLO={Path(yolo_weights).name} | "
            f"CATI={Path(cati_weights).name} | "
            f"hooks={hook_layers}"
        )

    def _register_hooks(self, hook_layers: list[int], use_neck_film: bool):
        """Register forward hooks that apply FiLM conditioning."""
        import torch
        model_layers = list(self._yolo.model.model)
        backbone_set = set(self.BACKBONE_HOOK_LAYERS)
        neck_set     = set(self.NECK_HOOK_LAYERS) if use_neck_film else set()

        def make_hook(layer_idx: int, is_neck: bool):
            def hook(module, input, output):
                if self._ctx_vec is None:
                    return output
                try:
                    if is_neck:
                        gamma, beta = self._cati.get_neck_film_params(self._ctx_vec)
                        # neck_film_layers maps layer_idx → index in gamma list
                        neck_idx = sorted(neck_set).index(layer_idx)
                        g, b = gamma[neck_idx], beta[neck_idx]
                    else:
                        film = self._cati.film_generator(self._ctx_vec)
                        bb_idx = sorted(backbone_set).index(layer_idx)
                        n_ch = output.shape[1]
                        g = film[:, bb_idx * n_ch * 2 : bb_idx * n_ch * 2 + n_ch]
                        b = film[:, bb_idx * n_ch * 2 + n_ch : bb_idx * n_ch * 2 + 2 * n_ch]
                        g = g.view(-1, n_ch, 1, 1)
                        b = b.view(-1, n_ch, 1, 1)
                    return g * output + b
                except Exception:
                    return output
            return hook

        for i, layer in enumerate(model_layers):
            if i in backbone_set:
                h = layer.register_forward_hook(make_hook(i, is_neck=False))
                self._handles.append(h)
            elif i in neck_set:
                h = layer.register_forward_hook(make_hook(i, is_neck=True))
                self._handles.append(h)

    def run(
        self,
        image_bgr: np.ndarray,
        image_stem: str = "",
        weather: str = "unknown",
        temperature: float = 28.0,
        hour: float = 12.0,
        camera_idx: int = 0,
    ) -> tuple[list[dict], float]:
        """
        Run FiLM-conditioned detection on one frame.

        Returns:
            (detections, inference_ms)
            detections: list of dicts with cls, bbox (x1,y1,x2,y2), conf
        """
        import torch

        # Build context — use lookup if available, else use provided metadata
        meta = self._ctx_lookup.get(image_stem) if image_stem else None
        if meta is None:
            meta = {
                "weather_condition": weather,
                "temperature_celsius": temperature,
                "hour": hour,
                "camera_idx": camera_idx,
                "pm25_reading": 15.0,
                "image_width": image_bgr.shape[1],
                "image_height": image_bgr.shape[0],
            }
        ctx = self._ctx_lookup.to_tensors(meta)

        # Encode context into FiLM vector
        with torch.no_grad():
            self._ctx_vec = self._cati.encode_context(**ctx)

        # Run YOLO (FiLM hooks fire inside forward pass)
        t0 = time.perf_counter()
        results = self._yolo(
            image_bgr,
            conf=self.conf,
            verbose=False,
            device=self.device,
        )
        inference_ms = (time.perf_counter() - t0) * 1000

        self._ctx_vec = None   # clear after inference

        # Parse detections
        CLASS_NAMES = {0: "person", 1: "bicycle", 2: "car",
                       3: "motorcycle", 5: "bus", 7: "truck"}
        detections = []
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                cls_id = int(box.cls[0])
                if cls_id not in CLASS_NAMES:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                detections.append({
                    "cls":  CLASS_NAMES[cls_id],
                    "bbox": (x1, y1, x2, y2),
                    "conf": float(box.conf[0]),
                })

        return detections, inference_ms

    def close(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

class CATIPipeline:
    """
    End-to-end CATI inference pipeline for the Singapore expressway network.

    Stateful across frames:
      - Per-camera SingaporeTracker (Kalman state persists between frames)
      - Shared ReIDGallery (cross-camera embedding store)
      - Shared SpeedEstimator (rolling speed buffer per road segment)
      - CameraMap for road/region/area lookup per camera

    Args:
        yolo_weights:    Path to Phase 2 YOLO best.pt
        cati_weights:    Path to cati_phase2_final.pt
        feature_dir:     Path to feature extraction JSONs (for ContextLookup)
        device:          "cuda" or "cpu"
        conf:            Detection confidence threshold
        use_neck_film:   Enable neck FiLM conditioning (matches training config)
        reid_thresh:     Cosine similarity threshold for re-ID match
        gallery_max_age: Seconds to keep gallery entries alive
    """

    def __init__(
        self,
        yolo_weights:    str,
        cati_weights:    str,
        feature_dir:     str,
        device:          str   = "cuda",
        conf:            float = 0.25,
        use_neck_film:   bool  = True,
        reid_thresh:     float = 0.72,
        gallery_max_age: float = 180.0,
    ):
        from src.analytics.camera_map    import CameraMap
        from src.analytics.camera_network import CameraNetwork
        from src.analytics.traffic_analytics import TrafficAnalytics
        from src.analytics.speed_estimator  import SpeedEstimator
        from src.tracking.vehicle_reid      import VehicleReID, ReIDGallery

        self.device = device

        # Detection (CATI + FiLM hooks)
        self._detector = _CATIInferenceDetector(
            yolo_weights=yolo_weights,
            cati_weights=cati_weights,
            feature_dir=feature_dir,
            device=device,
            conf=conf,
            use_neck_film=use_neck_film,
        )

        # Camera topology
        self._cam_map = CameraMap()
        self._network = CameraNetwork()

        # Analytics
        self._analytics = TrafficAnalytics()
        self._speed_est = SpeedEstimator()

        # Re-ID
        self._reid    = VehicleReID(device=device)
        self._gallery = ReIDGallery(
            max_age_seconds=gallery_max_age,
            similarity_thresh=reid_thresh,
        )

        # Per-camera tracker instances (created on first frame for that camera)
        self._trackers: dict[str, object] = {}

        logger.info("CATIPipeline ready")

    def _get_tracker(self, camera_id: str):
        """Get or create a SingaporeTracker for this camera."""
        if camera_id not in self._trackers:
            from src.tracking.tracker import SingaporeTracker
            self._trackers[camera_id] = SingaporeTracker(camera_id=camera_id)
            logger.info(f"Tracker created for camera {camera_id} "
                        f"({self._trackers[camera_id].road})")
        return self._trackers[camera_id]

    def process_frame(
        self,
        image_bgr:   np.ndarray,
        camera_id:   str,
        timestamp:   str  = "",
        weather:     str  = "unknown",
        temperature: float = 28.0,
        frame_id:    int  = 0,
        image_stem:  str  = "",
    ) -> FrameResult:
        """
        Process one camera frame through the full pipeline.

        Args:
            image_bgr:   BGR frame from camera (numpy array H×W×3).
            camera_id:   LTA camera ID string (e.g. "1001").
            timestamp:   ISO 8601 timestamp.
            weather:     Current weather condition string.
            temperature: Air temperature in Celsius.
            frame_id:    Sequential frame counter for this camera.
            image_stem:  Image filename stem for ContextLookup (optional).

        Returns:
            FrameResult with detection, tracking, analytics, and speed readings.
        """
        t_start = time.perf_counter()

        # 1 — Lookup camera topology
        cam_info = self._cam_map.lookup(
            camera_id,
            lat=self._network.nodes[camera_id].lat if camera_id in self._network.nodes else 1.35,
            lon=self._network.nodes[camera_id].lon if camera_id in self._network.nodes else 103.82,
        )

        H, W = image_bgr.shape[:2]
        hour = datetime.fromisoformat(timestamp).hour if timestamp else 12.0

        # 2 — CATI detection (FiLM-conditioned)
        raw_dets, inference_ms = self._detector.run(
            image_bgr=image_bgr,
            image_stem=image_stem or Path(timestamp).stem,
            weather=weather,
            temperature=temperature,
            hour=float(hour),
            camera_idx=frame_id % 90,
        )

        # 3 — Convert to tracker Detection format
        from src.tracking.tracker import Detection as TrackerDetection
        from src.analytics.traffic_analytics import Detection as AnalyticsDetection

        tracker_dets = [
            TrackerDetection(cls=d["cls"], bbox=d["bbox"], conf=d["conf"])
            for d in raw_dets
        ]

        # 4 — Track
        tracker = self._get_tracker(camera_id)
        active_tracks = tracker.update(
            tracker_dets, frame_id=frame_id,
            frame_wh=(W, H), timestamp=timestamp,
        )

        # 5 — Extract re-ID embeddings for confirmed tracks
        for track in active_tracks:
            x1, y1, x2, y2 = track.bbox_tlbr
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(W, int(x2)), min(H, int(y2))
            if x2 > x1 + 8 and y2 > y1 + 8:
                crop = image_bgr[y1:y2, x1:x2]
                try:
                    track.embedding = self._reid.extract(crop)
                except Exception:
                    pass

        # 6 — Traffic analytics
        analytics_dets = [
            AnalyticsDetection(cls=d["cls"], bbox=d["bbox"], conf=d["conf"])
            for d in raw_dets
        ]
        traffic_state = self._analytics.compute(
            detections=analytics_dets,
            frame_wh=(W, H),
            camera_id=camera_id,
            timestamp=timestamp,
            weather=weather,
            road=cam_info.road,
            region=cam_info.region,
            area=cam_info.area,
        )

        # 7 — Update re-ID gallery with tracks that have exited the frame
        speed_readings = []
        from src.tracking.tracker import FrameEdge
        for track in active_tracks:
            if track.embedding is None:
                continue
            # Track exited via a road-direction edge — add to gallery
            if track.exit_edge != FrameEdge.INTERIOR and track.exit_timestamp:
                self._gallery.add(
                    camera_id=camera_id,
                    track_id=track.track_id,
                    embedding=track.embedding,
                    timestamp=track.exit_timestamp,
                    exit_edge=str(track.exit_edge),
                    cls=track.cls,
                )

            # Query gallery of upstream cameras for this track's entry
            if track.entry_edge != FrameEdge.INTERIOR and track.entry_timestamp:
                upstream = self._get_upstream_cameras(camera_id)
                for upstream_id in upstream:
                    match = self._gallery.query(
                        query_embedding=track.embedding,
                        query_camera_id=camera_id,
                        query_timestamp=track.entry_timestamp,
                        query_track_id=track.track_id,
                        query_cls=track.cls,
                        gallery_camera_id=upstream_id,
                    )
                    if match:
                        reading = self._speed_est.estimate(match)
                        if reading and not reading.is_outlier:
                            speed_readings.append(reading)
                            logger.info(
                                f"Speed: {reading.speed_kmh:.1f} km/h on {reading.road} "
                                f"({upstream_id}→{camera_id}, sim={match.similarity:.3f})"
                            )

        pipeline_ms = (time.perf_counter() - t_start) * 1000

        return FrameResult(
            camera_id=camera_id,
            timestamp=timestamp,
            road=cam_info.road,
            region=cam_info.region,
            area=cam_info.area,
            detections=raw_dets,
            num_vehicles=traffic_state.total_vehicles,
            active_tracks=active_tracks,
            num_active_tracks=len(active_tracks),
            traffic_state=traffic_state,
            speed_readings=speed_readings,
            inference_ms=inference_ms,
            pipeline_ms=pipeline_ms,
        )

    def _get_upstream_cameras(self, camera_id: str) -> list[str]:
        """Return camera IDs adjacent and upstream of camera_id."""
        neighbors = self._network.neighbors(camera_id)
        return [n.camera_id for n, _ in neighbors]

    def network_state(self) -> dict:
        """
        Current city-wide traffic state across all processed cameras.
        Aggregates all TrafficStates and SpeedEstimator profile.
        """
        return {
            "speed_profile": self._speed_est.road_speed_profile(),
            "gallery_stats": self._gallery.stats(),
            "active_cameras": list(self._trackers.keys()),
        }

    def draw(self, image_bgr: np.ndarray, result: FrameResult) -> np.ndarray:
        """
        Draw annotated overlay on frame — bounding boxes, track IDs,
        LOS grade, congestion score, and speed readings.

        Returns:
            Annotated BGR frame.
        """
        import cv2

        img = image_bgr.copy()
        H, W = img.shape[:2]

        CLASS_COLOURS = {
            "car":        (0, 255, 0),
            "bus":        (0, 165, 255),
            "truck":      (0, 0, 255),
            "motorcycle": (255, 255, 0),
            "bicycle":    (255, 128, 0),
            "person":     (200, 200, 200),
        }
        LOS_COLOURS = {
            "A": (0, 220, 0),
            "B": (0, 180, 60),
            "C": (0, 210, 210),
            "D": (0, 140, 255),
            "E": (0, 60, 255),
            "F": (0, 0, 220),
        }

        # Draw bounding boxes + track IDs
        for track in result.active_tracks:
            x1, y1, x2, y2 = [int(v) for v in track.bbox_tlbr]
            colour = CLASS_COLOURS.get(track.cls, (180, 180, 180))
            cv2.rectangle(img, (x1, y1), (x2, y2), colour, 2)
            label = f"#{track.track_id} {track.cls[:3]}"
            cv2.putText(img, label, (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)

        # HUD — top-left overlay
        ts = result.traffic_state
        los_col = LOS_COLOURS.get(ts.los.value, (255, 255, 255))
        hud_lines = [
            f"{result.camera_id}  {result.road}  {result.area}",
            f"Vehicles: {ts.total_vehicles}  Tracks: {result.num_active_tracks}",
            f"Occupancy: {ts.occupancy * 100:.1f}%  LOS: {ts.los.value}  ({ts.los_label})",
            f"Congestion: {ts.congestion_score:.2f}  Weather: {ts.weather}",
        ]
        for i, line in enumerate(hud_lines):
            y = 22 + i * 22
            cv2.rectangle(img, (0, y - 16), (min(W, 480), y + 6), (0, 0, 0), -1)
            col = los_col if i == 2 else (220, 220, 220)
            cv2.putText(img, line, (6, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, col, 1, cv2.LINE_AA)

        # Speed readings — bottom-left
        for i, sr in enumerate(result.speed_readings):
            y = H - 12 - i * 22
            text = f"Speed {sr.camera_from}→{sr.camera_to}: {sr.speed_kmh:.0f} km/h [{sr.congestion_band}]"
            cv2.rectangle(img, (0, y - 16), (min(W, 420), y + 4), (0, 0, 0), -1)
            cv2.putText(img, text, (6, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 200), 1, cv2.LINE_AA)

        return img

    def close(self):
        self._detector.close()
