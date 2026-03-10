"""
Singapore Smart City — REST API Server

FastAPI backend serving live analytics:
- Camera statuses and latest detections
- Congestion scores per camera
- Fleet-wide failure and drift reports
- Prediction results
- WebSocket feed for real-time updates

Designed to run on Azure App Service (free tier) or locally.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Singapore Smart City Analytics",
    description="Real-time traffic analytics from 90 LTA cameras",
    version="1.0.0",
)

# CORS — allow dashboard frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Data Store (in-memory for MVP, swap to Redis/DB for production)
# =============================================================================


class AnalyticsStore:
    """In-memory store for latest analytics results."""

    def __init__(self):
        self.camera_metadata: dict = {}  # camera_id → location, resolution
        self.latest_detections: dict = {}  # camera_id → DetectionResult
        self.latest_tracking: dict = {}  # camera_id → TrackingResult
        self.congestion_scores: dict = {}  # camera_id → congestion dict
        self.failure_reports: dict = {}  # camera_id → FailureReport
        self.drift_alerts: list = []  # Recent drift alerts
        self.fleet_report: dict = {}  # Latest fleet report
        self.predictions: dict = {}  # camera_id → predicted counts
        self.last_updated: str | None = None

    def update_detection(self, camera_id: str, result: dict):
        self.latest_detections[camera_id] = result
        self.last_updated = datetime.now().isoformat()

    def update_congestion(self, camera_id: str, score: dict):
        self.congestion_scores[camera_id] = score
        self.last_updated = datetime.now().isoformat()

    def get_all_cameras_summary(self) -> list:
        """Get summary for all cameras — designed for map overlay."""
        summaries = []
        for cam_id, meta in self.camera_metadata.items():
            detection = self.latest_detections.get(cam_id, {})
            congestion = self.congestion_scores.get(cam_id, {})
            failure = self.failure_reports.get(cam_id, {})

            summaries.append(
                {
                    "camera_id": cam_id,
                    "latitude": meta.get("latitude", 0),
                    "longitude": meta.get("longitude", 0),
                    "resolution": f"{meta.get('width', '?')}x{meta.get('height', '?')}",
                    "num_vehicles": detection.get("num_vehicles", 0),
                    "congestion_level": congestion.get("level", "unknown"),
                    "congestion_score": congestion.get("score", 0),
                    "reliability_score": failure.get("reliability_score", 1.0),
                    "failure_flags": failure.get("failure_flags", []),
                }
            )

        return summaries


# Global store
store = AnalyticsStore()


# =============================================================================
# API Endpoints
# =============================================================================


@app.get("/")
async def root():
    return {
        "service": "Singapore Smart City Analytics API",
        "version": "1.0.0",
        "cameras": len(store.camera_metadata),
        "last_updated": store.last_updated,
    }


@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "cameras_tracked": len(store.camera_metadata),
        "detections_loaded": len(store.latest_detections),
        "last_updated": store.last_updated,
    }


@app.get("/api/cameras")
async def get_all_cameras():
    """Get summary for all cameras — used for map overlay."""
    return store.get_all_cameras_summary()


@app.get("/api/cameras/{camera_id}")
async def get_camera_detail(camera_id: str):
    """Get detailed info for a specific camera."""
    if camera_id not in store.camera_metadata:
        raise HTTPException(404, f"Camera {camera_id} not found")

    return {
        "metadata": store.camera_metadata.get(camera_id, {}),
        "detection": store.latest_detections.get(camera_id, {}),
        "tracking": store.latest_tracking.get(camera_id, {}),
        "congestion": store.congestion_scores.get(camera_id, {}),
        "failure": store.failure_reports.get(camera_id, {}),
        "prediction": store.predictions.get(camera_id, {}),
    }


@app.get("/api/congestion")
async def get_congestion_map():
    """Get congestion scores for all cameras — used for heatmap layer."""
    return {
        "timestamp": store.last_updated,
        "cameras": [
            {
                "camera_id": cam_id,
                "latitude": store.camera_metadata.get(cam_id, {}).get("latitude", 0),
                "longitude": store.camera_metadata.get(cam_id, {}).get("longitude", 0),
                "score": score.get("score", 0),
                "level": score.get("level", "unknown"),
                "vehicles": score.get("unique_vehicles", 0),
            }
            for cam_id, score in store.congestion_scores.items()
        ],
    }


@app.get("/api/failures")
async def get_failure_report():
    """Get fleet-wide failure analysis report."""
    return store.fleet_report or {"message": "No failure report available yet"}


@app.get("/api/failures/{camera_id}")
async def get_camera_failures(camera_id: str):
    """Get failure analysis for a specific camera."""
    if camera_id not in store.failure_reports:
        raise HTTPException(404, f"No failure data for camera {camera_id}")
    return store.failure_reports[camera_id]


@app.get("/api/drift")
async def get_drift_alerts(
    limit: int = Query(20, description="Number of recent alerts to return"),
):
    """Get recent drift detection alerts."""
    return {
        "total_alerts": len(store.drift_alerts),
        "recent": store.drift_alerts[-limit:],
    }


@app.get("/api/predictions")
async def get_predictions():
    """Get congestion predictions for all cameras."""
    return {
        "timestamp": store.last_updated,
        "predictions": store.predictions,
    }


@app.get("/api/stats")
async def get_system_stats():
    """Get overall system statistics."""
    # Count congestion levels
    levels = {}
    for score in store.congestion_scores.values():
        level = score.get("level", "unknown")
        levels[level] = levels.get(level, 0) + 1

    # Count failure types
    failure_types = {}
    for report in store.failure_reports.values():
        for flag in report.get("failure_flags", []):
            failure_types[flag] = failure_types.get(flag, 0) + 1

    total_vehicles = sum(d.get("num_vehicles", 0) for d in store.latest_detections.values())

    return {
        "cameras_total": len(store.camera_metadata),
        "cameras_with_detections": len(store.latest_detections),
        "total_vehicles_detected": total_vehicles,
        "congestion_distribution": levels,
        "failure_distribution": failure_types,
        "drift_alerts_total": len(store.drift_alerts),
        "last_updated": store.last_updated,
    }


# =============================================================================
# Data Loading (called on startup or refresh)
# =============================================================================


@app.post("/api/refresh")
async def refresh_data(data_dir: str = "data/processed"):
    """Reload latest analytics results from disk.

    This endpoint is called by the collection pipeline after
    processing a new batch of data.
    """
    data_path = Path(data_dir)

    try:
        # Load camera metadata
        cameras_file = data_path / "cameras.json"
        if cameras_file.exists():
            with open(cameras_file) as f:
                store.camera_metadata = json.load(f)

        # Load latest detections
        detections_file = data_path / "latest_detections.json"
        if detections_file.exists():
            with open(detections_file) as f:
                store.latest_detections = json.load(f)

        # Load congestion scores
        congestion_file = data_path / "congestion_scores.json"
        if congestion_file.exists():
            with open(congestion_file) as f:
                store.congestion_scores = json.load(f)

        # Load failure reports
        failures_file = data_path / "failure_reports.json"
        if failures_file.exists():
            with open(failures_file) as f:
                store.failure_reports = json.load(f)

        # Load fleet report
        fleet_file = data_path / "fleet_report.json"
        if fleet_file.exists():
            with open(fleet_file) as f:
                store.fleet_report = json.load(f)

        # Load drift alerts
        drift_file = data_path / "drift_alerts.json"
        if drift_file.exists():
            with open(drift_file) as f:
                store.drift_alerts = json.load(f)

        # Load predictions
        predictions_file = data_path / "predictions.json"
        if predictions_file.exists():
            with open(predictions_file) as f:
                store.predictions = json.load(f)

        store.last_updated = datetime.now().isoformat()

        return {
            "status": "refreshed",
            "cameras": len(store.camera_metadata),
            "detections": len(store.latest_detections),
        }

    except Exception as e:
        logger.exception(f"Failed to refresh data: {e}")
        raise HTTPException(500, f"Refresh failed: {e}") from e
