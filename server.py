"""
CATI REST API — Singapore Smart City Live Traffic Analytics

FastAPI backend that:
  - Loads CATIPipeline once at startup (GPU-bound, ~45s on T4)
  - Fetches live LTA camera frames in a background thread
  - Runs CATI inference across the 8 active LTA checkpoint cameras
  - Exposes clean REST endpoints consumed by the Streamlit frontend (app.py)

Camera network (post June 30 2026 LTA decommission):
  Woodlands Checkpoint  — cams 2701, 2702, 2704
  Tuas Second Link      — cams 4703, 4712, 4713
  Sentosa Gateway       — cams 4798, 4799

Architecture:
  One CATIPipeline instance (module-level singleton).
  One background refresh loop (asyncio + ThreadPoolExecutor) so HTTP serving
  is never blocked by GPU inference.
  CATIStore holds the latest results in memory behind a threading.Lock.

Run locally:
    python server.py

Run on HF Spaces (launched by app.py):
    uvicorn server:app --host 0.0.0.0 --port 7860
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("cati.server")

# ---------------------------------------------------------------------------
# Singapore config
# ---------------------------------------------------------------------------

SGT = timezone(timedelta(hours=8))

# Active cameras since LTA decommissioned 82 of 90 cameras on 30 June 2026.
# All 8 are 1920×1080. Actual pixel refresh rate ~4 min (CDN cache-busting
# makes URLs rotate every 20s, but content only changes every ~4 min).
SELECTED_CAMERAS = [
    "2701", "2702", "2704",   # Woodlands Checkpoint
    "4703", "4712", "4713",   # Tuas Second Link
    "4798", "4799",           # Sentosa Gateway
]

CHECKPOINT_LABEL = {
    "2701": "Woodlands", "2702": "Woodlands", "2704": "Woodlands",
    "4703": "Tuas",      "4712": "Tuas",      "4713": "Tuas",
    "4798": "Sentosa",   "4799": "Sentosa",
}

REFRESH_INTERVAL_S = 90.0  # match camera hardware refresh rate


# ---------------------------------------------------------------------------
# In-memory store
# ---------------------------------------------------------------------------


@dataclass
class CATIStore:
    frame_results: dict[str, dict] = field(default_factory=dict)
    annotated_jpegs: dict[str, bytes] = field(default_factory=dict)
    raw_jpegs: dict[str, bytes] = field(default_factory=dict)
    network_summary: dict = field(default_factory=dict)
    network_state: dict = field(default_factory=dict)
    weather: str = "unknown"
    temperature: float = 28.0
    last_refresh: str = ""
    refresh_count: int = 0
    avg_pipeline_ms: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock)
    bg_task: object = field(default=None)

    def snapshot(self) -> CATIStore:
        """Thread-safe shallow copy for read operations."""
        with self._lock:
            return CATIStore(
                frame_results={**self.frame_results},
                annotated_jpegs={**self.annotated_jpegs},
                raw_jpegs={**self.raw_jpegs},
                network_summary={**self.network_summary},
                network_state={**self.network_state},
                weather=self.weather,
                temperature=self.temperature,
                last_refresh=self.last_refresh,
                refresh_count=self.refresh_count,
                avg_pipeline_ms=self.avg_pipeline_ms,
            )


# Module-level singletons
_pipeline = None
_store = CATIStore()
_executor = ThreadPoolExecutor(max_workers=1)
_refresh_event = asyncio.Event()  # set by POST /api/refresh for forced refresh
_startup_done = threading.Event()


# ---------------------------------------------------------------------------
# LTA data fetching
# ---------------------------------------------------------------------------


def _fetch_weather() -> tuple[str, float]:
    """Fetch current Singapore weather and temperature from NEA API."""
    try:
        req = urllib.request.Request(
            "https://api.data.gov.sg/v1/environment/24-hour-weather-forecast",
            headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"},
        )
        data = json.loads(urllib.request.urlopen(req, timeout=8).read())
        weather = data["items"][0]["general"]["forecast"]
    except Exception:
        weather = "unknown"

    try:
        req = urllib.request.Request(
            "https://api.data.gov.sg/v1/environment/air-temperature",
            headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"},
        )
        data = json.loads(urllib.request.urlopen(req, timeout=8).read())
        readings = data["items"][0]["readings"]
        temperature = round(sum(r["value"] for r in readings) / len(readings), 1)
    except Exception:
        temperature = 28.0

    return weather, temperature


def _fetch_frames(weather: str, temperature: float) -> dict[str, dict]:
    """
    Fetch live JPEG frames for all selected cameras from LTA API.
    Returns {camera_id: {image: np.ndarray, lat: float, lon: float}}.
    """
    frames: dict[str, dict] = {}

    try:
        req = urllib.request.Request(
            "https://api.data.gov.sg/v1/transport/traffic-images",
            headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"},
        )
        data = json.loads(urllib.request.urlopen(req, timeout=10).read())
        cameras = data["items"][0]["cameras"]
        cam_map = {c["camera_id"]: c for c in cameras}

        def _dl_one(cid: str):
            cam = cam_map.get(cid)
            if cam is None:
                return
            try:
                img_req = urllib.request.Request(
                    cam["image"],
                    headers={
                        "User-Agent": "Mozilla/5.0",
                        "Referer": "https://data.gov.sg/",
                        "Accept": "image/jpeg,image/*",
                    },
                )
                resp = urllib.request.urlopen(img_req, timeout=8)
                buf = np.frombuffer(resp.read(), np.uint8)
                img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                if img is not None:
                    frames[cid] = {
                        "image": img,
                        "lat": cam["location"]["latitude"],
                        "lon": cam["location"]["longitude"],
                    }
            except Exception as e:
                logger.warning(f"Camera {cid}: {e}")

        # Parallel download (I/O bound — safe to use threads here)
        with ThreadPoolExecutor(max_workers=9) as pool:
            list(pool.map(_dl_one, SELECTED_CAMERAS))

    except Exception as e:
        logger.error(f"LTA API fetch failed: {e}")

    return frames


# ---------------------------------------------------------------------------
# Inference batch (runs in executor thread, not async)
# ---------------------------------------------------------------------------


def _run_inference_batch():
    """
    Fetch live frames, run CATI inference on all 9 cameras, update CATIStore.
    Called from the background refresh loop via run_in_executor.
    """
    global _pipeline, _store

    if _pipeline is None:
        logger.warning("Pipeline not loaded yet — skipping inference batch")
        return

    timestamp = datetime.now(SGT).isoformat()
    weather, temperature = _fetch_weather()
    frames = _fetch_frames(weather, temperature)

    if not frames:
        logger.warning("No frames fetched — skipping this cycle")
        return

    results = {}
    annotated = {}
    raw_jpegs = {}
    pipeline_ms_list = []

    for camera_id, fd in frames.items():
        try:
            result = _pipeline.process_frame(
                image_bgr=fd["image"],
                camera_id=camera_id,
                timestamp=timestamp,
                weather=weather,
                temperature=temperature,
                frame_id=_store.refresh_count,
            )
            results[camera_id] = result.to_dict()
            pipeline_ms_list.append(result.pipeline_ms)

            # Annotated JPEG
            ann_bgr = _pipeline.draw(fd["image"], result)
            _, ann_buf = cv2.imencode(".jpg", ann_bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
            annotated[camera_id] = bytes(ann_buf)

            # Raw JPEG
            _, raw_buf = cv2.imencode(".jpg", fd["image"], [cv2.IMWRITE_JPEG_QUALITY, 85])
            raw_jpegs[camera_id] = bytes(raw_buf)

        except Exception as e:
            logger.error(f"Inference failed for camera {camera_id}: {e}")

    # Build lightweight network summary from stored result dicts.
    net_summary = _build_network_summary(results)
    net_state = _pipeline.network_state()

    with _store._lock:
        _store.frame_results.update(results)
        _store.annotated_jpegs.update(annotated)
        _store.raw_jpegs.update(raw_jpegs)
        _store.network_summary = net_summary
        _store.network_state = net_state
        _store.weather = weather
        _store.temperature = temperature
        _store.last_refresh = timestamp
        _store.refresh_count += 1
        _store.avg_pipeline_ms = (
            sum(pipeline_ms_list) / len(pipeline_ms_list) if pipeline_ms_list else 0.0
        )

    logger.info(
        f"Refresh #{_store.refresh_count} complete — "
        f"{len(results)}/{len(SELECTED_CAMERAS)} cameras — "
        f"avg {_store.avg_pipeline_ms:.0f}ms/frame"
    )


def _build_network_summary(frame_results: dict[str, dict]) -> dict:
    """Build network-level summary from FrameResult dicts."""
    if not frame_results:
        return {}

    states = [r["traffic_state"] for r in frame_results.values()]

    total_vehicles = sum(s["total_vehicles"] for s in states)
    avg_occupancy = sum(s["occupancy"] for s in states) / len(states)
    avg_congestion = sum(s["congestion_score"] for s in states) / len(states)
    worst_camera = max(
        frame_results, key=lambda k: frame_results[k]["traffic_state"]["congestion_score"]
    )

    by_road: dict[str, list] = {}
    by_region: dict[str, list] = {}
    for _cam_id, r in frame_results.items():
        s = r["traffic_state"]
        road = s.get("road", "Unknown")
        reg = s.get("region", "Unknown")
        by_road.setdefault(road, []).append(s)
        by_region.setdefault(reg, []).append(s)

    def _avg(lst, key):
        return round(sum(x[key] for x in lst) / len(lst), 4) if lst else 0.0

    return {
        "total_cameras": len(frame_results),
        "total_vehicles": total_vehicles,
        "avg_occupancy": round(avg_occupancy, 4),
        "avg_congestion": round(avg_congestion, 4),
        "worst_camera": worst_camera,
        "by_road": {
            road: {
                "cameras": len(ss),
                "total_vehicles": sum(s["total_vehicles"] for s in ss),
                "avg_congestion": _avg(ss, "congestion_score"),
                "avg_los": max(ss, key=lambda s: s["congestion_score"])["los"],
                "avg_speed_kmh": _avg([s for s in ss if s.get("speed_kmh", 0) > 0], "speed_kmh"),
            }
            for road, ss in by_road.items()
        },
        "by_region": {
            reg: {
                "cameras": len(ss),
                "total_vehicles": sum(s["total_vehicles"] for s in ss),
                "avg_congestion": _avg(ss, "congestion_score"),
                "avg_occupancy": _avg(ss, "occupancy"),
            }
            for reg, ss in by_region.items()
        },
    }


# ---------------------------------------------------------------------------
# Background refresh loop
# ---------------------------------------------------------------------------


async def _refresh_loop():
    """Runs forever: inference every REFRESH_INTERVAL_S seconds."""
    loop = asyncio.get_event_loop()
    while True:
        t0 = time.monotonic()
        try:
            await loop.run_in_executor(_executor, _run_inference_batch)
        except Exception as e:
            logger.error(f"Refresh loop error: {e}")

        _startup_done.set()  # signal that first batch is done

        elapsed = time.monotonic() - t0
        wait = max(0.0, REFRESH_INTERVAL_S - elapsed)
        try:
            await asyncio.wait_for(_refresh_event.wait(), timeout=wait)
            _refresh_event.clear()
        except TimeoutError:
            pass


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="CATI Singapore Traffic API",
    description="Real-time expressway analytics powered by CATI FiLM-conditioned YOLOv11",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_start_time = time.monotonic()


def _resolve_weight(env_var: str, filename: str) -> str:
    """
    Resolve a model weight path from an environment variable.

    Accepts two formats:
      - HF Hub:  "username/repo-name"  (downloads to /tmp/cati_weights/)
      - Local:   "/absolute/path/to/weights.pt"

    Returns the local filesystem path to the .pt file.
    """
    value = os.environ.get(env_var, "").strip()
    if not value:
        return ""

    # HF Hub format: contains "/" but no file extension → it's a repo ID
    if "/" in value and not value.endswith(".pt") and not value.startswith("/"):
        try:
            from huggingface_hub import hf_hub_download

            cache_dir = "/tmp/cati_weights"
            os.makedirs(cache_dir, exist_ok=True)
            logger.info(f"Downloading {filename} from HF Hub repo: {value}")
            local_path = hf_hub_download(
                repo_id=value,
                filename=filename,
                cache_dir=cache_dir,
            )
            logger.info(f"Downloaded {filename} → {local_path}")
            return local_path
        except Exception as e:
            logger.error(f"HF Hub download failed for {env_var}={value}: {e}")
            return ""

    # Local path — use as-is
    return value


@app.on_event("startup")
async def _startup():
    global _pipeline

    loop = asyncio.get_event_loop()

    def _load_pipeline():
        yolo_path = _resolve_weight("YOLO_WEIGHTS", "yolo_best.pt")
        cati_path = _resolve_weight("CATI_WEIGHTS", "cati_phase2_final.pt")
        feature_dir = os.environ.get("FEATURE_DIR", "")
        device = os.environ.get("DEVICE", "cuda")

        if not (yolo_path and cati_path):
            logger.warning(
                "YOLO_WEIGHTS / CATI_WEIGHTS not set or download failed. "
                "Running in no-model mode — inference disabled, API still serves cached data."
            )
            return None

        try:
            from src.inference import CATIPipeline

            pipeline = CATIPipeline(
                yolo_weights=yolo_path,
                cati_weights=cati_path,
                feature_dir=feature_dir or "",
                device=device,
                conf=0.25,
                use_neck_film=True,
            )
            logger.info("CATIPipeline loaded successfully.")
            return pipeline
        except Exception as e:
            logger.error(f"Pipeline load failed: {e}")
            return None

    # Load weights in the executor so startup doesn't block the event loop
    global _pipeline
    _pipeline = await loop.run_in_executor(_executor, _load_pipeline)

    _bg_task = asyncio.create_task(_refresh_loop())
    _store.bg_task = _bg_task  # prevent GC of the long-lived task


# ---------------------------------------------------------------------------
# Routes — Health
# ---------------------------------------------------------------------------


@app.get("/")
def root():
    snap = _store.snapshot()
    return {
        "service": "CATI Singapore Traffic API",
        "version": "1.0.0",
        "cameras_active": len(snap.frame_results),
        "last_refresh": snap.last_refresh,
        "refresh_count": snap.refresh_count,
        "uptime_s": round(time.monotonic() - _start_time, 1),
    }


@app.get("/api/health")
def health():
    snap = _store.snapshot()
    stale = False
    if snap.last_refresh:
        last_dt = datetime.fromisoformat(snap.last_refresh)
        stale = (datetime.now(SGT) - last_dt).total_seconds() > 60
    return {
        "status": "degraded" if (stale or _pipeline is None) else "healthy",
        "pipeline_loaded": _pipeline is not None,
        "last_refresh": snap.last_refresh,
        "refresh_count": snap.refresh_count,
        "avg_inference_ms": round(snap.avg_pipeline_ms, 1),
        "cameras_active": len(snap.frame_results),
        "weather": snap.weather,
        "temperature_c": snap.temperature,
    }


# ---------------------------------------------------------------------------
# Routes — Cameras
# ---------------------------------------------------------------------------


@app.get("/api/cameras/list")
def cameras_list():
    """All 90 LTA cameras with lat/lon for map rendering."""
    from src.analytics.camera_network import CameraNetwork

    net = CameraNetwork()
    return [
        {
            "camera_id": node.camera_id,
            "lat": node.lat,
            "lon": node.lon,
            "road": node.road,
            "region": node.region,
            "area": node.area,
            "active": node.camera_id in SELECTED_CAMERAS,
        }
        for node in net.nodes.values()
    ]


@app.get("/api/cameras")
def cameras():
    """Current state summary for all active cameras (used to colour map markers)."""
    snap = _store.snapshot()
    out = []
    for cam_id, result in snap.frame_results.items():
        ts = result["traffic_state"]
        out.append(
            {
                "camera_id": cam_id,
                "road": result["road"],
                "region": result["region"],
                "area": result["area"],
                "los": ts["los"],
                "los_label": ts["los_label"],
                "congestion_score": ts["congestion_score"],
                "total_vehicles": ts["total_vehicles"],
                "occupancy": ts["occupancy"],
                "speed_kmh": ts.get("speed_kmh", 0.0),
                "speed_limit": ts.get("speed_limit", 80),
                "peak_hour_mult": ts.get("peak_hour_mult", 1.0),
                "weather": ts["weather"],
                "timestamp": result["timestamp"],
                "pipeline_ms": result["pipeline_ms"],
            }
        )
    return out


@app.get("/api/cameras/{camera_id}")
def camera_detail(camera_id: str):
    snap = _store.snapshot()
    if camera_id not in snap.frame_results:
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")
    return snap.frame_results[camera_id]


@app.get("/api/cameras/{camera_id}/image")
def camera_image(camera_id: str, raw: bool = False):
    """Annotated (default) or raw JPEG for a camera."""
    snap = _store.snapshot()
    store = snap.raw_jpegs if raw else snap.annotated_jpegs
    if camera_id not in store:
        raise HTTPException(status_code=404, detail=f"No image for camera {camera_id}")
    ts = snap.frame_results.get(camera_id, {}).get("traffic_state", {})
    return Response(
        content=store[camera_id],
        media_type="image/jpeg",
        headers={
            "Cache-Control": "no-store",
            "X-LOS-Grade": ts.get("los", "?"),
            "X-Timestamp": snap.last_refresh,
        },
    )


# ---------------------------------------------------------------------------
# Routes — Network Analytics
# ---------------------------------------------------------------------------


@app.get("/api/network")
def network():
    snap = _store.snapshot()
    return {
        "timestamp": snap.last_refresh,
        "summary": snap.network_summary,
        "speed_profile": snap.network_state.get("speed_profile", {}),
        "gallery_stats": snap.network_state.get("gallery_stats", {}),
        "active_cameras": snap.network_state.get("active_cameras", []),
    }


@app.get("/api/speed")
def speed_readings():
    snap = _store.snapshot()
    readings = []
    for result in snap.frame_results.values():
        for sr in result.get("speed_readings", []):
            if not sr.get("is_outlier", True):
                readings.append(sr)
    return {
        "timestamp": snap.last_refresh,
        "readings": readings,
        "road_profile": snap.network_state.get("speed_profile", {}),
    }


# ---------------------------------------------------------------------------
# Routes — Admin
# ---------------------------------------------------------------------------


@app.post("/api/refresh")
async def force_refresh():
    """Trigger an immediate inference cycle."""
    _refresh_event.set()
    return {"triggered": True, "queued_at": datetime.now(SGT).isoformat()}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 7860)),
        log_level="info",
    )
