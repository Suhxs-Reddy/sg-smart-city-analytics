"""
CATI Smart City Analytics — Singapore Expressway Dashboard
Continuous city-wide CATI inference in a background thread.
Frontend is pure analytics — no inference on click.
"""

from __future__ import annotations

import csv
import os
import sys
import tempfile
import threading
import time
import urllib.request
from io import BytesIO, StringIO
from pathlib import Path

import folium
import requests
import streamlit as st
from PIL import Image
from streamlit_autorefresh import st_autorefresh
from streamlit_folium import st_folium

sys.path.insert(0, os.path.dirname(__file__))

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CATI · Singapore Analytics",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
html, body { background-color: #0d0f14 !important; color: #e6edf3; }
.stApp { background-color: #0d0f14; }
header[data-testid="stHeader"] { background: #161b22; border-bottom: 1px solid #30363d; }
.stTabs [data-baseweb="tab-list"] { background: #161b22; border-bottom: 1px solid #30363d; gap: 4px; padding: 0 16px; }
.stTabs [data-baseweb="tab"] { color: #8b949e !important; font-size: 0.88rem; font-weight: 500; padding: 10px 20px; }
.stTabs [aria-selected="true"] { color: #58a6ff !important; border-bottom: 2px solid #58a6ff !important; background: transparent !important; }
.kpi { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 14px 16px; text-align: center; }
.kpi-val { font-size: 1.8rem; font-weight: 700; color: #58a6ff; }
.kpi-lbl { font-size: 0.72rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.05em; }
.road-header { font-size: 0.78rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; border-bottom: 1px solid #30363d; padding-bottom: 4px; margin: 16px 0 8px; }
#MainMenu, footer, .stDeployButton { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
LTA_API = "https://api.data.gov.sg/v1/transport/traffic-images"
NEA_API = "https://api.data.gov.sg/v1/environment/24-hour-weather-forecast"
SG_CENTER = [1.3521, 103.8198]
DATASET_PATH = Path("/tmp/cati_dataset.csv")
INFERENCE_INTERVAL = 90  # seconds between full sweeps

ROAD_COLOR = {
    "PIE": "#42a5f5", "CTE": "#ab47bc", "MCE": "#ef5350",
    "TPE": "#26a69a", "BKE": "#ffa726", "AYE": "#66bb6a",
    "KJE": "#ec407a", "SLE": "#7e57c2", "ECP": "#26c6da", "—": "#78909c",
}
CATI_CLASSES = ["car", "motorcycle", "bus", "truck", "van", "lorry"]

_PREFIX_ROAD = {"1": "CTE", "2": "CTE", "3": "ECP", "4": "PIE",
                "5": "AYE", "6": "ECP", "7": "TPE", "8": "KJE", "9": "BKE"}
_MCE_IDS = {"6702", "6703", "6704", "6705"}

HF_MODEL_REPO = "SuhxsReddy/cati-singapore"


def _cam_road(cam_id: str) -> str:
    if cam_id in _MCE_IDS:
        return "MCE"
    return _PREFIX_ROAD.get(cam_id[0], "—")


# ── API helpers ────────────────────────────────────────────────────────────────
@st.cache_data(ttl=20)
def fetch_cameras() -> list[dict]:
    try:
        r = requests.get(LTA_API, timeout=10)
        if r.ok:
            return r.json().get("items", [{}])[0].get("cameras", [])
    except Exception:
        pass
    return []


@st.cache_data(ttl=300)
def fetch_weather() -> str:
    try:
        r = requests.get(NEA_API, timeout=8)
        if r.ok:
            items = r.json().get("items", [])
            if items:
                forecasts = items[0].get("periods", [])
                if forecasts:
                    return forecasts[0].get("regions", {}).get("central", "—")
    except Exception:
        pass
    return "—"


def load_image(url: str) -> Image.Image | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Referer": "https://data.gov.sg/"})
        data = urllib.request.urlopen(req, timeout=6).read()
        return Image.open(BytesIO(data))
    except Exception:
        return None


# ── CATI model (loaded once, shared across all reruns) ─────────────────────────
@st.cache_resource(show_spinner=False)
def get_model():
    try:
        from huggingface_hub import hf_hub_download
        from src.models.cati_detector import CATIBackboneWrapper, CATIConfig

        cati_path = hf_hub_download(repo_id=HF_MODEL_REPO, filename="cati_best.pt")
        yolo_path = hf_hub_download(repo_id=HF_MODEL_REPO, filename="yolo_backbone.pt")
        config = CATIConfig(use_context_augmentation=False, conf_threshold=0.15)
        return CATIBackboneWrapper(yolo_model_path=yolo_path, config=config,
                                   cati_weights_path=cati_path, device="cpu"), None
    except Exception as e:
        return None, str(e)


# ── Shared inference state (persists across Streamlit reruns) ──────────────────
@st.cache_resource(show_spinner=False)
def get_state() -> dict:
    return {
        "results": {},       # camera_id -> {count, by_class, road, lat, lon, ts}
        "running": False,
        "last_sweep": None,
        "cameras_done": 0,
        "total_cameras": 0,
        "error": None,
        "started": False,
    }


ANNOTATED_DIR = Path("/tmp/annotated")
CATI_CLASS_COLORS = {
    "car": "#58a6ff", "motorcycle": "#f78166", "bus": "#3fb950",
    "truck": "#d29922", "van": "#bc8cff", "lorry": "#ff7b72",
}


def _draw_boxes(image: Image.Image, detections: list[dict]) -> Image.Image:
    from PIL import ImageDraw, ImageFont
    img = image.convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default(size=14)
    except Exception:
        font = ImageFont.load_default()
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cls = CATI_CLASSES[det["class_id"]] if det["class_id"] < len(CATI_CLASSES) else "unknown"
        color = CATI_CLASS_COLORS.get(cls, "#ffffff")
        conf = det["confidence"]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        label = f"{cls} {conf:.2f}"
        draw.rectangle([x1, y1 - 16, x1 + len(label) * 8, y1], fill=color)
        draw.text((x1 + 2, y1 - 15), label, fill="#0d0f14", font=font)
    return img


def _run_inference_loop(state: dict, model):
    """Inference loop — model is passed in, already loaded in main thread."""

    is_first_sweep = True
    ANNOTATED_DIR.mkdir(exist_ok=True)

    # On startup: pull existing dataset from HF Hub so we don't lose history
    if not DATASET_PATH.exists():
        try:
            from huggingface_hub import hf_hub_download
            hf_token = os.environ.get("HF_TOKEN")
            existing = hf_hub_download(
                repo_id="SuhxsReddy/cati-singapore-dataset",
                filename="cati_detections.csv",
                repo_type="dataset",
                token=hf_token,
            )
            import shutil
            shutil.copy(existing, DATASET_PATH)
        except Exception:
            pass  # no existing dataset yet, start fresh

    # Init dataset CSV
    write_header = not DATASET_PATH.exists()
    dataset_file = open(DATASET_PATH, "a", newline="")
    writer = csv.writer(dataset_file)
    if write_header:
        writer.writerow(["timestamp", "camera_id", "road", "lat", "lon",
                         "weather", "total_vehicles", "car", "motorcycle",
                         "bus", "truck", "van", "lorry"])

    while True:
        try:
            cameras = fetch_cameras.__wrapped__()  # bypass Streamlit cache in thread
        except Exception:
            try:
                r = requests.get(LTA_API, timeout=10)
                cameras = r.json().get("items", [{}])[0].get("cameras", []) if r.ok else []
            except Exception:
                cameras = []

        try:
            r = requests.get(NEA_API, timeout=8)
            weather = r.json().get("items", [{}])[0].get("periods", [{}])[0].get("regions", {}).get("central", "clear") if r.ok else "clear"
        except Exception:
            weather = "clear"

        state["total_cameras"] = len(cameras)
        state["running"] = True
        state["cameras_done"] = 0
        hour = time.localtime().tm_hour + time.localtime().tm_min / 60.0

        for cam in cameras:
            cam_id = cam.get("camera_id", "0")
            img_url = cam.get("image", "")
            loc = cam.get("location", {})
            lat = loc.get("latitude", 0)
            lon = loc.get("longitude", 0)
            road = _cam_road(cam_id)

            img = load_image(img_url)
            if img is None:
                state["cameras_done"] += 1
                continue

            try:
                w, h = img.size
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                    img.save(tmp.name)
                    result = model.predict(
                        image_path=tmp.name,
                        camera_id=int(cam_id) % 90,
                        weather=weather,
                        temperature=28.0,
                        pm25=15.0,
                        hour=hour,
                        resolution=(w, h),
                        camera_lat=lat,
                        camera_lon=lon,
                        use_film=True,
                    )
                    os.unlink(tmp.name)

                counts = {c: 0 for c in CATI_CLASSES}
                for det in result["detections"]:
                    cls = CATI_CLASSES[det["class_id"]] if det["class_id"] < len(CATI_CLASSES) else None
                    if cls:
                        counts[cls] += 1

                ts = time.strftime("%Y-%m-%d %H:%M:%S")
                state["results"][cam_id] = {
                    "count": result["num_detections"],
                    "by_class": counts,
                    "road": road,
                    "lat": lat,
                    "lon": lon,
                    "ts": ts,
                }

                writer.writerow([ts, cam_id, road, lat, lon, weather,
                                 result["num_detections"]] + [counts[c] for c in CATI_CLASSES])
                dataset_file.flush()

                # Save annotated image on first sweep only
                if is_first_sweep and result["num_detections"] > 0:
                    try:
                        annotated = _draw_boxes(img, result["detections"])
                        annotated.save(ANNOTATED_DIR / f"cam_{cam_id}_{road}.jpg")
                    except Exception:
                        pass

            except Exception:
                pass

            state["cameras_done"] += 1

        state["running"] = False
        state["last_sweep"] = time.time()

        # Push annotated images after first sweep
        if is_first_sweep:
            try:
                from huggingface_hub import HfApi
                hf_token = os.environ.get("HF_TOKEN")
                if hf_token:
                    api = HfApi()
                    for img_path in ANNOTATED_DIR.glob("*.jpg"):
                        api.upload_file(
                            path_or_fileobj=str(img_path),
                            path_in_repo=f"annotated/{img_path.name}",
                            repo_id="SuhxsReddy/cati-singapore-dataset",
                            repo_type="dataset",
                            token=hf_token,
                        )
            except Exception:
                pass
            is_first_sweep = False

        # Push dataset to HF Hub after every sweep so restarts don't lose data
        try:
            from huggingface_hub import HfApi
            hf_token = os.environ.get("HF_TOKEN")
            if hf_token and DATASET_PATH.exists():
                api = HfApi()
                api.create_repo("SuhxsReddy/cati-singapore-dataset", token=hf_token,
                                repo_type="dataset", exist_ok=True, private=False)
                api.upload_file(
                    path_or_fileobj=str(DATASET_PATH),
                    path_in_repo="cati_detections.csv",
                    repo_id="SuhxsReddy/cati-singapore-dataset",
                    repo_type="dataset",
                    token=hf_token,
                )
        except Exception:
            pass  # dataset push failing should never stop inference

        time.sleep(INFERENCE_INTERVAL)


@st.cache_resource(show_spinner=False)
def start_inference_thread():
    state = get_state()
    if not state["started"]:
        model, err = get_model()  # load in main Streamlit thread
        if err:
            state["error"] = f"Model load failed: {err}"
        else:
            state["started"] = True
            t = threading.Thread(target=_run_inference_loop, args=(state, model), daemon=True)
            t.start()
    return True


# ── Start background inference ─────────────────────────────────────────────────
start_inference_thread()
state = get_state()
cameras = fetch_cameras()
weather = fetch_weather()

# Auto-refresh every 15s
st_autorefresh(interval=15_000, key="auto")

# ── Header ─────────────────────────────────────────────────────────────────────
results = state["results"]
total_vehicles = sum(r["count"] for r in results.values())
active_cams = len(results)

h1, h2, h3 = st.columns([5, 2, 1])
with h1:
    st.markdown("## 🚦 CATI · Singapore City Analytics")
    if state["running"]:
        done = state["cameras_done"]
        total = state["total_cameras"]
        st.caption(f"Inference running — {done}/{total} cameras processed")
    elif state["last_sweep"]:
        ago = int(time.time() - state["last_sweep"])
        st.caption(f"Last sweep {ago}s ago · Next in ~{max(0, INFERENCE_INTERVAL - ago)}s")
    elif state["error"]:
        st.caption(f"⚠ {state['error']}")
    else:
        st.caption("Starting inference engine…")

with h2:
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    if active_cams > 0:
        st.markdown('<span style="display:inline-flex;align-items:center;gap:5px;background:rgba(0,230,118,0.12);color:#00e676;border:1px solid #00e676;border-radius:20px;padding:3px 10px;font-size:0.75rem;font-weight:600">● LIVE</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span style="display:inline-flex;align-items:center;gap:5px;background:rgba(244,67,54,0.12);color:#f44336;border:1px solid #f44336;border-radius:20px;padding:3px 10px;font-size:0.75rem;font-weight:600">● LOADING</span>', unsafe_allow_html=True)

st.markdown("---")

# ── KPI row ────────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Vehicles", total_vehicles)
k2.metric("Cameras Analysed", f"{active_cams} / {len(cameras)}")
k3.metric("Weather (Central)", weather)
k4.metric("Last Updated", time.strftime("%H:%M:%S"))
k5.metric("Dataset Records", sum(1 for _ in open(DATASET_PATH)) - 1 if DATASET_PATH.exists() else 0)

st.markdown("---")

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_map, tab_roads, tab_dataset = st.tabs(["🗺️  Map", "📊  Road Analytics", "📦  Dataset"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — MAP with vehicle count overlays
# ══════════════════════════════════════════════════════════════════════════════
with tab_map:
    m = folium.Map(location=SG_CENTER, zoom_start=11, tiles="CartoDB dark_matter", prefer_canvas=True)

    for cam in cameras:
        cam_id = cam.get("camera_id", "")
        loc = cam.get("location", {})
        lat = loc.get("latitude", 0)
        lon = loc.get("longitude", 0)
        road = _cam_road(cam_id)
        color = ROAD_COLOR.get(road, "#546e7a")
        res = results.get(cam_id)

        count = res["count"] if res else 0
        radius = max(5, min(18, 5 + count * 1.5))
        opacity = 0.9 if res else 0.3
        label = f"Cam {cam_id} · {road}<br>{count} vehicles" if res else f"Cam {cam_id} · pending"

        folium.CircleMarker(
            location=[lat, lon],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=opacity,
            weight=1.5,
            tooltip=folium.Tooltip(label, style="font-size:12px"),
        ).add_to(m)

    # Legend
    legend = """<div style="position:fixed;bottom:24px;left:24px;z-index:1000;
         background:rgba(22,27,34,0.92);border:1px solid #30363d;
         border-radius:8px;padding:10px 14px;font-size:11px;color:#e6edf3">
      <b>Circle size = vehicle count</b><br><br>"""
    for road, color in ROAD_COLOR.items():
        if road != "—":
            legend += f'<span style="color:{color}">●</span> {road}<br>'
    legend += "</div>"
    m.get_root().html.add_child(folium.Element(legend))
    st_folium(m, width="100%", height=560, returned_objects=[])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ROAD ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
with tab_roads:
    if not results:
        st.info("Inference is running — analytics will appear after the first sweep completes.")
    else:
        # Aggregate by road
        road_data: dict[str, dict] = {}
        for cam_id, res in results.items():
            road = res["road"]
            if road not in road_data:
                road_data[road] = {"total": 0, "cameras": 0, "by_class": {c: 0 for c in CATI_CLASSES}}
            road_data[road]["total"] += res["count"]
            road_data[road]["cameras"] += 1
            for cls in CATI_CLASSES:
                road_data[road]["by_class"][cls] += res["by_class"].get(cls, 0)

        road_order = ["CTE", "PIE", "AYE", "ECP", "MCE", "TPE", "BKE", "SLE", "KJE"]
        present = [r for r in road_order if r in road_data]

        # Top-level road summary
        cols = st.columns(len(present)) if present else []
        for col, road in zip(cols, present):
            color = ROAD_COLOR.get(road, "#546e7a")
            d = road_data[road]
            col.markdown(f"""
            <div class="kpi">
                <div class="kpi-val" style="color:{color}">{d['total']}</div>
                <div class="kpi-lbl">{road}</div>
                <div style="font-size:0.65rem;color:#8b949e">{d['cameras']} cams</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")

        # Per-road breakdown
        for road in present:
            d = road_data[road]
            color = ROAD_COLOR.get(road, "#546e7a")
            st.markdown(f'<div class="road-header" style="color:{color}">{road} · {d["total"]} vehicles across {d["cameras"]} cameras</div>', unsafe_allow_html=True)

            cls_cols = st.columns(len(CATI_CLASSES))
            for col, cls in zip(cls_cols, CATI_CLASSES):
                col.metric(cls.capitalize(), d["by_class"].get(cls, 0))

        # Most congested cameras
        st.markdown("---")
        st.markdown("**Most congested cameras**")
        sorted_cams = sorted(results.items(), key=lambda x: x[1]["count"], reverse=True)[:10]
        for cam_id, res in sorted_cams:
            bar_width = min(100, res["count"] * 8)
            color = ROAD_COLOR.get(res["road"], "#546e7a")
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:10px;margin:4px 0">'
                f'<span style="width:80px;font-size:0.75rem;color:#8b949e">Cam {cam_id}</span>'
                f'<span style="width:40px;font-size:0.75rem;color:{color}">{res["road"]}</span>'
                f'<div style="background:{color};height:12px;width:{bar_width}px;border-radius:3px"></div>'
                f'<span style="font-size:0.75rem">{res["count"]}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — DATASET
# ══════════════════════════════════════════════════════════════════════════════
with tab_dataset:
    st.markdown("### Detection Dataset")
    st.caption("Every CATI inference result is logged here. Download to publish on HuggingFace Datasets.")

    if not DATASET_PATH.exists() or DATASET_PATH.stat().st_size == 0:
        st.info("Dataset is empty — inference is still running its first sweep.")
    else:
        try:
            import pandas as pd
            df = pd.read_csv(DATASET_PATH)
            st.metric("Total records", len(df))
            st.dataframe(df.tail(50), use_container_width=True)

            csv_bytes = df.to_csv(index=False).encode()
            st.download_button(
                "⬇ Download full dataset (CSV)",
                data=csv_bytes,
                file_name=f"cati_singapore_{time.strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        except Exception as e:
            st.error(f"Could not load dataset: {e}")
