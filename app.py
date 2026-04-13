"""
CATI Smart City Analytics — Singapore Expressway Live Feed
Standalone Streamlit frontend. Fetches directly from LTA Traffic Images API.
CATI inference tab runs context-aware detection using trained weights from HF Hub.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
import urllib.request
from io import BytesIO

import folium
import requests
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from streamlit_autorefresh import st_autorefresh
from streamlit_folium import st_folium

# Make src/ importable (works both locally and in Docker /app)
sys.path.insert(0, os.path.dirname(__file__))

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CATI · Singapore Traffic",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
html, body, [class*="css"] {
    background-color: #0d0f14 !important;
    color: #e6edf3;
}
.stApp { background-color: #0d0f14; }
header[data-testid="stHeader"] {
    background: #161b22;
    border-bottom: 1px solid #30363d;
}

/* Tab strip */
.stTabs [data-baseweb="tab-list"] {
    background: #161b22;
    border-bottom: 1px solid #30363d;
    gap: 4px; padding: 0 16px;
}
.stTabs [data-baseweb="tab"] {
    color: #8b949e !important;
    font-size: 0.88rem; font-weight: 500;
    padding: 10px 20px;
}
.stTabs [aria-selected="true"] {
    color: #58a6ff !important;
    border-bottom: 2px solid #58a6ff !important;
    background: transparent !important;
}

/* Cards */
.cam-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 8px;
    margin-bottom: 10px;
}
.cam-label {
    font-size: 0.72rem;
    color: #8b949e;
    margin-bottom: 4px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.road-badge {
    display: inline-block;
    border-radius: 4px;
    padding: 1px 7px;
    font-size: 0.7rem;
    font-weight: 700;
    margin-left: 6px;
}

/* KPI cards */
.kpi-row { display: flex; gap: 12px; margin-bottom: 16px; }
.kpi {
    flex: 1;
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 14px 16px;
    text-align: center;
}
.kpi-val { font-size: 1.8rem; font-weight: 700; color: #58a6ff; }
.kpi-lbl { font-size: 0.72rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.05em; }

/* Hide Streamlit chrome */
#MainMenu, footer, .stDeployButton { display: none !important; }

/* Status pills */
.live-pill {
    display: inline-flex; align-items: center; gap: 5px;
    background: rgba(0,230,118,0.12); color: #00e676;
    border: 1px solid #00e676; border-radius: 20px;
    padding: 3px 10px; font-size: 0.75rem; font-weight: 600;
}
.err-pill {
    display: inline-flex; align-items: center; gap: 5px;
    background: rgba(244,67,54,0.12); color: #f44336;
    border: 1px solid #f44336; border-radius: 20px;
    padding: 3px 10px; font-size: 0.75rem; font-weight: 600;
}
</style>
""",
    unsafe_allow_html=True,
)

# ── Constants ──────────────────────────────────────────────────────────────────
LTA_API = "https://api.data.gov.sg/v1/transport/traffic-images"
NEA_API = "https://api.data.gov.sg/v1/environment/24-hour-weather-forecast"

SG_CENTER = [1.3521, 103.8198]

ROAD_COLOR = {
    "PIE": "#42a5f5",
    "CTE": "#ab47bc",
    "MCE": "#ef5350",
    "TPE": "#26a69a",
    "BKE": "#ffa726",
    "AYE": "#66bb6a",
    "KJE": "#ec407a",
    "SLE": "#7e57c2",
    "ECP": "#26c6da",
    "NSC": "#78909c",
}

EXPRESSWAYS = {
    "PIE": [
        [1.340, 103.690],
        [1.335, 103.730],
        [1.327, 103.770],
        [1.320, 103.810],
        [1.313, 103.860],
        [1.307, 103.900],
    ],
    "CTE": [
        [1.430, 103.834],
        [1.390, 103.831],
        [1.355, 103.825],
        [1.321, 103.820],
        [1.290, 103.840],
    ],
    "AYE": [[1.312, 103.760], [1.307, 103.790], [1.305, 103.810], [1.290, 103.835]],
    "ECP": [[1.290, 103.860], [1.300, 103.890], [1.310, 103.920], [1.320, 103.960]],
    "MCE": [[1.278, 103.844], [1.270, 103.860], [1.266, 103.875], [1.270, 103.890]],
    "TPE": [[1.380, 103.920], [1.385, 103.950], [1.390, 103.980], [1.395, 104.000]],
    "BKE": [[1.430, 103.775], [1.390, 103.785], [1.350, 103.790]],
    "SLE": [[1.430, 103.834], [1.410, 103.860], [1.390, 103.890], [1.370, 103.920]],
    "KJE": [[1.390, 103.720], [1.370, 103.750], [1.350, 103.770]],
}

# LTA prefix → road
_PREFIX_ROAD = {
    "1": "CTE",
    "2": "CTE",
    "3": "ECP",
    "4": "PIE",
    "5": "AYE",
    "6": "ECP",
    "7": "TPE",
    "8": "KJE",
    "9": "BKE",
}
_MCE_IDS = {"6702", "6703", "6704", "6705"}


def _cam_road(cam_id: str) -> str:
    if cam_id in _MCE_IDS:
        return "MCE"
    return _PREFIX_ROAD.get(cam_id[0], "—")


# ── Data fetching ──────────────────────────────────────────────────────────────
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
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0",
                "Referer": "https://data.gov.sg/",
            },
        )
        data = urllib.request.urlopen(req, timeout=6).read()
        return Image.open(BytesIO(data))
    except Exception:
        return None


# ── Auto-refresh every 20 s ────────────────────────────────────────────────────
st_autorefresh(interval=20_000, key="auto")

# ── Header ─────────────────────────────────────────────────────────────────────
hcol1, hcol2, hcol3 = st.columns([5, 2, 1])
with hcol1:
    st.markdown("## 🚦 CATI · Singapore Expressway Monitor")
    st.caption("Real-time traffic camera feed · LTA Data · Auto-refreshes every 20s")

cameras = fetch_cameras()
weather = fetch_weather()

with hcol2:
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    if cameras:
        st.markdown('<span class="live-pill">● LIVE</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="err-pill">● NO DATA</span>', unsafe_allow_html=True)

with hcol3:
    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    if st.button("⟳", help="Force refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

st.markdown("---")

# KPI row
k1, k2, k3, k4 = st.columns(4)
k1.metric("Live Cameras", len(cameras))
k2.metric("Weather (Central)", weather)
k3.metric("Last Refresh", time.strftime("%H:%M:%S"))
k4.metric("Network", "Singapore Expressways")

st.markdown("---")

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_map, tab_feeds, tab_detect = st.tabs(["🗺️  Map", "📷  Live Feeds", "🤖  CATI Detection"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — MAP
# ══════════════════════════════════════════════════════════════════════════════
with tab_map:
    m = folium.Map(
        location=SG_CENTER,
        zoom_start=11,
        tiles="CartoDB dark_matter",
        prefer_canvas=True,
    )

    # Draw expressway polylines
    for road, coords in EXPRESSWAYS.items():
        color = ROAD_COLOR.get(road, "#546e7a")
        folium.PolyLine(
            locations=coords,
            color=color,
            weight=4,
            opacity=0.75,
            tooltip=road,
        ).add_to(m)
        mid = coords[len(coords) // 2]
        folium.Marker(
            location=mid,
            icon=folium.DivIcon(
                html=f"""<div style="font-size:9px;font-weight:700;color:{color};
                         background:rgba(13,15,20,0.8);padding:1px 5px;
                         border-radius:3px;white-space:nowrap">{road}</div>""",
                icon_size=(40, 16),
                icon_anchor=(20, 8),
            ),
        ).add_to(m)

    # Plot camera markers
    for cam in cameras:
        cam_id = cam.get("camera_id", "")
        lat = cam.get("location", {}).get("latitude", 0)
        lng = cam.get("location", {}).get("longitude", 0)
        road = _cam_road(cam_id)
        color = ROAD_COLOR.get(road, "#546e7a")

        folium.CircleMarker(
            location=[lat, lng],
            radius=6,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            weight=1.5,
            tooltip=folium.Tooltip(
                f"<b>Camera {cam_id}</b><br>{road}",
                style="font-size:12px",
            ),
        ).add_to(m)

    # Legend
    legend = """<div style="position:fixed;bottom:24px;left:24px;z-index:1000;
         background:rgba(22,27,34,0.92);border:1px solid #30363d;
         border-radius:8px;padding:10px 14px;font-size:11px;color:#e6edf3">
      <b>Expressways</b><br><br>"""
    for road, color in ROAD_COLOR.items():
        legend += f'<span style="color:{color}">━━</span> {road}<br>'
    legend += "</div>"
    m.get_root().html.add_child(folium.Element(legend))

    st_folium(m, width="100%", height=540, returned_objects=[])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — LIVE FEEDS
# ══════════════════════════════════════════════════════════════════════════════
with tab_feeds:
    if not cameras:
        st.warning("Could not fetch camera data from LTA API. Try refreshing.")
    else:
        # Group cameras by road
        by_road: dict[str, list] = {}
        for cam in cameras:
            road = _cam_road(cam.get("camera_id", ""))
            by_road.setdefault(road, []).append(cam)

        road_order = ["CTE", "PIE", "AYE", "ECP", "MCE", "TPE", "BKE", "SLE", "KJE", "—"]

        for road in road_order:
            cams = by_road.get(road, [])
            if not cams:
                continue

            road_color = ROAD_COLOR.get(road, "#546e7a")
            st.markdown(
                f"""<div style="margin:16px 0 8px;font-size:0.78rem;font-weight:700;
                color:{road_color};text-transform:uppercase;letter-spacing:0.08em;
                border-bottom:1px solid #30363d;padding-bottom:4px">
                {road} &nbsp;·&nbsp; {len(cams)} cameras</div>""",
                unsafe_allow_html=True,
            )

            # 4-column grid per road
            cols_per_row = 4
            for row_start in range(0, len(cams), cols_per_row):
                row_cams = cams[row_start : row_start + cols_per_row]
                cols = st.columns(cols_per_row)
                for col, cam in zip(cols, row_cams, strict=False):
                    cam_id = cam.get("camera_id", "")
                    img_url = cam.get("image", "")
                    with col:
                        img = load_image(img_url) if img_url else None
                        if img:
                            st.image(img, use_container_width=True)
                        else:
                            st.markdown(
                                """<div style="background:#1c2333;border:1px solid #30363d;
                                border-radius:4px;height:120px;display:flex;align-items:center;
                                justify-content:center;color:#8b949e;font-size:0.75rem">
                                No image</div>""",
                                unsafe_allow_html=True,
                            )
                        st.caption(f"Cam {cam_id}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — CATI DETECTION
# ══════════════════════════════════════════════════════════════════════════════

# Class names (6-class Singapore traffic model)
CATI_CLASSES = ["car", "motorcycle", "bus", "truck", "van", "lorry"]
CLASS_COLORS = {
    "car":        "#58a6ff",
    "motorcycle": "#f78166",
    "bus":        "#3fb950",
    "truck":      "#d29922",
    "van":        "#bc8cff",
    "lorry":      "#ff7b72",
}
HF_MODEL_REPO = "SuhxsReddy/cati-singapore"


@st.cache_resource(show_spinner="Loading CATI model from HuggingFace…")
def load_cati_model():
    """Download CATI + YOLO Phase 2 weights from HF Hub and initialise CATIBackboneWrapper."""
    try:
        from huggingface_hub import hf_hub_download
        from src.models.cati_detector import CATIBackboneWrapper, CATIConfig

        cati_path = hf_hub_download(repo_id=HF_MODEL_REPO, filename="cati_best.pt")
        yolo_path = hf_hub_download(repo_id=HF_MODEL_REPO, filename="yolo_backbone.pt")

        config = CATIConfig(use_context_augmentation=False)
        wrapper = CATIBackboneWrapper(
            yolo_model_path=yolo_path,
            config=config,
            cati_weights_path=cati_path,
            device="cpu",
        )
        return wrapper, None
    except Exception as e:
        return None, str(e)


def draw_detections(image: Image.Image, detections: list[dict]) -> Image.Image:
    """Draw bounding boxes on a PIL image."""
    img = image.convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default(size=14)
    except Exception:
        font = ImageFont.load_default()

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cls_name = CATI_CLASSES[det["class_id"]] if det["class_id"] < len(CATI_CLASSES) else "unknown"
        color = CLASS_COLORS.get(cls_name, "#ffffff")
        conf = det["confidence"]

        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        label = f"{cls_name} {conf:.2f}"
        draw.rectangle([x1, y1 - 16, x1 + len(label) * 8, y1], fill=color)
        draw.text((x1 + 2, y1 - 15), label, fill="#0d0f14", font=font)

    return img


with tab_detect:
    st.markdown("### CATI Context-Aware Detection")
    st.caption("Runs YOLOv11s + FiLM conditioning using real-time Singapore environmental metadata.")

    if not cameras:
        st.warning("No camera data available. Refresh the page.")
    else:
        cam_options = {f"Cam {c['camera_id']} ({_cam_road(c['camera_id'])})": c for c in cameras}
        selected_label = st.selectbox("Select camera", list(cam_options.keys()))
        selected_cam = cam_options[selected_label]

        col_ctx, col_img = st.columns([1, 2])

        with col_ctx:
            st.markdown("**Environmental context**")
            weather_choice = st.selectbox(
                "Weather",
                ["clear", "partly_cloudy", "cloudy", "overcast",
                 "light_rain", "moderate_rain", "heavy_rain",
                 "thunderstorm", "haze", "fog"],
                index=0,
            )
            temperature = st.slider("Temperature (°C)", 22.0, 36.0, 28.0, 0.5)
            pm25 = st.slider("PM2.5 (µg/m³)", 0.0, 150.0, 15.0, 1.0)
            hour = float(time.strftime("%H")) + float(time.strftime("%M")) / 60.0
            st.metric("Hour (auto)", f"{hour:.1f}")
            conf_thresh = st.slider("Confidence threshold", 0.05, 0.9, 0.10, 0.05)

        run_detect = st.button("▶  Run CATI Detection", use_container_width=True)

        with col_img:
            cam_img_url = selected_cam.get("image", "")
            preview = load_image(cam_img_url) if cam_img_url else None
            img_slot = st.empty()  # single slot — preview or annotated renders here
            caption_slot = st.empty()

            if preview:
                img_slot.image(preview, use_container_width=True)
                caption_slot.caption("Live feed — press Run to detect")
            else:
                img_slot.warning("Could not load camera image.")

        if run_detect:
            model, err = load_cati_model()
            if err:
                st.error(f"Model load failed: {err}")
            elif preview is None:
                st.error("No image to run detection on.")
            else:
                with st.spinner("Running CATI inference…"):
                    try:
                        cam_id_int = int(selected_cam.get("camera_id", 0))
                        loc = selected_cam.get("location", {})
                        lat = loc.get("latitude")
                        lon = loc.get("longitude")
                        w, h = preview.size
                        resolution = (w, h)

                        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                            preview.save(tmp.name)
                            model.config.conf_threshold = conf_thresh
                            result = model.predict(
                                image_path=tmp.name,
                                camera_id=cam_id_int % 90,
                                weather=weather_choice,
                                temperature=temperature,
                                pm25=pm25,
                                hour=hour,
                                resolution=resolution,
                                camera_lat=lat,
                                camera_lon=lon,
                                use_film=True,
                            )

                        n = result["num_detections"]
                        annotated = draw_detections(preview, result["detections"])
                        # Replace preview in the same slot
                        img_slot.image(annotated, use_container_width=True)
                        caption_slot.caption(f"CATI detections — {n} object(s) found")
                        if n == 0:
                            st.info("No detections above threshold. Try lowering the confidence slider.")

                        # Analytics
                        counts = {}
                        for det in result["detections"]:
                            cls = CATI_CLASSES[det["class_id"]] if det["class_id"] < len(CATI_CLASSES) else "unknown"
                            counts[cls] = counts.get(cls, 0) + 1

                        st.markdown("**Detection results**")
                        mc = st.columns(len(CATI_CLASSES))
                        for i, cls in enumerate(CATI_CLASSES):
                            mc[i].metric(cls.capitalize(), counts.get(cls, 0))

                        ctx = result["context"]
                        st.caption(
                            f"Weather: {ctx['weather']} · Temp: {ctx['temperature']}°C · "
                            f"PM2.5: {ctx['pm25']} · Hour: {ctx['hour']:.1f} · "
                            f"Device: {result['inference_device']} · "
                            f"CATI params: {result['cati_params']['total_cati_overhead']:,}"
                        )
                    except Exception as e:
                        st.error(f"Inference failed: {e}")
