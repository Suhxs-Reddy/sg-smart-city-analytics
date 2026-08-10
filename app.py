"""
CATI Smart City Analytics — Singapore Checkpoint Monitor
Standalone Streamlit frontend. Fetches directly from LTA Traffic Images API.
No backend or GPU required — deploys instantly on HuggingFace Spaces.

Covers the 8 active LTA cameras (post June 30 2026):
  Woodlands Checkpoint  — cams 2701, 2702, 2704
  Tuas Second Link      — cams 4703, 4712, 4713
  Sentosa Gateway       — cams 4798, 4799
"""

from __future__ import annotations

import time
import urllib.request
from io import BytesIO

import folium
import requests
import streamlit as st
from PIL import Image
from streamlit_autorefresh import st_autorefresh
from streamlit_folium import st_folium

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

# Active cameras since LTA decommissioned 82 of 90 on 30 June 2026
ACTIVE_CAMERAS = {"2701", "2702", "2704", "4703", "4712", "4713", "4798", "4799"}

CHECKPOINT_META = {
    "Woodlands": {
        "cams": ["2701", "2702", "2704"],
        "color": "#42a5f5",
        "location": [1.4473, 103.7691],
        "desc": "Malaysia–Singapore Causeway · 500K+ daily crossings",
    },
    "Tuas": {
        "cams": ["4703", "4712", "4713"],
        "color": "#fb923c",
        "location": [1.3239, 103.6388],
        "desc": "Second Link · Heavy freight & commercial vehicles",
    },
    "Sentosa": {
        "cams": ["4798", "4799"],
        "color": "#a78bfa",
        "location": [1.2548, 103.8218],
        "desc": "Sentosa Gateway · Tourist & leisure traffic",
    },
}

CAM_CHECKPOINT = {
    cam: cp
    for cp, meta in CHECKPOINT_META.items()
    for cam in meta["cams"]
}


# ── Data fetching ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=90)
def fetch_cameras() -> list[dict]:
    try:
        r = requests.get(LTA_API, timeout=10)
        if r.ok:
            all_cams = r.json().get("items", [{}])[0].get("cameras", [])
            return [c for c in all_cams if c.get("camera_id") in ACTIVE_CAMERAS]
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


# ── Auto-refresh every 90 s (matches camera hardware refresh rate) ─────────────
st_autorefresh(interval=90_000, key="auto")

# ── Header ─────────────────────────────────────────────────────────────────────
hcol1, hcol2, hcol3 = st.columns([5, 2, 1])
with hcol1:
    st.markdown("## 🚦 CATI · Singapore Checkpoint Monitor")
    st.caption("Live checkpoint cameras · Woodlands · Tuas · Sentosa · Auto-refreshes every 90s")

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
k1.metric("Live Cameras", f"{len(cameras)}/8")
k2.metric("Weather (Central)", weather)
k3.metric("Last Refresh", time.strftime("%H:%M:%S"))
k4.metric("Network", "3 Checkpoints")

st.markdown("---")

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_map, tab_feeds = st.tabs(["🗺️  Map", "📷  Live Feeds"])

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

    cam_lookup = {c["camera_id"]: c for c in cameras}

    for cp_name, meta in CHECKPOINT_META.items():
        color = meta["color"]

        # Checkpoint pin
        folium.Marker(
            location=meta["location"],
            icon=folium.DivIcon(
                html=f"""<div style="font-size:10px;font-weight:700;color:{color};
                         background:rgba(13,15,20,0.9);padding:2px 8px;
                         border:1px solid {color};border-radius:4px;
                         white-space:nowrap">{cp_name}</div>""",
                icon_size=(90, 22),
                icon_anchor=(45, 11),
            ),
        ).add_to(m)

        # Individual camera dots
        for cam_id in meta["cams"]:
            cam = cam_lookup.get(cam_id)
            if not cam:
                continue
            lat = cam["location"]["latitude"]
            lng = cam["location"]["longitude"]
            folium.CircleMarker(
                location=[lat, lng],
                radius=7,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.9,
                weight=1.5,
                tooltip=folium.Tooltip(
                    f"<b>Cam {cam_id}</b><br>{cp_name} Checkpoint<br>{meta['desc']}",
                    style="font-size:12px",
                ),
            ).add_to(m)

    # Legend
    legend = """<div style="position:fixed;bottom:24px;left:24px;z-index:1000;
         background:rgba(22,27,34,0.92);border:1px solid #30363d;
         border-radius:8px;padding:10px 14px;font-size:11px;color:#e6edf3">
      <b>Checkpoints</b><br><br>"""
    for cp_name, meta in CHECKPOINT_META.items():
        legend += f'<span style="color:{meta["color"]}">●</span> {cp_name} ({len(meta["cams"])} cams)<br>'
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
        cam_lookup = {c["camera_id"]: c for c in cameras}

        for cp_name, meta in CHECKPOINT_META.items():
            color = meta["color"]
            st.markdown(
                f"""<div style="margin:16px 0 8px;font-size:0.78rem;font-weight:700;
                color:{color};text-transform:uppercase;letter-spacing:0.08em;
                border-bottom:1px solid #30363d;padding-bottom:4px">
                {cp_name} Checkpoint &nbsp;·&nbsp; {meta['desc']}</div>""",
                unsafe_allow_html=True,
            )

            cp_cams = [cam_lookup[cid] for cid in meta["cams"] if cid in cam_lookup]
            cols = st.columns(len(meta["cams"]))
            for col, cam in zip(cols, cp_cams, strict=False):
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
