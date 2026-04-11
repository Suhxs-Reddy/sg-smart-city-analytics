"""
CATI Smart City Analytics — Streamlit frontend
Connects to the FastAPI backend (server.py) running on the same process or localhost:8000.
"""

import os
import time

import folium
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh
from streamlit_folium import st_folium

# ── Config ─────────────────────────────────────────────────────────────────────
API_BASE = os.getenv("API_BASE", "http://localhost:8000")
REFRESH_INTERVAL_MS = 20_000  # 20 s

LOS_COLOR = {
    "A": "#00e676",
    "B": "#69f0ae",
    "C": "#ffeb3b",
    "D": "#ff9800",
    "E": "#f44336",
    "F": "#b71c1c",
    "?": "#78909c",
}
LOS_LABEL = {
    "A": "Free Flow",
    "B": "Stable",
    "C": "Stable (near cap)",
    "D": "Approaching unstable",
    "E": "Unstable",
    "F": "Breakdown",
    "?": "Unknown",
}

SG_CENTER = [1.3521, 103.8198]

EXPRESSWAYS = {
    "PIE": [
        [1.3400, 103.6900],
        [1.3350, 103.7300],
        [1.3270, 103.7700],
        [1.3200, 103.8100],
        [1.3130, 103.8600],
        [1.3070, 103.9000],
    ],
    "CTE": [
        [1.4300, 103.8340],
        [1.3900, 103.8310],
        [1.3550, 103.8250],
        [1.3210, 103.8200],
        [1.2900, 103.8400],
    ],
    "MCE": [[1.2780, 103.8440], [1.2700, 103.8600], [1.2660, 103.8750], [1.2700, 103.8900]],
    "TPE": [[1.3800, 103.9200], [1.3850, 103.9500], [1.3900, 103.9800], [1.3950, 104.0000]],
    "BKE": [[1.4300, 103.7750], [1.3900, 103.7850], [1.3500, 103.7900]],
    "AYE": [[1.3120, 103.7600], [1.3070, 103.7900], [1.3050, 103.8100], [1.2900, 103.8350]],
    "KJE": [[1.3900, 103.7200], [1.3700, 103.7500], [1.3500, 103.7700]],
    "SLE": [[1.4300, 103.8340], [1.4100, 103.8600], [1.3900, 103.8900], [1.3700, 103.9200]],
    "ECP": [[1.2900, 103.8600], [1.3000, 103.8900], [1.3100, 103.9200], [1.3200, 103.9600]],
}

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
}

# ── Page setup ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CATI Smart City Analytics",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
  /* ── Dark base ── */
  :root {
    --bg-primary: #0d0f14;
    --bg-panel:   #161b22;
    --bg-card:    #1c2333;
    --accent:     #58a6ff;
    --text-main:  #e6edf3;
    --text-muted: #8b949e;
    --border:     #30363d;
  }
  html, body, [class*="css"] { background-color: var(--bg-primary) !important; color: var(--text-main); }
  .stApp { background-color: var(--bg-primary); }
  header[data-testid="stHeader"] { background: var(--bg-panel); border-bottom: 1px solid var(--border); }

  /* ── Tab strip ── */
  .stTabs [data-baseweb="tab-list"] {
    background: var(--bg-panel);
    border-bottom: 1px solid var(--border);
    gap: 4px;
    padding: 0 16px;
  }
  .stTabs [data-baseweb="tab"] {
    color: var(--text-muted) !important;
    border-radius: 4px 4px 0 0;
    font-size: 0.88rem;
    font-weight: 500;
    padding: 10px 20px;
  }
  .stTabs [aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
    background: transparent !important;
  }
  .stTabs [data-baseweb="tab-panel"] { padding: 16px 0; }

  /* ── KPI cards ── */
  .kpi-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 18px 20px;
    text-align: center;
  }
  .kpi-value { font-size: 2rem; font-weight: 700; line-height: 1; margin-bottom: 4px; }
  .kpi-label { font-size: 0.78rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em; }

  /* ── Camera cards ── */
  .cam-card {
    background: var(--bg-card);
    border: 2px solid var(--border);
    border-radius: 8px;
    padding: 10px;
    margin-bottom: 12px;
  }
  .cam-header { font-size: 0.78rem; color: var(--text-muted); margin-bottom: 6px; display: flex; justify-content: space-between; }
  .los-badge {
    display: inline-block; border-radius: 4px; padding: 2px 8px;
    font-size: 0.72rem; font-weight: 700; color: #000;
  }

  /* ── Section headers ── */
  .section-title {
    font-size: 0.75rem; font-weight: 600; color: var(--text-muted);
    text-transform: uppercase; letter-spacing: 0.08em;
    margin: 16px 0 8px;
    border-bottom: 1px solid var(--border); padding-bottom: 4px;
  }

  /* ── Status pill ── */
  .status-pill {
    display: inline-flex; align-items: center; gap: 6px;
    border-radius: 20px; padding: 4px 12px; font-size: 0.78rem;
  }
  .status-live   { background: rgba(0,230,118,0.15); color: #00e676; border: 1px solid #00e676; }
  .status-stale  { background: rgba(255,152,0,0.15);  color: #ff9800;  border: 1px solid #ff9800; }
  .status-error  { background: rgba(244,67,54,0.15);  color: #f44336;  border: 1px solid #f44336; }

  /* Plotly chart backgrounds */
  .js-plotly-plot .plotly { background: transparent !important; }
  .stDataFrame { background: var(--bg-card); }

  /* Hide Streamlit chrome */
  #MainMenu { visibility: hidden; }
  footer     { visibility: hidden; }
  .stDeployButton { display: none; }
</style>
""",
    unsafe_allow_html=True,
)

# ── Auto-refresh ───────────────────────────────────────────────────────────────
st_autorefresh(interval=REFRESH_INTERVAL_MS, key="main_refresh")


# ── API helpers ────────────────────────────────────────────────────────────────
@st.cache_data(ttl=18)
def fetch_cameras():
    try:
        r = requests.get(f"{API_BASE}/api/cameras", timeout=10)
        return r.json() if r.ok else {}
    except Exception:
        return {}


@st.cache_data(ttl=18)
def fetch_camera_list():
    try:
        r = requests.get(f"{API_BASE}/api/cameras/list", timeout=10)
        return r.json() if r.ok else []
    except Exception:
        return []


@st.cache_data(ttl=18)
def fetch_network():
    try:
        r = requests.get(f"{API_BASE}/api/network", timeout=10)
        return r.json() if r.ok else {}
    except Exception:
        return {}


@st.cache_data(ttl=18)
def fetch_speed():
    try:
        r = requests.get(f"{API_BASE}/api/speed", timeout=10)
        return r.json() if r.ok else []
    except Exception:
        return []


@st.cache_data(ttl=18)
def fetch_health():
    try:
        r = requests.get(f"{API_BASE}/api/health", timeout=5)
        return r.json() if r.ok else {}
    except Exception:
        return {}


def image_url(cam_id: str, raw: bool = False) -> str:
    return f"{API_BASE}/api/cameras/{cam_id}/image?raw={'true' if raw else 'false'}&t={int(time.time())}"


# ── Header ─────────────────────────────────────────────────────────────────────
col_title, col_status, col_refresh = st.columns([4, 2, 1])
with col_title:
    st.markdown("## 🚦 CATI Smart City Analytics")
    st.caption("Singapore Expressway Monitoring — Real-time AI Traffic Intelligence")

health = fetch_health()
with col_status:
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    if health.get("status") == "ok":
        age = health.get("data_age_s", 999)
        if age < 60:
            st.markdown(
                '<span class="status-pill status-live">● LIVE</span>', unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<span class="status-pill status-stale">● STALE ({age:.0f}s ago)</span>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<span class="status-pill status-error">● BACKEND OFFLINE</span>',
            unsafe_allow_html=True,
        )

with col_refresh:
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    if st.button("⟳ Refresh", use_container_width=True):
        import contextlib

        with contextlib.suppress(Exception):
            requests.post(f"{API_BASE}/api/refresh", timeout=5)
        st.cache_data.clear()
        st.rerun()

st.markdown("---")

# ── Load data ──────────────────────────────────────────────────────────────────
all_cameras = fetch_camera_list()
active_cameras = fetch_cameras()
network = fetch_network()
speed_readings = fetch_speed()

summary = network.get("summary", {})
speed_profile = network.get("speed_profile", {})

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_map, tab_feeds, tab_analytics = st.tabs(
    ["🗺️  Live Map", "📷  Camera Feeds", "📊  Network Analytics"]
)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — LIVE MAP
# ══════════════════════════════════════════════════════════════════════════════
with tab_map:
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

    total_vehicles = summary.get("total_vehicles", 0)
    active_count = summary.get("active_cameras", 0)
    avg_speed = summary.get("avg_speed_kmh", 0)
    network_los = summary.get("network_los", "?")
    congestion = summary.get("avg_congestion", 0)

    with kpi1:
        st.markdown(
            f"""
        <div class="kpi-card">
          <div class="kpi-value" style="color:#58a6ff">{total_vehicles}</div>
          <div class="kpi-label">Vehicles Detected</div>
        </div>""",
            unsafe_allow_html=True,
        )

    with kpi2:
        st.markdown(
            f"""
        <div class="kpi-card">
          <div class="kpi-value" style="color:#58a6ff">{active_count}</div>
          <div class="kpi-label">Active Cameras</div>
        </div>""",
            unsafe_allow_html=True,
        )

    with kpi3:
        spd_color = "#00e676" if avg_speed >= 60 else ("#ffeb3b" if avg_speed >= 40 else "#f44336")
        st.markdown(
            f"""
        <div class="kpi-card">
          <div class="kpi-value" style="color:{spd_color}">{avg_speed:.0f}</div>
          <div class="kpi-label">Avg Speed (km/h)</div>
        </div>""",
            unsafe_allow_html=True,
        )

    with kpi4:
        los_c = LOS_COLOR.get(network_los, "#78909c")
        st.markdown(
            f"""
        <div class="kpi-card">
          <div class="kpi-value" style="color:{los_c}">{network_los}</div>
          <div class="kpi-label">Network LOS</div>
        </div>""",
            unsafe_allow_html=True,
        )

    with kpi5:
        cong_pct = congestion * 100
        cong_c = "#00e676" if cong_pct < 30 else ("#ffeb3b" if cong_pct < 60 else "#f44336")
        st.markdown(
            f"""
        <div class="kpi-card">
          <div class="kpi-value" style="color:{cong_c}">{cong_pct:.0f}%</div>
          <div class="kpi-label">Congestion Index</div>
        </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # ── Folium map ─────────────────────────────────────────────────────────────
    m = folium.Map(
        location=SG_CENTER,
        zoom_start=11,
        tiles="CartoDB dark_matter",
        prefer_canvas=True,
    )

    # Expressway polylines
    for road, coords in EXPRESSWAYS.items():
        speed_data = speed_profile.get(road, {})
        road_speed = speed_data.get("avg_speed_kmh", 0)
        road_los = speed_data.get("los", "?")
        road_color = LOS_COLOR.get(road_los, ROAD_COLOR.get(road, "#546e7a"))
        opacity = 0.85 if road_los in ("E", "F") else 0.65

        tooltip_html = f"""
        <b style='color:{ROAD_COLOR.get(road, "#fff")}'>{road}</b><br>
        LOS: <b style='color:{road_color}'>{road_los} — {LOS_LABEL.get(road_los, "")}</b><br>
        Speed: <b>{road_speed:.0f} km/h</b>
        """
        folium.PolyLine(
            locations=coords,
            color=road_color,
            weight=5,
            opacity=opacity,
            tooltip=folium.Tooltip(tooltip_html),
        ).add_to(m)

        # Road label
        mid = coords[len(coords) // 2]
        folium.Marker(
            location=mid,
            icon=folium.DivIcon(
                html=f"""<div style="font-size:9px;font-weight:700;color:{ROAD_COLOR.get(road, "#fff")};
                         background:rgba(13,15,20,0.75);padding:1px 4px;border-radius:3px;
                         white-space:nowrap">{road}</div>""",
                icon_size=(40, 16),
                icon_anchor=(20, 8),
            ),
        ).add_to(m)

    # Camera markers
    active_ids = set(active_cameras.keys())
    for cam in all_cameras:
        cam_id = str(cam.get("camera_id", ""))
        lat = cam.get("latitude", 0)
        lng = cam.get("longitude", 0)

        if cam_id in active_cameras:
            state = active_cameras[cam_id]
            los = state.get("los", "?")
            veh = state.get("vehicle_count", 0)
            cong = state.get("congestion_index", 0)
            road = state.get("road", "")
            color = LOS_COLOR.get(los, "#78909c")
            radius = 10
            fill_opacity = 0.9
            tooltip_html = f"""
            <b>Camera {cam_id}</b> — {road}<br>
            LOS: <b style='color:{color}'>{los} {LOS_LABEL.get(los, "")}</b><br>
            Vehicles: <b>{veh}</b> &nbsp; Congestion: <b>{cong * 100:.0f}%</b>
            """
        else:
            color = "#37474f"
            radius = 5
            fill_opacity = 0.4
            tooltip_html = f"<b>Camera {cam_id}</b><br>No active data"

        folium.CircleMarker(
            location=[lat, lng],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=fill_opacity,
            weight=2,
            tooltip=folium.Tooltip(tooltip_html),
        ).add_to(m)

    # Legend
    legend_html = """
    <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
         background:rgba(22,27,34,0.92);border:1px solid #30363d;
         border-radius:8px;padding:12px 16px;font-size:11px;color:#e6edf3">
      <b style="font-size:12px">Level of Service</b><br><br>
    """
    for los_key, los_c in LOS_COLOR.items():
        if los_key == "?":
            continue
        legend_html += (
            f'<span style="color:{los_c}">●</span> &nbsp;{los_key} — {LOS_LABEL[los_key]}<br>'
        )
    legend_html += "</div>"
    m.get_root().html.add_child(folium.Element(legend_html))

    st_folium(m, width="100%", height=560, returned_objects=[])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — CAMERA FEEDS
# ══════════════════════════════════════════════════════════════════════════════
with tab_feeds:
    if not active_cameras:
        st.info("No active camera data yet — backend is still processing the first batch.")
    else:
        # Sort by congestion descending so busiest come first
        sorted_cams = sorted(
            active_cameras.items(),
            key=lambda kv: kv[1].get("congestion_index", 0),
            reverse=True,
        )

        cols_per_row = 3
        for row_start in range(0, len(sorted_cams), cols_per_row):
            row_cams = sorted_cams[row_start : row_start + cols_per_row]
            cols = st.columns(cols_per_row)
            for col, (cam_id, state) in zip(cols, row_cams, strict=False):
                los = state.get("los", "?")
                veh = state.get("vehicle_count", 0)
                road = state.get("road", "")
                cong = state.get("congestion_index", 0)
                spd = state.get("speed_kmh", 0)
                spd_l = state.get("speed_limit", 90)
                color = LOS_COLOR.get(los, "#78909c")
                with col:
                    st.markdown(
                        f"""<div class="cam-card" style="border-color:{color}">
                          <div class="cam-header">
                            <span>📍 {road} · Cam {cam_id}</span>
                            <span class="los-badge" style="background:{color}">LOS {los}</span>
                          </div>""",
                        unsafe_allow_html=True,
                    )
                    # Annotated camera image
                    st.image(
                        image_url(cam_id, raw=False),
                        use_container_width=True,
                        caption=None,
                    )
                    # Mini metrics
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Vehicles", veh)
                    m2.metric("Speed", f"{spd:.0f}/{spd_l}" if spd > 0 else "—")
                    m3.metric("Congestion", f"{cong * 100:.0f}%")
                    st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — NETWORK ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
with tab_analytics:
    if not speed_profile and not speed_readings:
        st.info("Speed analytics build up after 2+ minutes of data collection.")
    else:
        # ── Speed profile bar chart ────────────────────────────────────────────
        if speed_profile:
            st.markdown(
                '<div class="section-title">Road Speed Profile</div>', unsafe_allow_html=True
            )

            roads = list(speed_profile.keys())
            speeds = [speed_profile[r].get("avg_speed_kmh", 0) for r in roads]
            limits = [speed_profile[r].get("speed_limit", 90) for r in roads]
            los_list = [speed_profile[r].get("los", "?") for r in roads]
            bar_colors = [LOS_COLOR.get(los_key, "#546e7a") for los_key in los_list]

            fig_speed = go.Figure()
            fig_speed.add_trace(
                go.Bar(
                    x=roads,
                    y=speeds,
                    name="Avg Speed",
                    marker_color=bar_colors,
                    text=[f"{s:.0f}" for s in speeds],
                    textposition="outside",
                    textfont=dict(color="#e6edf3", size=11),
                )
            )
            fig_speed.add_trace(
                go.Scatter(
                    x=roads,
                    y=limits,
                    mode="markers+lines",
                    name="Speed Limit",
                    line=dict(color="#546e7a", dash="dash", width=1.5),
                    marker=dict(color="#546e7a", size=6),
                )
            )
            fig_speed.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e6edf3", size=12),
                height=320,
                margin=dict(l=10, r=10, t=10, b=10),
                legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#e6edf3")),
                xaxis=dict(gridcolor="#30363d", tickfont=dict(color="#8b949e")),
                yaxis=dict(
                    gridcolor="#30363d",
                    tickfont=dict(color="#8b949e"),
                    title="km/h",
                    range=[0, max(limits or [90]) * 1.15],
                ),
            )
            st.plotly_chart(fig_speed, use_container_width=True)

        # ── Two-column: LOS treemap + congestion heatmap ───────────────────────
        col_tree, col_heat = st.columns(2)

        with col_tree:
            if active_cameras:
                st.markdown(
                    '<div class="section-title">Camera LOS Distribution</div>',
                    unsafe_allow_html=True,
                )
                los_counts: dict[str, int] = {}
                for state in active_cameras.values():
                    los_key = state.get("los", "?")
                    los_counts[los_key] = los_counts.get(los_key, 0) + 1

                labels = [
                    f"LOS {k}<br>{n} cam{'s' if n != 1 else ''}" for k, n in los_counts.items()
                ]
                values = list(los_counts.values())
                colors = [LOS_COLOR.get(k, "#546e7a") for k in los_counts]

                fig_tree = go.Figure(
                    go.Treemap(
                        labels=labels,
                        parents=[""] * len(labels),
                        values=values,
                        marker=dict(colors=colors, line=dict(width=1, color="#0d0f14")),
                        textfont=dict(color="#0d0f14", size=13),
                    )
                )
                fig_tree.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=0, r=0, t=0, b=0),
                    height=260,
                )
                st.plotly_chart(fig_tree, use_container_width=True)

        with col_heat:
            if active_cameras:
                st.markdown(
                    '<div class="section-title">Congestion by Road</div>', unsafe_allow_html=True
                )

                # group congestion by road
                road_cong: dict[str, list[float]] = {}
                for state in active_cameras.values():
                    r = state.get("road", "Unknown")
                    c = state.get("congestion_index", 0)
                    road_cong.setdefault(r, []).append(c)

                road_avg = {r: sum(v) / len(v) for r, v in road_cong.items()}
                road_sorted = sorted(road_avg.items(), key=lambda x: x[1], reverse=True)

                r_names = [x[0] for x in road_sorted]
                r_vals = [x[1] * 100 for x in road_sorted]
                r_colors = [
                    "#f44336" if v >= 60 else ("#ff9800" if v >= 30 else "#00e676") for v in r_vals
                ]

                fig_cong = go.Figure(
                    go.Bar(
                        x=r_vals,
                        y=r_names,
                        orientation="h",
                        marker_color=r_colors,
                        text=[f"{v:.0f}%" for v in r_vals],
                        textposition="inside",
                        textfont=dict(color="#0d0f14", size=11),
                    )
                )
                fig_cong.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#e6edf3", size=11),
                    height=260,
                    margin=dict(l=0, r=0, t=0, b=0),
                    xaxis=dict(
                        gridcolor="#30363d",
                        range=[0, 105],
                        title="Congestion %",
                        tickfont=dict(color="#8b949e"),
                    ),
                    yaxis=dict(gridcolor="rgba(0,0,0,0)", tickfont=dict(color="#e6edf3")),
                )
                st.plotly_chart(fig_cong, use_container_width=True)

        # ── Speed readings table ───────────────────────────────────────────────
        if speed_readings:
            st.markdown(
                '<div class="section-title">Recent Speed Readings (Cross-Camera Re-ID)</div>',
                unsafe_allow_html=True,
            )

            rows = []
            for sr in speed_readings[-50:]:
                rows.append(
                    {
                        "Track ID": sr.get("track_id", ""),
                        "Road": sr.get("road", ""),
                        "From": sr.get("camera_id_entry", ""),
                        "To": sr.get("camera_id_exit", ""),
                        "Speed (km/h)": f"{sr.get('speed_kmh', 0):.1f}",
                        "Conf": f"{sr.get('reid_confidence', 0):.2f}",
                        "Time": sr.get("timestamp", "")[-8:] if sr.get("timestamp") else "",
                    }
                )

            df = pd.DataFrame(rows[::-1])  # newest first
            st.dataframe(
                df,
                use_container_width=True,
                height=280,
                hide_index=True,
            )

        # ── Network summary footer ─────────────────────────────────────────────
        if summary:
            st.markdown('<div class="section-title">Network Summary</div>', unsafe_allow_html=True)
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Total Vehicles", summary.get("total_vehicles", 0))
            s2.metric("Active Cameras", summary.get("active_cameras", 0))
            s3.metric("Avg Speed (km/h)", f"{summary.get('avg_speed_kmh', 0):.1f}")
            s4.metric("Heavy Vehicles %", f"{summary.get('avg_heavy_vehicle_ratio', 0) * 100:.1f}%")
