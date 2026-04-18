"""
Singapore LTA Camera Road Network
==================================
Directed graph of all 90 LTA cameras with:
  - Ground-truth direction labels from camera_config.json (read from actual LTA images)
  - PCA-based road axis sorting (correct for E-W roads like PIE/AYE)
  - Directed road edges: A→B (dir_a flow) and B→A (dir_b flow)
  - Junction edges: cross-road cameras within JUNCTION_THRESHOLD_KM
  - Ramp camera flagging via proximity to same-road neighbours
  - Version field to detect stale cached JSON

NETWORK_VERSION must be bumped whenever build logic changes.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable

import networkx as nx
import numpy as np
from PIL import Image

NETWORK_PATH = Path("/tmp/camera_network.json")
NETWORK_VERSION = "6"               # bumped: wires arrow anchors + extra_directions into nodes
JUNCTION_THRESHOLD_KM = 0.5         # cross-road proximity for junction edges
RAMP_THRESHOLD_KM = 0.15            # same-road proximity to flag ramp candidates

# ── Ground-truth camera config (read from actual LTA image overlays) ───────────
_CONFIG_PATH = Path(__file__).parent / "camera_config.json"
_CAMERA_CONFIG: dict[str, dict] = {}
try:
    _CAMERA_CONFIG = json.loads(_CONFIG_PATH.read_text())
except Exception:
    pass  # falls back to prefix table + FALLBACK_DIRECTIONS below

# ── Camera ID → Road mapping (fallback for cameras not in config) ─────────────
_PREFIX_ROAD: dict[str, str] = {
    "1": "KPE",
    "2": "SLE",
    "3": "ECP",
    "4": "PIE",
    "5": "AYE",
    "6": "ECP",   # 6702-6705 = MCE handled separately
    "7": "TPE",
    "8": "KJE",
    "9": "BKE",
}
_MCE_IDS = {"6702", "6703", "6704", "6705"}


def cam_road(cam_id: str, lat: float = 0.0, lon: float = 0.0) -> str:
    """Authoritative camera → road mapping. Config takes priority over prefix table."""
    cfg = _CAMERA_CONFIG.get(str(cam_id))
    if cfg:
        return cfg["road"]
    if cam_id in _MCE_IDS:
        return "MCE"
    return _PREFIX_ROAD.get(cam_id[0] if cam_id else "", "—")


# ── Known fallback direction labels per road (used only when camera not in config) ──
FALLBACK_DIRECTIONS: dict[str, tuple[str, str]] = {
    "CTE": ("towards SLE",          "towards City"),
    "PIE": ("towards Changi",       "towards Jurong"),
    "AYE": ("towards Changi",       "towards Jurong"),
    "ECP": ("towards Changi",       "towards City"),
    "MCE": ("towards ECP",          "towards AYE"),
    "KPE": ("towards ECP",          "towards TPE"),
    "TPE": ("towards Punggol",      "towards PIE"),
    "BKE": ("towards TPE",          "towards Woodlands"),
    "KJE": ("towards PIE",          "towards Kranji"),
    "SLE": ("towards Woodlands",    "towards PIE"),
    "—":   ("Direction A",          "Direction B"),
}


# ── Geometry helpers ───────────────────────────────────────────────────────────

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


def _principal_axis(cams: list[dict]) -> np.ndarray:
    """Unit vector along road's dominant direction via PCA on lat/lon."""
    if len(cams) < 2:
        return np.array([0.0, 1.0])
    pts = np.array([[c["lon"], c["lat"]] for c in cams])
    pts -= pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(pts, full_matrices=False)
    return Vt[0]


def _sort_by_road_axis(cams: list[dict]) -> list[dict]:
    if len(cams) <= 1:
        return cams
    axis = _principal_axis(cams)
    lons = np.array([c["lon"] for c in cams])
    lats = np.array([c["lat"] for c in cams])
    center = np.array([lons.mean(), lats.mean()])
    projections = (np.column_stack([lons, lats]) - center) @ axis
    return [cams[i] for i in np.argsort(projections)]


# ── Camera Network ─────────────────────────────────────────────────────────────

class CameraNetwork:
    """Directed road network of Singapore LTA cameras."""

    def __init__(self):
        self.graph: nx.DiGraph = nx.DiGraph()
        self._labels: dict[str, tuple[str, str]] = {}

    # ── Build ──────────────────────────────────────────────────────────────────

    def build(
        self,
        cameras: list[dict],
        load_image_fn: Callable[[str], Image.Image | None],
        save: bool = True,
    ) -> None:
        print(f"[CameraNetwork] Building v{NETWORK_VERSION} from {len(cameras)} cameras…")

        # Bucket cameras by road using authoritative cam_road()
        by_road: dict[str, list[dict]] = {}
        for cam in cameras:
            cam_id = str(cam.get("camera_id", ""))
            loc = cam.get("location", {})
            lat = float(loc.get("latitude", 0))
            lon = float(loc.get("longitude", 0))
            road = cam_road(cam_id, lat, lon)
            by_road.setdefault(road, []).append({
                "id": cam_id, "road": road,
                "lat": lat, "lon": lon,
                "img_url": cam.get("image", ""),
            })

        # Add nodes + directed road edges
        for road, cams in by_road.items():
            ordered = _sort_by_road_axis(cams)

            # Detect ramp candidates: same-road neighbour within RAMP_THRESHOLD_KM
            ramp_candidates: set[str] = set()
            for i, a in enumerate(ordered):
                for b in ordered:
                    if a["id"] == b["id"]:
                        continue
                    if haversine(a["lat"], a["lon"], b["lat"], b["lon"]) < RAMP_THRESHOLD_KM:
                        ramp_candidates.add(a["id"])

            for cam in ordered:
                img = load_image_fn(cam["img_url"]) if cam["img_url"] else None

                # Direction labels: ground-truth config takes priority
                cfg = _CAMERA_CONFIG.get(cam["id"])
                if cfg:
                    dir_a, dir_b = cfg["dir_a"], cfg["dir_b"]
                    dir_a_x = float(cfg.get("dir_a_x", 0.75))
                    dir_b_x = float(cfg.get("dir_b_x", 0.25))
                    is_junction_cfg = cfg.get("is_junction", False)
                    junction_roads_cfg = cfg.get("junction_roads", [])
                    extra_dirs_cfg = cfg.get("extra_directions", []) or []
                else:
                    dir_a, dir_b = FALLBACK_DIRECTIONS.get(road, ("Direction A", "Direction B"))
                    dir_a_x, dir_b_x = 0.75, 0.25
                    is_junction_cfg = False
                    junction_roads_cfg = []
                    extra_dirs_cfg = []
                self._labels[cam["id"]] = (dir_a, dir_b)

                # Lane detection
                from src.network.lane_detector import detect_lanes, apply_road_directions
                lanes = detect_lanes(img, road) if img else []
                lanes = apply_road_directions(lanes, dir_a, dir_b)

                self.graph.add_node(
                    cam["id"],
                    road=road,
                    lat=cam["lat"],
                    lon=cam["lon"],
                    dir_a=dir_a,
                    dir_b=dir_b,
                    dir_a_x=dir_a_x,
                    dir_b_x=dir_b_x,
                    extra_directions=extra_dirs_cfg,
                    is_ramp=cam["id"] in ramp_candidates,
                    is_junction_cfg=is_junction_cfg,
                    junction_roads=junction_roads_cfg,
                    lanes=lanes,
                    n_lanes=len(lanes),
                )

            # Directed road edges
            for i in range(len(ordered) - 1):
                a, b = ordered[i], ordered[i + 1]
                dist = round(haversine(a["lat"], a["lon"], b["lat"], b["lon"]), 3)
                self.graph.add_edge(a["id"], b["id"], type="road", road=road,
                                    direction="dir_a", distance_km=dist)
                self.graph.add_edge(b["id"], a["id"], type="road", road=road,
                                    direction="dir_b", distance_km=dist)

        # Junction edges across roads
        self._add_junction_edges()

        # Visibility analysis — run after junction edges are built
        from src.network.visibility import analyse_visibility
        for cam_id in self.graph.nodes:
            vis = analyse_visibility(self.graph, cam_id)
            self.graph.nodes[cam_id]["visible_directions"] = vis
            # A camera is a junction camera if: graph analysis OR config says so
            is_junc_graph = len(vis) > 2
            is_junc_cfg   = self.graph.nodes[cam_id].get("is_junction_cfg", False)
            self.graph.nodes[cam_id]["is_junction_camera"] = is_junc_graph or is_junc_cfg

        s = self.stats()
        print(f"[CameraNetwork] {s['nodes']} nodes | "
              f"{s['road_edges']} road edges | {s['junction_edges']} junction edges | "
              f"{s['ramp_candidates']} ramp candidates")

        if save:
            self._save()

    def _add_junction_edges(self) -> None:
        nodes = list(self.graph.nodes(data=True))
        for i, (id1, d1) in enumerate(nodes):
            for id2, d2 in nodes[i + 1:]:
                if d1["road"] == d2["road"]:
                    continue
                dist = haversine(d1["lat"], d1["lon"], d2["lat"], d2["lon"])
                if dist <= JUNCTION_THRESHOLD_KM:
                    self.graph.add_edge(id1, id2, type="junction",
                                        distance_km=round(dist, 3))
                    self.graph.add_edge(id2, id1, type="junction",
                                        distance_km=round(dist, 3))

    # ── Query ──────────────────────────────────────────────────────────────────

    def direction_labels(self, cam_id: str) -> tuple[str, str]:
        node = self.graph.nodes.get(str(cam_id), {})
        return self._labels.get(
            str(cam_id),
            FALLBACK_DIRECTIONS.get(node.get("road", "—"), ("Direction A", "Direction B")),
        )

    def is_ramp(self, cam_id: str) -> bool:
        return self.graph.nodes.get(str(cam_id), {}).get("is_ramp", False)

    def visible_directions(self, cam_id: str) -> list[dict]:
        return self.graph.nodes.get(str(cam_id), {}).get("visible_directions", [])

    def is_junction_camera(self, cam_id: str) -> bool:
        return self.graph.nodes.get(str(cam_id), {}).get("is_junction_camera", False)

    def lanes(self, cam_id: str) -> list[dict]:
        return self.graph.nodes.get(str(cam_id), {}).get("lanes", [])

    def n_lanes(self, cam_id: str) -> int:
        return self.graph.nodes.get(str(cam_id), {}).get("n_lanes", 0)

    def stats(self) -> dict:
        by_road: dict[str, int] = {}
        ramp_count = 0
        for _, d in self.graph.nodes(data=True):
            r = d.get("road", "—")
            by_road[r] = by_road.get(r, 0) + 1
            if d.get("is_ramp"):
                ramp_count += 1
        road_e = sum(1 for *_, d in self.graph.edges(data=True) if d.get("type") == "road")
        junc_e = sum(1 for *_, d in self.graph.edges(data=True) if d.get("type") == "junction")
        return {"nodes": self.graph.number_of_nodes(), "road_edges": road_e,
                "junction_edges": junc_e, "ramp_candidates": ramp_count,
                "cameras_by_road": by_road}

    # ── Persistence ────────────────────────────────────────────────────────────

    def _save(self) -> None:
        data = {
            "version": NETWORK_VERSION,
            "nodes": {n: dict(self.graph.nodes[n]) for n in self.graph.nodes},
            "edges": [{"from": u, "to": v, **self.graph.edges[u, v]}
                      for u, v in self.graph.edges],
        }
        NETWORK_PATH.write_text(json.dumps(data, indent=2))
        print(f"[CameraNetwork] Saved to {NETWORK_PATH}")

    @classmethod
    def load(cls) -> "CameraNetwork | None":
        if not NETWORK_PATH.exists():
            return None
        try:
            data = json.loads(NETWORK_PATH.read_text())
            # Version check — stale network triggers rebuild
            if data.get("version") != NETWORK_VERSION:
                print(f"[CameraNetwork] Stale version {data.get('version')} != {NETWORK_VERSION}, rebuilding")
                NETWORK_PATH.unlink()
                return None
            net = cls()
            for node_id, attrs in data["nodes"].items():
                net.graph.add_node(node_id, **attrs)
                net._labels[node_id] = (attrs.get("dir_a", "Direction A"),
                                        attrs.get("dir_b", "Direction B"))
            for edge in data["edges"]:
                src, dst = edge.pop("from"), edge.pop("to")
                net.graph.add_edge(src, dst, **edge)
            print(f"[CameraNetwork] Loaded v{NETWORK_VERSION}: {net.stats()}")
            return net
        except Exception as e:
            print(f"[CameraNetwork] Load failed: {e}")
            return None

    def push_to_hub(self, token: str | None) -> None:
        if not token:
            return
        try:
            from huggingface_hub import HfApi
            HfApi().upload_file(
                path_or_fileobj=str(NETWORK_PATH),
                path_in_repo="camera_network.json",
                repo_id="SuhxsReddy/cati-singapore-dataset",
                repo_type="dataset", token=token,
            )
            print("[CameraNetwork] Pushed to HF Hub")
        except Exception as e:
            print(f"[CameraNetwork] Hub push failed: {e}")

    @classmethod
    def load_from_hub(cls, token: str | None = None) -> "CameraNetwork | None":
        # Try local first
        local = cls.load()
        if local:
            return local
        # Pull from HF Hub
        try:
            from huggingface_hub import hf_hub_download
            import shutil
            path = hf_hub_download(
                repo_id="SuhxsReddy/cati-singapore-dataset",
                filename="camera_network.json",
                repo_type="dataset", token=token,
            )
            shutil.copy(path, NETWORK_PATH)
            return cls.load()   # version check inside
        except Exception:
            return None
