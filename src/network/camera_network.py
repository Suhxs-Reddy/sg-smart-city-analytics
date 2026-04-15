"""
Singapore LTA Camera Road Network
==================================
Builds a properly directed graph of all 90 LTA cameras:

  - Nodes   : cameras (lat, lon, road, OCR direction labels)
  - Road edges : directed A→B and B→A along each expressway,
                 sorted by position on the road's principal axis (PCA),
                 not naively by latitude (fixes E-W roads like PIE/AYE)
  - Junction edges : cameras from different roads within JUNCTION_THRESHOLD_KM
                     get a cross-road junction edge (handles PIE↔CTE etc.)

Direction labels come from EasyOCR on the text overlay of each camera image.
Built once, persisted to HF Hub as camera_network.json.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Callable

import networkx as nx
import numpy as np
from PIL import Image

NETWORK_PATH = Path("/tmp/camera_network.json")
JUNCTION_THRESHOLD_KM = 0.5   # cameras within this distance get junction edges

# LTA camera ID prefix → expressway
# "2" prefix covers both upper-CTE and SLE — refined by lat/lon at build time
PREFIX_ROAD: dict[str, str] = {
    "1": "CTE",
    "2": "CTE",   # 2701-2708 area — some may be SLE; junction edges handle it
    "3": "ECP",
    "4": "PIE",
    "5": "AYE",
    "6": "ECP",   # 6702-6705 = MCE (handled separately)
    "7": "TPE",
    "8": "KJE",
    "9": "BKE",
}
MCE_IDS = {"6702", "6703", "6704", "6705"}

# Known fallback direction labels per road
# (positive-axis direction, negative-axis direction)
FALLBACK_DIRECTIONS: dict[str, tuple[str, str]] = {
    "CTE": ("towards Woodlands", "towards City"),
    "PIE": ("towards Changi",    "towards Tuas"),
    "AYE": ("towards City",      "towards Tuas"),
    "ECP": ("towards Changi",    "towards City"),
    "MCE": ("towards Marina East", "towards HarbourFront"),
    "TPE": ("towards Punggol",   "towards PIE"),
    "BKE": ("towards Woodlands", "towards PIE"),
    "KJE": ("towards Kranji",    "towards PIE"),
    "SLE": ("towards Woodlands", "towards TPE"),
    "—":   ("Direction A",       "Direction B"),
}

_DIR_PATTERNS = [
    r"towards?\s+([A-Za-z][A-Za-z\s]{2,20?)(?:\s*$|\s+\d)",
    r"to\s+([A-Za-z][A-Za-z\s]{2,20})(?:\s*$|\s+\d)",
    r"([A-Za-z][A-Za-z\s]{2,20})\s*(?:bound|BND)",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def cam_road(cam_id: str) -> str:
    if cam_id in MCE_IDS:
        return "MCE"
    return PREFIX_ROAD.get(cam_id[0], "—")


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat, dlon = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * \
        math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def _principal_axis(cams: list[dict]) -> np.ndarray:
    """Unit vector along the road's dominant direction via PCA on lat/lon."""
    if len(cams) < 2:
        return np.array([0.0, 1.0])  # default N-S
    pts = np.array([[c["lon"], c["lat"]] for c in cams])
    pts -= pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(pts, full_matrices=False)
    return Vt[0]  # first right singular vector = principal axis


def _sort_by_road_axis(cams: list[dict]) -> list[dict]:
    """Sort cameras by projection onto road's principal axis (not just lat)."""
    if len(cams) <= 1:
        return cams
    axis = _principal_axis(cams)
    lons = np.array([c["lon"] for c in cams])
    lats = np.array([c["lat"] for c in cams])
    center = np.array([lons.mean(), lats.mean()])
    projections = (np.column_stack([lons, lats]) - center) @ axis
    order = np.argsort(projections)
    return [cams[i] for i in order]


def _ocr_direction(image: Image.Image, road: str) -> tuple[str, str]:
    """Extract direction labels from LTA camera image text overlay."""
    try:
        import easyocr
        reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        w, h = image.size
        # LTA overlays direction text at top and bottom strips
        crops = [
            image.crop((0, 0, w, int(h * 0.15))),
            image.crop((0, int(h * 0.88), w, h)),
        ]
        found = []
        for crop in crops:
            for text in reader.readtext(np.array(crop), detail=0):
                text = text.strip()
                for pattern in _DIR_PATTERNS:
                    m = re.search(pattern, text, re.IGNORECASE)
                    if m:
                        found.append(m.group(1).strip().title())
        if len(found) >= 2:
            return f"towards {found[0]}", f"towards {found[1]}"
        if len(found) == 1:
            fb = FALLBACK_DIRECTIONS.get(road, ("Direction A", "Direction B"))
            return f"towards {found[0]}", fb[1]
    except Exception:
        pass
    return FALLBACK_DIRECTIONS.get(road, ("Direction A", "Direction B"))


# ── Network ───────────────────────────────────────────────────────────────────

class CameraNetwork:
    """
    Directed road network of Singapore LTA cameras.

    Graph edges:
      type="road"     — consecutive cameras on same expressway (both directions)
      type="junction" — cameras from different roads within JUNCTION_THRESHOLD_KM
    """

    def __init__(self):
        self.graph: nx.DiGraph = nx.DiGraph()
        self._labels: dict[str, tuple[str, str]] = {}

    # ── Build ─────────────────────────────────────────────────────────────────

    def build(
        self,
        cameras: list[dict],
        load_image_fn: Callable[[str], Image.Image | None],
        save: bool = True,
    ) -> None:
        print(f"[CameraNetwork] Building from {len(cameras)} cameras…")

        # Bucket cameras by road
        by_road: dict[str, list[dict]] = {}
        for cam in cameras:
            cam_id = str(cam.get("camera_id", ""))
            road = cam_road(cam_id)
            loc = cam.get("location", {})
            by_road.setdefault(road, []).append({
                "id": cam_id,
                "road": road,
                "lat": float(loc.get("latitude", 0)),
                "lon": float(loc.get("longitude", 0)),
                "img_url": cam.get("image", ""),
            })

        # Add nodes + directed road edges per expressway
        for road, cams in by_road.items():
            ordered = _sort_by_road_axis(cams)

            for cam in ordered:
                img = load_image_fn(cam["img_url"])
                dir_a, dir_b = _ocr_direction(img, road) if img else \
                    FALLBACK_DIRECTIONS.get(road, ("Direction A", "Direction B"))

                self._labels[cam["id"]] = (dir_a, dir_b)
                self.graph.add_node(
                    cam["id"],
                    road=road,
                    lat=cam["lat"],
                    lon=cam["lon"],
                    dir_a=dir_a,
                    dir_b=dir_b,
                )

            # Directed edges: both A→B (dir_a flow) and B→A (dir_b flow)
            for i in range(len(ordered) - 1):
                a, b = ordered[i], ordered[i + 1]
                dist = round(haversine(a["lat"], a["lon"], b["lat"], b["lon"]), 3)
                # A→B = positive axis direction = dir_a
                self.graph.add_edge(a["id"], b["id"],
                                    type="road", road=road,
                                    direction="dir_a", distance_km=dist)
                # B→A = negative axis direction = dir_b
                self.graph.add_edge(b["id"], a["id"],
                                    type="road", road=road,
                                    direction="dir_b", distance_km=dist)

        # Junction edges: cross-road cameras within threshold
        self._add_junction_edges()

        n_road = sum(1 for _, _, d in self.graph.edges(data=True) if d.get("type") == "road")
        n_junc = sum(1 for _, _, d in self.graph.edges(data=True) if d.get("type") == "junction")
        print(f"[CameraNetwork] {self.graph.number_of_nodes()} nodes | "
              f"{n_road} road edges | {n_junc} junction edges")

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
                    self.graph.add_edge(id1, id2,
                                        type="junction", distance_km=round(dist, 3))
                    self.graph.add_edge(id2, id1,
                                        type="junction", distance_km=round(dist, 3))

    # ── Query ─────────────────────────────────────────────────────────────────

    def direction_labels(self, cam_id: str) -> tuple[str, str]:
        return self._labels.get(str(cam_id),
                                FALLBACK_DIRECTIONS.get(
                                    self.graph.nodes.get(str(cam_id), {}).get("road", "—"),
                                    ("Direction A", "Direction B")))

    def road_cameras(self, road: str) -> list[str]:
        """Cameras on a road in order (positive axis direction)."""
        return [n for n, d in self.graph.nodes(data=True) if d.get("road") == road]

    def junction_cameras(self, cam_id: str) -> list[str]:
        """Cameras on other roads connected to this camera via a junction."""
        return [v for _, v, d in self.graph.out_edges(str(cam_id), data=True)
                if d.get("type") == "junction"]

    def stats(self) -> dict:
        by_road: dict[str, int] = {}
        for _, d in self.graph.nodes(data=True):
            by_road[d.get("road", "—")] = by_road.get(d.get("road", "—"), 0) + 1
        road_edges = sum(1 for _, _, d in self.graph.edges(data=True) if d.get("type") == "road")
        junc_edges = sum(1 for _, _, d in self.graph.edges(data=True) if d.get("type") == "junction")
        return {"nodes": self.graph.number_of_nodes(),
                "road_edges": road_edges,
                "junction_edges": junc_edges,
                "cameras_by_road": by_road}

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save(self) -> None:
        data = {
            "nodes": {n: dict(self.graph.nodes[n]) for n in self.graph.nodes},
            "edges": [{"from": u, "to": v, **self.graph.edges[u, v]}
                      for u, v in self.graph.edges],
        }
        NETWORK_PATH.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls) -> "CameraNetwork | None":
        if not NETWORK_PATH.exists():
            return None
        try:
            data = json.loads(NETWORK_PATH.read_text())
            net = cls()
            for node_id, attrs in data["nodes"].items():
                net.graph.add_node(node_id, **attrs)
                net._labels[node_id] = (attrs.get("dir_a", "Direction A"),
                                        attrs.get("dir_b", "Direction B"))
            for edge in data["edges"]:
                src, dst = edge.pop("from"), edge.pop("to")
                net.graph.add_edge(src, dst, **edge)
            print(f"[CameraNetwork] Loaded: {net.stats()}")
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
                repo_type="dataset",
                token=token,
            )
            print("[CameraNetwork] Pushed to HF Hub")
        except Exception as e:
            print(f"[CameraNetwork] Hub push failed: {e}")

    @classmethod
    def load_from_hub(cls, token: str | None = None) -> "CameraNetwork | None":
        if NETWORK_PATH.exists():
            return cls.load()
        try:
            from huggingface_hub import hf_hub_download
            import shutil
            path = hf_hub_download(
                repo_id="SuhxsReddy/cati-singapore-dataset",
                filename="camera_network.json",
                repo_type="dataset",
                token=token,
            )
            shutil.copy(path, NETWORK_PATH)
            return cls.load()
        except Exception:
            return None
