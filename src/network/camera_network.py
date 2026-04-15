"""
Singapore LTA Camera Road Network
==================================
Builds a directed graph of all 90 LTA cameras as nodes, connected by road
topology. Direction labels are extracted once via OCR from the text overlaid
on each camera image, then stored in camera_network.json for all future runs.

Graph structure:
    - Node: camera_id
      attrs: road, lat, lon, dir_a_label, dir_b_label
    - Edge: (cam_i → cam_j) on the same road, ordered by position along road
      attrs: road, distance_km

Usage:
    net = CameraNetwork()
    net.build(cameras, load_image_fn)   # one-time, saves JSON
    # or
    net = CameraNetwork.load()          # subsequent runs

    label_a, label_b = net.direction_labels("1703")
    # e.g. ("towards Woodlands", "towards City")
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Callable

import networkx as nx
from PIL import Image

NETWORK_PATH = Path("/tmp/camera_network.json")

# Known Singapore expressway direction labels (fallback if OCR fails)
# Format: road -> (positive-lat direction, negative-lat direction)
# Singapore runs roughly N-S and E-W
_FALLBACK_DIRECTIONS: dict[str, tuple[str, str]] = {
    "CTE": ("towards Woodlands", "towards City"),
    "PIE": ("towards Tuas", "towards Changi"),
    "AYE": ("towards Tuas", "towards City"),
    "ECP": ("towards Changi", "towards City"),
    "MCE": ("towards Marina East", "towards HarbourFront"),
    "TPE": ("towards Punggol", "towards PIE"),
    "BKE": ("towards Woodlands", "towards PIE"),
    "KJE": ("towards Kranji", "towards PIE"),
    "SLE": ("towards Woodlands", "towards TPE"),
    "—":   ("Direction A", "Direction B"),
}

# Regex patterns to find direction text in LTA image overlays
_DIR_PATTERNS = [
    r"towards?\s+([A-Za-z\s]+?)(?:\s*$|\s+\d)",
    r"to\s+([A-Za-z\s]+?)(?:\s*$|\s+\d)",
    r"([A-Za-z\s]+?)\s*(?:bound|BND)",
]


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance in km between two lat/lon points."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def _ocr_direction(image: Image.Image, road: str) -> tuple[str, str]:
    """Extract direction text from camera image overlay using EasyOCR.
    Falls back to known labels if OCR finds nothing useful.
    """
    try:
        import easyocr
        import numpy as np

        reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        # Crop top 15% and bottom 10% where LTA overlays direction text
        w, h = image.size
        top_crop = image.crop((0, 0, w, int(h * 0.15)))
        bot_crop = image.crop((0, int(h * 0.90), w, h))

        directions = []
        for crop in [top_crop, bot_crop]:
            results = reader.readtext(np.array(crop), detail=0)
            for text in results:
                text = text.strip()
                for pattern in _DIR_PATTERNS:
                    match = re.search(pattern, text, re.IGNORECASE)
                    if match:
                        directions.append(match.group(1).strip().title())

        if len(directions) >= 2:
            return f"towards {directions[0]}", f"towards {directions[1]}"
        elif len(directions) == 1:
            fallback = _FALLBACK_DIRECTIONS.get(road, ("Direction A", "Direction B"))
            return f"towards {directions[0]}", fallback[1]
    except Exception:
        pass

    return _FALLBACK_DIRECTIONS.get(road, ("Direction A", "Direction B"))


class CameraNetwork:
    """Directed graph of Singapore LTA cameras connected by road topology."""

    def __init__(self):
        self.graph = nx.DiGraph()
        self._labels: dict[str, tuple[str, str]] = {}  # cam_id -> (dir_a, dir_b)

    def build(
        self,
        cameras: list[dict],
        load_image_fn: Callable[[str], Image.Image | None],
        save: bool = True,
    ) -> None:
        """Build network from camera list. Runs OCR on each camera image once."""
        print(f"[CameraNetwork] Building from {len(cameras)} cameras...")

        # Group cameras by road and sort by lat (proxy for position along road)
        by_road: dict[str, list[dict]] = {}
        for cam in cameras:
            from app import _cam_road  # avoid circular at module level
            cam_id = cam.get("camera_id", "")
            road = _cam_road(cam_id)
            loc = cam.get("location", {})
            lat = loc.get("latitude", 0)
            lon = loc.get("longitude", 0)
            by_road.setdefault(road, []).append({
                "id": cam_id,
                "road": road,
                "lat": lat,
                "lon": lon,
                "img_url": cam.get("image", ""),
            })

        for road, cams in by_road.items():
            # Sort cameras along the road by latitude (approximation)
            cams_sorted = sorted(cams, key=lambda c: c["lat"])

            for cam in cams_sorted:
                # OCR direction labels from image
                img = load_image_fn(cam["img_url"])
                if img:
                    dir_a, dir_b = _ocr_direction(img, road)
                else:
                    dir_a, dir_b = _FALLBACK_DIRECTIONS.get(road, ("Direction A", "Direction B"))

                self._labels[cam["id"]] = (dir_a, dir_b)
                self.graph.add_node(
                    cam["id"],
                    road=road,
                    lat=cam["lat"],
                    lon=cam["lon"],
                    dir_a=dir_a,
                    dir_b=dir_b,
                )

            # Connect consecutive cameras on same road
            for i in range(len(cams_sorted) - 1):
                a, b = cams_sorted[i], cams_sorted[i + 1]
                dist = _haversine(a["lat"], a["lon"], b["lat"], b["lon"])
                self.graph.add_edge(a["id"], b["id"], road=road, distance_km=round(dist, 3))
                self.graph.add_edge(b["id"], a["id"], road=road, distance_km=round(dist, 3))

        print(f"[CameraNetwork] Built: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")

        if save:
            self._save()

    def direction_labels(self, cam_id: str) -> tuple[str, str]:
        """Return (dir_a_label, dir_b_label) for a camera."""
        return self._labels.get(cam_id, ("Direction A", "Direction B"))

    def neighbours(self, cam_id: str) -> list[str]:
        """Return adjacent camera IDs on the same road."""
        return list(self.graph.successors(cam_id))

    def _save(self) -> None:
        data = {
            "nodes": {
                n: dict(self.graph.nodes[n])
                for n in self.graph.nodes
            },
            "edges": [
                {"from": u, "to": v, **self.graph.edges[u, v]}
                for u, v in self.graph.edges
            ],
        }
        NETWORK_PATH.write_text(json.dumps(data, indent=2))
        print(f"[CameraNetwork] Saved to {NETWORK_PATH}")

    @classmethod
    def load(cls) -> "CameraNetwork | None":
        """Load from saved JSON. Returns None if file doesn't exist."""
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
                net.graph.add_edge(edge["from"], edge["to"],
                                   road=edge.get("road", ""),
                                   distance_km=edge.get("distance_km", 0))
            print(f"[CameraNetwork] Loaded: {net.graph.number_of_nodes()} nodes")
            return net
        except Exception as e:
            print(f"[CameraNetwork] Load failed: {e}")
            return None

    def push_to_hub(self, token: str) -> None:
        """Push network JSON to HF dataset repo for persistence across restarts."""
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            api.upload_file(
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
        """Pull network JSON from HF Hub if not cached locally."""
        if NETWORK_PATH.exists():
            return cls.load()
        try:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(
                repo_id="SuhxsReddy/cati-singapore-dataset",
                filename="camera_network.json",
                repo_type="dataset",
                token=token,
            )
            import shutil
            shutil.copy(path, NETWORK_PATH)
            return cls.load()
        except Exception:
            return None
