"""
Camera Visibility Analyzer
===========================
Determines what each LTA camera can actually see:
  - Which roads/lanes are in its field of view
  - All possible traffic directions (not just 2)
  - Whether it's a simple mainline camera or a junction camera

For a mainline camera (no junction edges):
  visible_directions = [dir_a, dir_b]  →  2 directions

For a junction camera (has cross-road junction edges):
  visible_directions = [main_dir_a, main_dir_b, ramp_dir_1, ...]
  = 3-5 directions depending on interchange complexity

This is the foundation that lane detection and directional counting
are built on top of.

Output per camera stored in camera_network.json:
  {
    "visible_directions": [
      {"label": "PIE towards Changi",  "source": "PIE",  "type": "mainline"},
      {"label": "PIE towards Tuas",    "source": "PIE",  "type": "mainline"},
      {"label": "CTE towards City",    "source": "CTE",  "type": "ramp"},
    ]
  }
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import networkx as nx


def analyse_visibility(
    graph: "nx.DiGraph",
    cam_id: str,
) -> list[dict]:
    """
    Return all traffic directions visible from this camera.

    A direction is visible if:
      - It is the camera's own road direction (always visible), OR
      - A junction neighbour's road passes within JUNCTION_THRESHOLD_KM
        (camera can see ramp/merge traffic from the neighbouring road)

    Returns list of direction dicts:
      {
        label: str,      # e.g. "PIE towards Changi"
        source_road: str,
        type: "mainline" | "ramp" | "merge",
        order: int,      # lower = more prominent in frame (mainline first)
      }
    """
    node = graph.nodes.get(str(cam_id))
    if not node:
        return []

    directions: list[dict] = []
    seen_labels: set[str] = set()

    # 1. Own road — always visible, always first
    own_dir_a = node.get("dir_a", "Direction A")
    own_dir_b = node.get("dir_b", "Direction B")
    own_road  = node.get("road", "—")

    for label in [own_dir_a, own_dir_b]:
        if label not in seen_labels:
            directions.append({
                "label": label,
                "source_road": own_road,
                "type": "mainline",
                "order": len(directions),
            })
            seen_labels.add(label)

    # 2. Junction neighbours — ramp/merge directions
    for _, neighbour_id, edge_data in graph.out_edges(str(cam_id), data=True):
        if edge_data.get("type") != "junction":
            continue

        nb = graph.nodes.get(neighbour_id, {})
        nb_road = nb.get("road", "—")
        if nb_road == own_road:
            continue

        # Classify direction type by proximity
        dist = edge_data.get("distance_km", 1.0)
        direction_type = "ramp" if dist < 0.25 else "merge"

        for label in [nb.get("dir_a"), nb.get("dir_b")]:
            if label and label not in seen_labels:
                directions.append({
                    "label": label,
                    "source_road": nb_road,
                    "type": direction_type,
                    "order": len(directions),
                })
                seen_labels.add(label)

    return directions


def is_junction_camera(graph: "nx.DiGraph", cam_id: str) -> bool:
    """True if camera has junction edges to cameras on other roads."""
    own_road = graph.nodes.get(str(cam_id), {}).get("road", "—")
    for _, _, d in graph.out_edges(str(cam_id), data=True):
        if d.get("type") == "junction":
            nb_road = graph.nodes.get(_, {}).get("road", "—")
            if nb_road != own_road:
                return True
    return False


def assign_detection_direction(
    bbox: list[float],
    img_width: int,
    img_height: int,
    lanes: list[dict],
    visible_directions: list[dict],
) -> str:
    """
    Assign a detection to one of the camera's visible directions.

    Strategy:
      - Assign detection to a lane via bbox centre-X
      - Map lane direction label to visible_directions
      - For junction cameras: use bbox Y position to distinguish
        ramp traffic (typically upper portion of image, further away)
        from mainline traffic (lower portion, closer)
    """
    if not visible_directions:
        return "unknown"

    n_mainline = sum(1 for d in visible_directions if d["type"] == "mainline")
    n_ramp     = sum(1 for d in visible_directions if d["type"] in ("ramp", "merge"))

    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2

    # ── Simple 2-direction camera ────────────────────────────────────────────
    if n_ramp == 0:
        if lanes:
            from src.network.lane_detector import assign_lane
            li = assign_lane(bbox, lanes, img_width)
            lane = next((l for l in lanes if l["lane_idx"] == li), None)
            if lane:
                dir_label = lane.get("direction", "unknown")
                match = next((d for d in visible_directions if d["label"] == dir_label), None)
                if match:
                    return match["label"]
        # Fallback: left half = dir_b, right half = dir_a (Singapore drives left)
        return visible_directions[1]["label"] if cx < img_width / 2 else visible_directions[0]["label"]

    # ── Junction camera (N directions) ───────────────────────────────────────
    # Mainline vehicles: lower 60% of image (closer, larger boxes)
    # Ramp vehicles:     upper 40% of image (further, entering/exiting)
    y_norm = cy / img_height
    mainline_dirs = [d for d in visible_directions if d["type"] == "mainline"]
    ramp_dirs     = [d for d in visible_directions if d["type"] in ("ramp", "merge")]

    if y_norm > 0.4:
        # Mainline — assign by lane or x-position
        if lanes:
            from src.network.lane_detector import assign_lane
            li = assign_lane(bbox, lanes, img_width)
            lane = next((l for l in lanes if l["lane_idx"] == li), None)
            if lane and lane.get("direction") in {d["label"] for d in mainline_dirs}:
                return lane["direction"]
        # x-split among mainline directions
        if mainline_dirs:
            idx = min(int(cx / img_width * len(mainline_dirs)), len(mainline_dirs) - 1)
            return mainline_dirs[idx]["label"]

    # Ramp / upper portion — assign by x-position among ramp directions
    if ramp_dirs:
        idx = min(int(cx / img_width * len(ramp_dirs)), len(ramp_dirs) - 1)
        return ramp_dirs[idx]["label"]

    return visible_directions[0]["label"]


def summarise_directions(
    detections: list[dict],
    img_width: int,
    img_height: int,
    lanes: list[dict],
    visible_directions: list[dict],
) -> list[dict]:
    """
    Count detections per visible direction.

    Returns:
      [
        {"label": "PIE towards Changi", "source_road": "PIE",
         "type": "mainline", "count": 12},
        {"label": "CTE towards City",   "source_road": "CTE",
         "type": "ramp",     "count": 3},
        ...
      ]
    """
    tally: dict[str, int] = {d["label"]: 0 for d in visible_directions}

    for det in detections:
        label = assign_detection_direction(
            det["bbox"], img_width, img_height, lanes, visible_directions
        )
        tally[label] = tally.get(label, 0) + 1

    return [
        {**d, "count": tally.get(d["label"], 0)}
        for d in sorted(visible_directions, key=lambda x: x["order"])
    ]
