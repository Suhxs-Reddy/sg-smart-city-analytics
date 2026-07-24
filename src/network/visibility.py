"""
Camera Visibility Analyzer
===========================
Determines what each LTA camera can actually see. Ground truth comes from
camera_config.json (verified by hand from actual LTA image overlays):

  - dir_a_x, dir_b_x    — mainline arrow x positions (y assumed ≈ 0.5)
  - extra_directions[]  — junction/ramp arrows with {label, source_road, x_norm, y_norm}

Each visible direction carries an image-space anchor (x_norm, y_norm),
so detections are assigned to the nearest anchor — replacing the old
"y > 0.4 = mainline" heuristic with proper ground-truth placement.

Graph-derived visibility (from junction edges to neighbouring cameras)
is used only when config is missing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import networkx as nx

# Extras at y_norm ≥ this are destination signposts (e.g. "towards Tuas" arrow
# painted above the same lane as the mainline direction). Camera cannot visually
# split them from the mainline, so we alias them rather than count twice.
SIGNPOST_Y_THRESHOLD = 0.45


def analyse_visibility(
    graph: nx.DiGraph,
    cam_id: str,
) -> list[dict]:
    """
    Return all traffic directions visible from this camera, each with an
    image-space anchor (x_norm, y_norm) for detection assignment.

    Preference order:
      1. Ground-truth config: mainline dir_a/dir_b at (dir_a_x, 0.5) / (dir_b_x, 0.5),
         plus extra_directions verbatim.
      2. Fallback: graph-derived junction neighbours (when config absent).

    Each direction dict:
      {label, source_road, type, x_norm, y_norm, order}
        type ∈ {"mainline", "ramp", "merge"}
    """
    node = graph.nodes.get(str(cam_id))
    if not node:
        return []

    directions: list[dict] = []
    seen_labels: set[str] = set()

    own_dir_a = node.get("dir_a", "Direction A")
    own_dir_b = node.get("dir_b", "Direction B")
    own_road = node.get("road", "—")
    dir_a_x = float(node.get("dir_a_x", 0.75))
    dir_b_x = float(node.get("dir_b_x", 0.25))
    dir_a_y = float(node.get("dir_a_y", 0.5))
    dir_b_y = float(node.get("dir_b_y", 0.5))
    extra = node.get("extra_directions", []) or []

    # 1. Mainline — always visible, anchored at (dir_?_x, dir_?_y).
    # For head-on cameras, dir_a_y ≠ dir_b_y splits near-vs-far lanes.
    for label, x, y in [(own_dir_a, dir_a_x, dir_a_y), (own_dir_b, dir_b_x, dir_b_y)]:
        if label and label not in seen_labels:
            directions.append(
                {
                    "label": label,
                    "source_road": own_road,
                    "type": "mainline",
                    "x_norm": x,
                    "y_norm": y,
                    "order": len(directions),
                }
            )
            seen_labels.add(label)

    # 2. Ground-truth extra directions (junction/ramp arrows verified on image).
    # Extras at y ≥ SIGNPOST_Y_THRESHOLD are signposts painted above mainline
    # lanes — they cannot be visually separated from the mainline count, so
    # they are skipped rather than double-counted.
    for ed in extra:
        label = ed.get("label")
        if not label or label in seen_labels:
            continue
        y_norm = float(ed.get("y_norm", 0.3))
        if y_norm >= SIGNPOST_Y_THRESHOLD:
            continue
        # Upper-frame arrows (y < 0.35) are ramps; mid-frame arrows are merges.
        direction_type = "ramp" if y_norm < 0.35 else "merge"
        directions.append(
            {
                "label": label,
                "source_road": ed.get("source_road", own_road),
                "type": direction_type,
                "x_norm": float(ed.get("x_norm", 0.5)),
                "y_norm": y_norm,
                "order": len(directions),
            }
        )
        seen_labels.add(label)

    # Config is authoritative for all 90 cameras — no graph fallback.
    return directions


def is_junction_camera(graph: nx.DiGraph, cam_id: str) -> bool:
    """True if camera has junction edges to cameras on other roads."""
    own_road = graph.nodes.get(str(cam_id), {}).get("road", "—")
    for _, _, d in graph.out_edges(str(cam_id), data=True):
        if d.get("type") == "junction":
            nb_road = graph.nodes.get(_, {}).get("road", "—")
            if nb_road != own_road:
                return True
    return False


_ASPECT = 1.78  # typical LTA frame aspect — y spread counts ~1.78x as much as x
# since lanes spread horizontally but ramps sit further up vertically


def assign_detection_direction(
    bbox: list[float],
    img_width: int,
    img_height: int,
    lanes: list[dict],
    visible_directions: list[dict],
) -> str:
    """
    Assign a detection to the visible direction whose ground-truth anchor
    (x_norm, y_norm) is closest to the detection centre.

    Weighting: y distance is scaled up because ramps are vertically distinct
    from mainline, while mainline lanes are horizontally distinct.
    """
    if not visible_directions:
        return "unknown"

    cx_n = ((bbox[0] + bbox[2]) / 2) / max(img_width, 1)
    cy_n = ((bbox[1] + bbox[3]) / 2) / max(img_height, 1)

    # Nearest anchor in weighted normalized space
    best_label = visible_directions[0]["label"]
    best_d2 = float("inf")
    for d in visible_directions:
        dx = cx_n - float(d.get("x_norm", 0.5))
        dy = (cy_n - float(d.get("y_norm", 0.5))) * _ASPECT
        d2 = dx * dx + dy * dy
        if d2 < best_d2:
            best_d2 = d2
            best_label = d["label"]
    return best_label


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
