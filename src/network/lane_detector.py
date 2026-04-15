"""
Per-Camera Lane Detector
========================
Uses OpenCV Hough line detection to find lane markings in each LTA camera
image and build per-lane ROIs. Runs once per camera during network build,
results stored in camera_network.json.

Approach:
  1. Crop bottom 55% of image (road surface, skip sky/gantry overhead)
  2. Canny edge detection on grayscale
  3. HoughLinesP to find line segments
  4. Filter for lane-marking angles (20-80° from horizontal)
  5. Cluster line x-intercepts at image bottom to find lane boundaries
  6. Return N lane ROIs as (x1, y1, x2, y2) polygons in full-image coords

Lane count prior per road (Singapore LTA):
  CTE/PIE/AYE: 3 lanes per carriageway (sometimes 4 near junctions)
  ECP/MCE: 3 lanes
  TPE/BKE/KJE/SLE: 2-3 lanes
"""

from __future__ import annotations

import math
import numpy as np
from PIL import Image


# Expected lane count per road — used as clustering prior
ROAD_LANES: dict[str, int] = {
    "CTE": 3, "PIE": 3, "AYE": 3, "ECP": 3,
    "MCE": 3, "TPE": 3, "BKE": 2, "KJE": 2, "SLE": 2, "—": 3,
}


def detect_lanes(
    image: Image.Image,
    road: str = "—",
) -> list[dict]:
    """
    Detect lane ROIs in a single camera image.

    Returns list of lane dicts, ordered left to right:
      {
        "lane_idx": int,           # 0-indexed left to right
        "x_center": float,         # normalised [0, 1] centre of lane
        "x_left": float,           # normalised left boundary
        "x_right": float,          # normalised right boundary
        "direction": str,          # "unknown" until road context applied
      }
    Falls back to equal-width lane strips if detection fails.
    """
    try:
        import cv2
    except ImportError:
        return _equal_strips(ROAD_LANES.get(road, 3))

    img = np.array(image.convert("RGB"))
    h, w = img.shape[:2]

    # ── 1. Focus on road surface (bottom 55%, skip top gantry/sky) ─────────
    roi_y_start = int(h * 0.45)
    roi = img[roi_y_start:, :]
    roi_h = roi.shape[0]

    # ── 2. Edge detection ───────────────────────────────────────────────────
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    # Enhance contrast for faded lane markings
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # ── 3. Hough line detection ─────────────────────────────────────────────
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=30,
        minLineLength=int(roi_h * 0.15),
        maxLineGap=int(roi_h * 0.1),
    )

    if lines is None or len(lines) < 2:
        return _equal_strips(ROAD_LANES.get(road, 3))

    # ── 4. Filter lane-marking angles (roughly 15-80° from horizontal) ──────
    x_intercepts = []  # x-coordinate where each line hits the bottom of ROI
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x1 == x2:
            continue  # skip vertical lines
        angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
        if not (15 <= angle <= 80):
            continue
        # Extrapolate line to bottom of ROI
        if y2 != y1:
            x_at_bottom = x1 + (x2 - x1) * (roi_h - y1) / (y2 - y1)
        else:
            x_at_bottom = (x1 + x2) / 2
        if 0 <= x_at_bottom <= w:
            x_intercepts.append(x_at_bottom)

    if len(x_intercepts) < 2:
        return _equal_strips(ROAD_LANES.get(road, 3))

    # ── 5. Cluster intercepts → lane boundaries ──────────────────────────────
    n_lanes = ROAD_LANES.get(road, 3)
    boundaries = _cluster_to_boundaries(x_intercepts, n_lanes, w)

    # ── 6. Build lane ROIs ───────────────────────────────────────────────────
    lanes = []
    for i in range(len(boundaries) - 1):
        x_l = boundaries[i] / w
        x_r = boundaries[i + 1] / w
        lanes.append({
            "lane_idx": i,
            "x_center": (x_l + x_r) / 2,
            "x_left": x_l,
            "x_right": x_r,
            "direction": "unknown",
        })

    return lanes if lanes else _equal_strips(n_lanes)


def assign_lane(
    bbox: list[float],
    lanes: list[dict],
    img_width: int,
) -> int:
    """Return lane_idx for a detection bounding box (centre-x based)."""
    if not lanes:
        return 0
    cx = ((bbox[0] + bbox[2]) / 2) / img_width  # normalised centre x
    for lane in lanes:
        if lane["x_left"] <= cx < lane["x_right"]:
            return lane["lane_idx"]
    # Clamp to nearest edge lane
    if cx < lanes[0]["x_center"]:
        return lanes[0]["lane_idx"]
    return lanes[-1]["lane_idx"]


def apply_road_directions(lanes: list[dict], dir_a: str, dir_b: str) -> list[dict]:
    """
    Label lane directions based on road geometry.
    For dual-carriageway expressways the camera typically shows one side,
    so all lanes get dir_a. If the camera shows both carriageways (unusual),
    left half gets dir_b, right half gets dir_a (Singapore drives on left).
    """
    if not lanes:
        return lanes
    n = len(lanes)
    mid = n // 2
    for i, lane in enumerate(lanes):
        # Singapore drives on the left: leftmost lanes = slower/nearside
        # All lanes on one carriageway go the same direction
        # We label left half dir_b, right half dir_a as a default
        # (network build can override with OCR context)
        lane["direction"] = dir_b if i < mid else dir_a
    return lanes


# ── Internal helpers ───────────────────────────────────────────────────────────

def _equal_strips(n_lanes: int) -> list[dict]:
    """Fallback: divide image into N equal vertical strips."""
    return [
        {
            "lane_idx": i,
            "x_center": (i + 0.5) / n_lanes,
            "x_left": i / n_lanes,
            "x_right": (i + 1) / n_lanes,
            "direction": "unknown",
        }
        for i in range(n_lanes)
    ]


def _cluster_to_boundaries(
    intercepts: list[float],
    n_lanes: int,
    width: int,
) -> list[float]:
    """
    Convert raw line x-intercepts to N+1 lane boundary x-coordinates.
    Uses simple k-means-style clustering.
    """
    pts = np.array(sorted(set(intercepts)))

    # Always include image edges
    boundaries = [0.0]

    # Try to find n_lanes-1 interior boundaries
    if len(pts) >= n_lanes - 1:
        # Equal-spacing initialisation for k-means
        centers = np.linspace(width / (n_lanes + 1), width * n_lanes / (n_lanes + 1), n_lanes - 1)
        for _ in range(10):
            # Assign each point to nearest centre
            dists = np.abs(pts[:, None] - centers[None, :])
            labels = dists.argmin(axis=1)
            new_centers = np.array([
                pts[labels == k].mean() if np.any(labels == k) else centers[k]
                for k in range(len(centers))
            ])
            if np.allclose(centers, new_centers, atol=1.0):
                break
            centers = new_centers
        boundaries.extend(sorted(centers.tolist()))

    boundaries.append(float(width))
    return boundaries
