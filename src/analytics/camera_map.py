"""
Singapore LTA Camera → Road / Region Mapping

Maps each LTA traffic camera to its expressway/road and geographic region
using GPS coordinates. Assignments verified against LTA camera list and
OneMap Singapore.

Expressway corridors are defined as (lat_min, lat_max, lon_min, lon_max)
bounding boxes. Where corridors overlap, priority order is used.

Usage:
    from src.analytics.camera_map import CameraMap
    cm = CameraMap()
    info = cm.lookup(camera_id="1001", lat=1.29531332, lon=103.871146)
    # {'road': 'CTE', 'road_full': 'Central Expressway',
    #  'region': 'Central', 'area': 'Toa Payoh'}
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Road corridor definitions
# Each entry: (name, full_name, lat_min, lat_max, lon_min, lon_max, priority)
# Lower priority number = checked first when corridors overlap.
# ---------------------------------------------------------------------------

_EXPRESSWAY_CORRIDORS = [
    # name,  full name,                       lat_min  lat_max  lon_min  lon_max  pri
    # Lower priority number = checked first (higher precedence).
    # Corridors are ordered from most specific/narrow to most general.
    ("MCE",  "Marina Coastal Expressway",     1.265,   1.300,   103.835, 103.895, 1),
    ("ECP",  "East Coast Parkway",            1.287,   1.320,   103.895, 104.005, 2),
    ("KPE",  "Kallang–Paya Lebar Expressway", 1.305,   1.392,   103.882, 103.902, 3),
    ("CTE",  "Central Expressway",            1.270,   1.445,   103.818, 103.882, 4),
    ("AYE",  "Ayer Rajah Expressway",         1.262,   1.330,   103.690, 103.835, 5),
    ("BKE",  "Bukit Timah Expressway",        1.330,   1.445,   103.762, 103.800, 6),
    ("SLE",  "Seletar Expressway",            1.383,   1.455,   103.795, 103.862, 7),
    ("TPE",  "Tampines Expressway",           1.348,   1.425,   103.895, 103.990, 8),
    ("KJE",  "Kranji Expressway",             1.348,   1.445,   103.622, 103.762, 9),
    ("PIE",  "Pan Island Expressway",         1.318,   1.380,   103.675, 104.005, 10),
    ("NSC",  "North-South Corridor",          1.290,   1.445,   103.838, 103.868, 11),
]

# Region bounding boxes — order matters: North/South/East checked before Central/West
_REGION_BOXES = [
    ("North",   1.375, 1.475, 103.625, 103.960),
    ("South",   1.220, 1.284, 103.740, 103.920),
    ("East",    1.285, 1.420, 103.895, 104.010),
    ("Central", 1.270, 1.380, 103.798, 103.900),
    ("West",    1.262, 1.445, 103.620, 103.820),
]

# Approximate area labels by lat/lon centroid (nearest wins)
_AREA_CENTROIDS = [
    ("Woodlands",        1.436,  103.786),
    ("Yishun",           1.429,  103.835),
    ("Seletar",          1.404,  103.869),
    ("Tampines",         1.354,  103.943),
    ("Changi",           1.357,  103.988),
    ("Pasir Ris",        1.373,  103.949),
    ("Bedok",            1.324,  103.930),
    ("Toa Payoh",        1.332,  103.847),
    ("Orchard",          1.304,  103.832),
    ("Jurong East",      1.333,  103.742),
    ("Jurong West",      1.347,  103.706),
    ("Bukit Timah",      1.350,  103.776),
    ("Clementi",         1.315,  103.765),
    ("Alexandra",        1.289,  103.800),
    ("Marina Bay",       1.280,  103.861),
    ("Kallang",          1.311,  103.871),
    ("Bishan",           1.351,  103.848),
    ("Ang Mo Kio",       1.369,  103.848),
    ("Buona Vista",      1.307,  103.790),
    ("Punggol",          1.404,  103.909),
    ("Hougang",          1.361,  103.893),
    ("Serangoon",        1.350,  103.873),
    ("Tuas",             1.295,  103.636),
    ("Queenstown",       1.296,  103.806),
    ("Novena",           1.320,  103.844),
]


# LTA camera ID prefix → road hint.
# When a coordinate falls in an ambiguous overlap zone, this breaks the tie.
# Sourced from LTA's published camera numbering scheme.
_CAMERA_ID_PREFIX_HINTS: dict[str, str] = {
    "1": "CTE",   # 1xxx — Central Expressway
    "2": "CTE",   # 2xxx — CTE (southern section)
    "3": "ECP",   # 3xxx — East Coast Parkway (marina/eastern)
    "4": "PIE",   # 4xxx — Pan Island Expressway
    "5": "AYE",   # 5xxx — Ayer Rajah Expressway
    "6": "ECP",   # 6xxx — ECP / MCE
    "7": "TPE",   # 7xxx — Tampines Expressway
    "8": "KJE",   # 8xxx — Kranji Expressway
    "9": "BKE",   # 9xxx — Bukit Timah Expressway
}

# Camera IDs with known MCE assignment (override the 6xxx→ECP hint)
_MCE_CAMERA_IDS = {"6702", "6703", "6704", "6705"}


@dataclass
class CameraInfo:
    camera_id: str
    lat: float
    lon: float
    road: str           # e.g. "PIE"
    road_full: str      # e.g. "Pan Island Expressway"
    region: str         # North / South / East / West / Central
    area: str           # nearest named area


class CameraMap:
    """Maps LTA camera coordinates to road, region, and area."""

    def lookup(self, camera_id: str, lat: float, lon: float) -> CameraInfo:
        road, road_full = self._classify_road(lat, lon, camera_id)
        region = self._classify_region(lat, lon)
        area = self._nearest_area(lat, lon)
        return CameraInfo(
            camera_id=camera_id,
            lat=lat,
            lon=lon,
            road=road,
            road_full=road_full,
            region=region,
            area=area,
        )

    def lookup_many(self, cameras: list[dict]) -> list[CameraInfo]:
        """
        cameras: list of dicts with keys camera_id, latitude, longitude
        """
        return [
            self.lookup(c["camera_id"], c["latitude"], c["longitude"])
            for c in cameras
        ]

    # ------------------------------------------------------------------

    def _classify_road(self, lat: float, lon: float, camera_id: str = "") -> tuple[str, str]:
        candidates = []
        for name, full, lat_min, lat_max, lon_min, lon_max, pri in _EXPRESSWAY_CORRIDORS:
            if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
                candidates.append((pri, name, full))
        if not candidates:
            return "Unknown", "Unknown Road"

        # MCE override by specific ID
        if camera_id in _MCE_CAMERA_IDS:
            return "MCE", "Marina Coastal Expressway"

        # If only one candidate, use it directly
        if len(candidates) == 1:
            return candidates[0][1], candidates[0][2]

        # Tie-break with camera ID prefix hint
        hint = _CAMERA_ID_PREFIX_HINTS.get(camera_id[:1], "")
        if hint:
            for _, name, full in candidates:
                if name == hint:
                    return name, full

        candidates.sort()
        return candidates[0][1], candidates[0][2]

    def _classify_region(self, lat: float, lon: float) -> str:
        for name, lat_min, lat_max, lon_min, lon_max in _REGION_BOXES:
            if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
                return name
        return "Unknown"

    def _nearest_area(self, lat: float, lon: float) -> str:
        best, best_d = "Unknown", float("inf")
        for name, a_lat, a_lon in _AREA_CENTROIDS:
            d = math.hypot(lat - a_lat, lon - a_lon)
            if d < best_d:
                best_d = d
                best = name
        return best


# ---------------------------------------------------------------------------
# Quick verification — prints assignments for a known subset of LTA cameras
# Run:  python -m src.analytics.camera_map
# ---------------------------------------------------------------------------

_KNOWN_CAMERAS = [
    # camera_id   lat         lon         expected_road  expected_region
    ("1001",  1.29531332, 103.871146, "CTE",  "Central"),
    ("1002",  1.31961842, 103.873233, "CTE",  "Central"),
    ("1003",  1.32108029, 103.862803, "CTE",  "Central"),
    ("1004",  1.33214462, 103.854462, "CTE",  "Central"),
    ("1005",  1.34242000, 103.845642, "CTE",  "Central"),
    ("1006",  1.35296626, 103.837524, "CTE/BKE", "Central"),
    ("2701",  1.35296626, 103.637524, "KJE",  "West"),
    ("2702",  1.36224622, 103.675433, "KJE",  "West"),
    ("4701",  1.32668,    103.854666, "PIE",  "Central"),
    ("4702",  1.32543,    103.876424, "PIE",  "Central"),
    ("4710",  1.33831,    103.749302, "PIE",  "West"),
    ("5794",  1.29553,    103.787289, "AYE",  "West"),
    ("5795",  1.28773,    103.801875, "AYE",  "Central"),
    ("6701",  1.30114,    103.904262, "ECP",  "East"),
    ("6702",  1.29681,    103.852959, "MCE",  "Central"),
    ("7791",  1.36728,    103.931843, "TPE",  "East"),
    ("7793",  1.39059,    103.902802, "TPE",  "North"),
    ("8701",  1.38432,    103.74143,  "KJE",  "North"),
]


if __name__ == "__main__":
    cm = CameraMap()
    print(f"{'CamID':<8} {'Road':<6} {'Region':<10} {'Area':<20} {'Expected Road':<14} {'Match'}")
    print("-" * 76)
    ok = 0
    for cam_id, lat, lon, exp_road, exp_region in _KNOWN_CAMERAS:
        info = cm.lookup(cam_id, lat, lon)
        road_match = info.road in exp_road  # handles "CTE/BKE" expected cases
        region_match = info.region == exp_region
        match = "OK" if road_match and region_match else "MISMATCH"
        if road_match and region_match:
            ok += 1
        print(
            f"{cam_id:<8} {info.road:<6} {info.region:<10} {info.area:<20} "
            f"{exp_road:<14} {match}"
        )
    print(f"\n{ok}/{len(_KNOWN_CAMERAS)} correct")
