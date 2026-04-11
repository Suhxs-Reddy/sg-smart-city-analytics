"""
Singapore Expressway Camera Network

Builds a graph of all 90 LTA cameras where:
  - Nodes = cameras (road, region, lat/lon)
  - Edges = consecutive cameras on the same expressway, weighted by GPS distance (km)

The graph mirrors real road topology — cameras on the same expressway are
linked in travel order, enabling inter-camera speed estimation via:
    speed (km/h) = edge_distance_km / travel_time_hours

Usage:
    from src.analytics.camera_network import CameraNetwork
    net = CameraNetwork()
    print(net.summary())
    pairs = net.adjacent_pairs("CTE")   # [(cam_a, cam_b, dist_km), ...]
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from src.analytics.camera_map import CameraInfo, CameraMap

# ---------------------------------------------------------------------------
# Road sort axis — which coordinate to use when ordering cameras along a road
# ---------------------------------------------------------------------------
_ROAD_SORT_AXIS: dict[str, str] = {
    "CTE": "lat",  # N-S
    "BKE": "lat",  # N-S
    "KJE": "lat",  # N-S
    "KPE": "lat",  # N-S
    "NSC": "lat",  # N-S
    "PIE": "lon",  # E-W
    "AYE": "lon",  # E-W
    "ECP": "lon",  # E-W
    "TPE": "lon",  # E-W
    "SLE": "lon",  # E-W
    "MCE": "lon",  # E-W
}

# All 90 LTA cameras: (camera_id, latitude, longitude)
# Fetched from data.gov.sg/v1/transport/traffic-images on 2026-04-09
_LTA_CAMERAS: list[tuple[str, float, float]] = [
    ("1001", 1.29531332, 103.871146),
    ("1002", 1.319541067, 103.8785627),
    ("1003", 1.323957439, 103.8728576),
    ("1004", 1.319535712, 103.8750668),
    ("1005", 1.363519886, 103.905394),
    ("1006", 1.357098686, 103.902042),
    ("1111", 1.365434, 103.953997),
    ("1112", 1.3605, 103.961412),
    ("1113", 1.317036, 103.988598),
    ("1501", 1.27414394, 103.851316),
    ("1502", 1.27135090, 103.861828),
    ("1503", 1.27066408, 103.856977),
    ("1504", 1.29409891, 103.876056),
    ("1505", 1.27529771, 103.866390),
    ("1701", 1.323604823, 103.8587802),
    ("1702", 1.34355015, 103.8601984),
    ("1703", 1.32814722, 103.862203),
    ("1704", 1.28569398, 103.837524),
    ("1705", 1.375925022, 103.8587986),
    ("1706", 1.38861, 103.85806),
    ("1707", 1.28036584, 103.830451),
    ("1709", 1.31384231, 103.845603),
    ("1711", 1.35296, 103.85719),
    ("2701", 1.447023728, 103.7716543),
    ("2702", 1.445554109, 103.7683397),
    ("2703", 1.35047790, 103.791033),
    ("2704", 1.429588536, 103.769311),
    ("2705", 1.36728572, 103.7794698),
    ("2706", 1.414142, 103.771168),
    ("2707", 1.3983, 103.774247),
    ("2708", 1.3865, 103.7747),
    ("3702", 1.33831, 103.98032),
    ("3704", 1.29585501, 103.880314),
    ("3705", 1.32743, 103.97383),
    ("3793", 1.309330837, 103.9350504),
    ("3795", 1.30145145, 103.910596),
    ("3796", 1.297512569, 103.8983019),
    ("3797", 1.29565733, 103.885283),
    ("3798", 1.29158484, 103.8615987),
    ("4701", 1.2871, 103.79633),
    ("4702", 1.27237, 103.8324),
    ("4703", 1.348697862, 103.6350413),
    ("4704", 1.27877, 103.82375),
    ("4705", 1.32618, 103.73028),
    ("4706", 1.29792, 103.78205),
    ("4707", 1.33344648, 103.652700),
    ("4708", 1.29939, 103.7799),
    ("4709", 1.312019, 103.763002),
    ("4710", 1.32153, 103.75273),
    ("4712", 1.341244001, 103.6439134),
    ("4713", 1.347645829, 103.6366955),
    ("4714", 1.31023, 103.76438),
    ("4716", 1.32227, 103.67453),
    ("4798", 1.25999999, 103.823611),
    ("4799", 1.26027777, 103.823888),
    ("5794", 1.3309693, 103.9168616),
    ("5795", 1.326024822, 103.905625),
    ("5797", 1.322875288, 103.8910793),
    ("5798", 1.32036078, 103.877174),
    ("5799", 1.328171608, 103.8685191),
    ("6701", 1.329334, 103.858222),
    ("6703", 1.328899, 103.84121),
    ("6704", 1.32657403, 103.826857),
    ("6705", 1.332124, 103.81768),
    ("6706", 1.349428893, 103.7952799),
    ("6708", 1.345996, 103.69016),
    ("6710", 1.344205, 103.78577),
    ("6711", 1.33771, 103.977827),
    ("6712", 1.332691, 103.770278),
    ("6713", 1.340298, 103.945652),
    ("6714", 1.361742, 103.703341),
    ("6715", 1.356299, 103.716071),
    ("6716", 1.322893, 103.6635051),
    ("7791", 1.354245, 103.963782),
    ("7793", 1.37704704, 103.92946983),
    ("7794", 1.37988658, 103.92009174),
    ("7795", 1.38432741, 103.91585701),
    ("7796", 1.39559294, 103.90515712),
    ("7797", 1.40002575, 103.85702534),
    ("7798", 1.39748842, 103.85400467),
    ("8701", 1.38647, 103.74143),
    ("8702", 1.39059, 103.7717),
    ("8704", 1.3899, 103.74843),
    ("8706", 1.3664, 103.70899),
    ("9701", 1.39466333, 103.83474601),
    ("9702", 1.39474081, 103.81797086),
    ("9703", 1.422857, 103.773005),
    ("9704", 1.42214311, 103.79542062),
    ("9705", 1.42627712, 103.78716637),
    ("9706", 1.41270056, 103.80642712),
]


# ---------------------------------------------------------------------------
# Haversine distance
# ---------------------------------------------------------------------------


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two GPS points."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ---------------------------------------------------------------------------
# Graph structures
# ---------------------------------------------------------------------------


@dataclass
class CameraNode:
    info: CameraInfo

    @property
    def camera_id(self) -> str:
        return self.info.camera_id

    @property
    def road(self) -> str:
        return self.info.road

    @property
    def region(self) -> str:
        return self.info.region

    @property
    def area(self) -> str:
        return self.info.area

    @property
    def lat(self) -> float:
        return self.info.lat

    @property
    def lon(self) -> float:
        return self.info.lon


@dataclass
class CameraEdge:
    cam_a: CameraNode
    cam_b: CameraNode
    distance_km: float
    road: str

    def travel_time_hours(self, speed_kmh: float) -> float:
        return self.distance_km / speed_kmh

    def estimate_speed(self, travel_time_seconds: float) -> float:
        """km/h given observed travel time between cameras in seconds."""
        if travel_time_seconds <= 0:
            return 0.0
        return self.distance_km / (travel_time_seconds / 3600.0)


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------


class CameraNetwork:
    """
    Graph of all 90 LTA cameras linked by expressway adjacency.

    Nodes are ordered along each road so edges represent consecutive
    camera pairs in travel direction.
    """

    # Max gap between consecutive cameras on the same road to form an edge.
    # Cameras further apart than this are likely not directly adjacent
    # (e.g., a different branch of the expressway).
    MAX_EDGE_DISTANCE_KM = 8.0

    def __init__(self):
        self._cm = CameraMap()
        self.nodes: dict[str, CameraNode] = {}
        self.edges: list[CameraEdge] = []
        self._road_nodes: dict[str, list[CameraNode]] = {}
        self._build()

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build(self):
        # Classify all cameras
        for cam_id, lat, lon in _LTA_CAMERAS:
            info = self._cm.lookup(cam_id, lat, lon)
            node = CameraNode(info)
            self.nodes[cam_id] = node
            self._road_nodes.setdefault(info.road, []).append(node)

        # Sort cameras within each road and create edges
        for road, nodes in self._road_nodes.items():
            axis = _ROAD_SORT_AXIS.get(road, "lon")
            nodes.sort(key=lambda n: n.lat if axis == "lat" else n.lon)

            for i in range(len(nodes) - 1):
                a, b = nodes[i], nodes[i + 1]
                dist = haversine_km(a.lat, a.lon, b.lat, b.lon)
                if dist <= self.MAX_EDGE_DISTANCE_KM:
                    self.edges.append(CameraEdge(a, b, round(dist, 3), road))

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def adjacent_pairs(self, road: str | None = None) -> list[CameraEdge]:
        """Return edges, optionally filtered by road."""
        if road:
            return [e for e in self.edges if e.road == road]
        return self.edges

    def neighbors(self, camera_id: str) -> list[tuple[CameraNode, float]]:
        """Return (neighbor_node, distance_km) for all edges touching camera_id."""
        result = []
        for e in self.edges:
            if e.cam_a.camera_id == camera_id:
                result.append((e.cam_b, e.distance_km))
            elif e.cam_b.camera_id == camera_id:
                result.append((e.cam_a, e.distance_km))
        return result

    def roads(self) -> list[str]:
        return sorted(self._road_nodes.keys())

    def cameras_on_road(self, road: str) -> list[CameraNode]:
        return self._road_nodes.get(road, [])

    def summary(self) -> str:
        lines = [
            "Singapore LTA Camera Network",
            f"  Cameras : {len(self.nodes)}",
            f"  Edges   : {len(self.edges)}",
            f"  Roads   : {len(self._road_nodes)}",
            "",
            f"  {'Road':<8} {'Cameras':>7} {'Edges':>6} {'Avg dist km':>12}",
            f"  {'-' * 38}",
        ]
        for road in sorted(self._road_nodes):
            cams = len(self._road_nodes[road])
            edges = self.adjacent_pairs(road)
            avg_d = (sum(e.distance_km for e in edges) / len(edges)) if edges else 0
            lines.append(f"  {road:<8} {cams:>7} {len(edges):>6} {avg_d:>12.2f}")
        return "\n".join(lines)


if __name__ == "__main__":
    net = CameraNetwork()
    print(net.summary())
    print()
    print("CTE corridor (S→N):")
    for e in net.adjacent_pairs("CTE"):
        print(
            f"  {e.cam_a.camera_id} → {e.cam_b.camera_id}  "
            f"{e.distance_km:.2f} km  ({e.cam_a.area} → {e.cam_b.area})"
        )
    print()
    print("PIE corridor (W→E):")
    for e in net.adjacent_pairs("PIE"):
        print(
            f"  {e.cam_a.camera_id} → {e.cam_b.camera_id}  "
            f"{e.distance_km:.2f} km  ({e.cam_a.area} → {e.cam_b.area})"
        )
