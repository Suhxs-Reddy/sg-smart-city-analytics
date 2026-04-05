"""
Context Encoder — Environmental Metadata -> Dense Embedding

Encodes real-time environmental signals available from Singapore's data.gov.sg
APIs into a dense context vector that conditions the YOLO backbone via FiLM.

Context signals and their sources:
    - Weather condition: 24-hour forecast API (categorical: clear/rain/fog/overcast/night)
    - Temperature: Air temperature API (continuous, Celsius)
    - PM2.5: Regional air quality API (continuous, ug/m3)
    - Time of day: Collection timestamp (cyclical encoding)
    - Camera identity: Camera ID (learned embedding -- captures viewpoint priors)
    - Resolution bucket: Image dimensions (categorical: 1080p/360p/240p)

Design decisions:
    - Camera embeddings learn per-camera priors (typical vehicle scales, road geometry)
    - Cyclical time encoding (sin/cos) avoids discontinuity at midnight
    - Weather is one-hot encoded, not ordinal -- rain and fog are different, not "more"
    - All features are projected to a shared space then fused with a 2-layer MLP
"""

import math

import torch
import torch.nn as nn

# Weather conditions observed in Singapore's tropical climate
WEATHER_CONDITIONS = [
    "clear",
    "partly_cloudy",
    "cloudy",
    "overcast",
    "light_rain",
    "moderate_rain",
    "heavy_rain",
    "thunderstorm",
    "haze",
    "fog",
    "unknown",
]

# Resolution categories based on Singapore LTA camera specifications
RESOLUTION_BUCKETS = {
    "high": 0,  # 1920x1080 (78 cameras)
    "medium": 1,  # 640x360 (1 camera)
    "low": 2,  # 320x240 (11 cameras)
}

# Total cameras in Singapore's LTA network
NUM_CAMERAS = 90


class ContextEncoder(nn.Module):
    """Encodes environmental metadata into a dense vector for FiLM conditioning.

    Input features:
        weather_id:     int     Index into WEATHER_CONDITIONS
        temperature:    float   Celsius (typically 24-35 for Singapore)
        pm25:           float   ug/m3
        hour:           float   Hour of day [0, 24)
        camera_id:      int     Camera index [0, NUM_CAMERAS)
        resolution_id:  int     Index into RESOLUTION_BUCKETS

    Output:
        context: (B, context_dim) dense embedding vector

    Args:
        num_cameras: Number of unique cameras in the network.
        camera_embed_dim: Dimension of per-camera learned embedding.
        weather_embed_dim: Dimension of weather embedding.
        context_dim: Output dimension of the fused context vector.
    """

    def __init__(
        self,
        num_cameras: int = NUM_CAMERAS,
        camera_embed_dim: int = 16,
        weather_embed_dim: int = 8,
        context_dim: int = 64,
    ):
        super().__init__()

        self.num_cameras = num_cameras
        self.context_dim = context_dim

        # Learned embeddings
        self.camera_embedding = nn.Embedding(num_cameras, camera_embed_dim)
        self.weather_embedding = nn.Embedding(len(WEATHER_CONDITIONS), weather_embed_dim)
        self.resolution_embedding = nn.Embedding(len(RESOLUTION_BUCKETS), 4)

        # Continuous feature normalization
        self.temp_norm = nn.BatchNorm1d(1)
        self.pm25_norm = nn.BatchNorm1d(1)

        # Fusion MLP
        # Input dim: camera_embed + weather_embed + resolution_embed + temp(1) + pm25(1) + time(2)
        raw_dim = camera_embed_dim + weather_embed_dim + 4 + 1 + 1 + 2

        self.fusion = nn.Sequential(
            nn.Linear(raw_dim, context_dim * 2),
            nn.GELU(),
            nn.LayerNorm(context_dim * 2),
            nn.Dropout(0.1),
            nn.Linear(context_dim * 2, context_dim),
            nn.GELU(),
            nn.LayerNorm(context_dim),
        )

    def encode_time(self, hour: torch.Tensor) -> torch.Tensor:
        """Cyclical time encoding to avoid midnight discontinuity.

        Maps hour in [0, 24) to (sin, cos) on the unit circle.

        Args:
            hour: (B,) or (B, 1) float tensor of hours.

        Returns:
            (B, 2) tensor of [sin(2*pi*h/24), cos(2*pi*h/24)]
        """
        hour = hour.float().view(-1)
        angle = 2 * math.pi * hour / 24.0
        return torch.stack([torch.sin(angle), torch.cos(angle)], dim=-1)

    def forward(
        self,
        weather_id: torch.Tensor,
        temperature: torch.Tensor,
        pm25: torch.Tensor,
        hour: torch.Tensor,
        camera_id: torch.Tensor,
        resolution_id: torch.Tensor,
    ) -> torch.Tensor:
        """Encode environmental context into a dense vector.

        All inputs have shape (B,) or (B, 1).

        Returns:
            context: (B, context_dim) fused embedding.
        """
        # Learned embeddings
        cam_emb = self.camera_embedding(camera_id.long())  # (B, camera_embed_dim)
        weather_emb = self.weather_embedding(weather_id.long())  # (B, weather_embed_dim)
        res_emb = self.resolution_embedding(resolution_id.long())  # (B, 4)

        # Continuous features
        temp = self.temp_norm(temperature.float().view(-1, 1))  # (B, 1)
        pm = self.pm25_norm(pm25.float().view(-1, 1))  # (B, 1)
        time_enc = self.encode_time(hour)  # (B, 2)

        # Concatenate all features
        combined = torch.cat([cam_emb, weather_emb, res_emb, temp, pm, time_enc], dim=-1)

        # Fuse into context vector
        context = self.fusion(combined)  # (B, context_dim)

        return context

    @staticmethod
    def weather_to_id(condition: str) -> int:
        """Map a weather condition string to its index."""
        condition = condition.lower().strip()
        for i, w in enumerate(WEATHER_CONDITIONS):
            if w in condition or condition in w:
                return i
        return len(WEATHER_CONDITIONS) - 1  # "unknown"

    @staticmethod
    def resolution_to_id(width: int, height: int) -> int:
        """Map image resolution to a bucket index."""
        pixels = width * height
        if pixels >= 1_000_000:
            return RESOLUTION_BUCKETS["high"]
        elif pixels >= 200_000:
            return RESOLUTION_BUCKETS["medium"]
        else:
            return RESOLUTION_BUCKETS["low"]
