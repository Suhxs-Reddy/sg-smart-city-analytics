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
    - GPS coordinates: Camera latitude/longitude (sinusoidal encoding, optional)

Design decisions:
    - Camera embeddings learn per-camera priors (typical vehicle scales, road geometry)
    - Cyclical time encoding (sin/cos) avoids discontinuity at midnight
    - Weather is one-hot encoded, not ordinal -- rain and fog are different, not "more"
    - GPS encoding uses multi-frequency sinusoids; captures spatial priors from fixed viewpoints
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

# Singapore geographic bounds (for GPS normalization)
_SG_LAT_MID, _SG_LAT_RANGE = 1.35, 0.15
_SG_LON_MID, _SG_LON_RANGE = 103.82, 0.25


class ContextAugmentation(nn.Module):
    """Stochastic context augmentation applied during training.

    Adds small Gaussian noise to continuous inputs (temperature, PM2.5, hour)
    to improve generalization to out-of-distribution environmental conditions.

    Args:
        temp_std: Noise std for temperature (Celsius).
        pm25_std: Noise std for PM2.5 (ug/m3).
        hour_std: Noise std for hour (fractional hours).
    """

    def __init__(
        self,
        temp_std: float = 1.0,
        pm25_std: float = 2.0,
        hour_std: float = 0.25,
    ):
        super().__init__()
        self.temp_std = temp_std
        self.pm25_std = pm25_std
        self.hour_std = hour_std

    def forward(
        self,
        weather: torch.Tensor,
        temp: torch.Tensor,
        pm25: torch.Tensor,
        hour: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply noise augmentation during training.

        Args:
            weather: (B,) integer weather IDs — unchanged.
            temp: (B,) temperature values.
            pm25: (B,) PM2.5 values.
            hour: (B,) hour-of-day values.

        Returns:
            Tuple of (weather, aug_temp, aug_pm25, aug_hour).
        """
        if self.training:
            temp = temp + torch.randn_like(temp) * self.temp_std
            pm25 = (pm25 + torch.randn_like(pm25) * self.pm25_std).clamp(min=0.0)
            hour = (hour + torch.randn_like(hour) * self.hour_std) % 24.0
        return weather, temp, pm25, hour


class ContextEncoder(nn.Module):
    """Encodes environmental metadata into a dense vector for FiLM conditioning.

    Input features:
        weather_id:     int     Index into WEATHER_CONDITIONS
        temperature:    float   Celsius (typically 24-35 for Singapore)
        pm25:           float   ug/m3
        hour:           float   Hour of day [0, 24)
        camera_id:      int     Camera index [0, NUM_CAMERAS)
        resolution_id:  int     Index into RESOLUTION_BUCKETS
        camera_lat:     float   GPS latitude (optional, requires use_gps_encoding=True)
        camera_lon:     float   GPS longitude (optional, requires use_gps_encoding=True)

    Output:
        context: (B, context_dim) dense embedding vector

    Args:
        num_cameras: Number of unique cameras in the network.
        camera_embed_dim: Dimension of per-camera learned embedding.
        weather_embed_dim: Dimension of weather embedding.
        context_dim: Output dimension of the fused context vector.
        use_gps_encoding: If True, include sinusoidal GPS encoding in context.
        gps_embed_dim: Dimension of GPS positional encoding (must be divisible by 4).
        use_augmentation: If True, add stochastic noise to continuous inputs during training.
    """

    def __init__(
        self,
        num_cameras: int = NUM_CAMERAS,
        camera_embed_dim: int = 16,
        weather_embed_dim: int = 8,
        context_dim: int = 64,
        use_gps_encoding: bool = False,
        gps_embed_dim: int = 8,
        use_augmentation: bool = False,
    ):
        super().__init__()

        self.num_cameras = num_cameras
        self.context_dim = context_dim
        self.use_gps_encoding = use_gps_encoding
        self.gps_embed_dim = gps_embed_dim

        # Learned embeddings
        self.camera_embedding = nn.Embedding(num_cameras, camera_embed_dim)
        self.weather_embedding = nn.Embedding(len(WEATHER_CONDITIONS), weather_embed_dim)
        self.resolution_embedding = nn.Embedding(len(RESOLUTION_BUCKETS), 4)

        # Continuous feature normalization — fixed domain-specific scaling.
        # Singapore temperature: ~24-35°C → normalize around 29°C ± 5°C
        # PM2.5: 0-100 μg/m3 typical → scale by 50
        # Using fixed normalization (not BatchNorm) so batch size 1 works during training.

        # Optional augmentation module
        self.augmentation = ContextAugmentation() if use_augmentation else None

        # Fusion MLP input dim:
        #   camera_embed + weather_embed + resolution_embed(4) + temp(1) + pm25(1) + time(2)
        raw_dim = camera_embed_dim + weather_embed_dim + 4 + 1 + 1 + 2
        if use_gps_encoding:
            raw_dim += gps_embed_dim

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

    def encode_gps(self, lat: torch.Tensor, lon: torch.Tensor) -> torch.Tensor:
        """Multi-frequency sinusoidal GPS encoding.

        Normalizes Singapore coordinates to [-1, 1] then applies sinusoidal
        encoding at multiple frequencies to capture spatial priors.

        Args:
            lat: (B,) GPS latitude values.
            lon: (B,) GPS longitude values.

        Returns:
            (B, gps_embed_dim) positional encoding.
        """
        lat = lat.float().view(-1)
        lon = lon.float().view(-1)

        # Normalize to approximate [-1, 1] for Singapore bounds
        lat_n = (lat - _SG_LAT_MID) / _SG_LAT_RANGE
        lon_n = (lon - _SG_LON_MID) / _SG_LON_RANGE

        # Multi-frequency sinusoidal encoding
        num_freqs = self.gps_embed_dim // 4  # each freq → 2 dims per coord
        freqs = torch.arange(num_freqs, device=lat.device, dtype=torch.float32)
        scales = (2.0 ** freqs) * math.pi  # [pi, 2pi, 4pi, ...]

        lat_enc = torch.cat(
            [torch.sin(lat_n.unsqueeze(1) * scales), torch.cos(lat_n.unsqueeze(1) * scales)],
            dim=-1,
        )  # (B, 2*num_freqs)
        lon_enc = torch.cat(
            [torch.sin(lon_n.unsqueeze(1) * scales), torch.cos(lon_n.unsqueeze(1) * scales)],
            dim=-1,
        )  # (B, 2*num_freqs)

        return torch.cat([lat_enc, lon_enc], dim=-1)  # (B, gps_embed_dim)

    def forward(
        self,
        weather_id: torch.Tensor,
        temperature: torch.Tensor,
        pm25: torch.Tensor,
        hour: torch.Tensor,
        camera_id: torch.Tensor,
        resolution_id: torch.Tensor,
        camera_lat: torch.Tensor | None = None,
        camera_lon: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode environmental context into a dense vector.

        All scalar inputs have shape (B,) or (B, 1).

        Args:
            weather_id: Integer weather condition index.
            temperature: Temperature in Celsius.
            pm25: PM2.5 concentration in ug/m3.
            hour: Hour of day [0, 24).
            camera_id: Camera index.
            resolution_id: Resolution bucket index.
            camera_lat: Optional GPS latitude (required if use_gps_encoding=True).
            camera_lon: Optional GPS longitude (required if use_gps_encoding=True).

        Returns:
            context: (B, context_dim) fused embedding.
        """
        # Optional augmentation during training
        if self.augmentation is not None:
            weather_id, temperature, pm25, hour = self.augmentation(
                weather_id, temperature, pm25, hour
            )

        # Learned embeddings
        cam_emb = self.camera_embedding(camera_id.long())  # (B, camera_embed_dim)
        weather_emb = self.weather_embedding(weather_id.long())  # (B, weather_embed_dim)
        res_emb = self.resolution_embedding(resolution_id.long())  # (B, 4)

        # Continuous features — fixed normalization using Singapore domain priors
        temp = (temperature.float().view(-1, 1) - 29.0) / 5.0  # (B, 1)
        pm = (pm25.float().view(-1, 1) / 50.0).clamp(-3.0, 3.0)  # (B, 1)
        time_enc = self.encode_time(hour)  # (B, 2)

        parts = [cam_emb, weather_emb, res_emb, temp, pm, time_enc]

        # Optional GPS encoding
        if self.use_gps_encoding:
            if camera_lat is None or camera_lon is None:
                # Zero-pad GPS when coordinates are unavailable
                B = cam_emb.shape[0]
                gps_enc = torch.zeros(B, self.gps_embed_dim, device=cam_emb.device)
            else:
                gps_enc = self.encode_gps(camera_lat, camera_lon)
            parts.append(gps_enc)

        combined = torch.cat(parts, dim=-1)
        return self.fusion(combined)

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
