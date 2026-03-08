"""
Singapore Smart City — Data Collector

Automated collection of traffic camera images and multi-modal metadata
(weather, taxi, air quality) from Singapore's data.gov.sg APIs.

Designed to run on Azure B1s VM or Google Colab — NOT locally.

Usage:
    python -m src.ingestion.collector --duration 24 --interval 60
    python -m src.ingestion.collector --duration 0.1 --interval 60  # Quick 6-min test
"""

import asyncio
import hashlib
import json
import logging
import os
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Optional

import aiohttp
import click
import yaml

# Singapore timezone: UTC+8
SGT = timezone(timedelta(hours=8))

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

def load_config(config_path: str = "configs/collection_config.yaml") -> dict:
    """Load collection configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# =============================================================================
# API Clients
# =============================================================================

class SingaporeAPIClient:
    """Async client for all Singapore data.gov.sg APIs."""

    def __init__(self, config: dict, session: aiohttp.ClientSession):
        self.config = config
        self.session = session
        self.timeout = aiohttp.ClientTimeout(
            total=config["collection"]["request_timeout_seconds"]
        )
        self.max_retries = config["collection"]["max_retries"]
        self.retry_delay = config["collection"]["retry_delay_seconds"]

    async def _fetch_json(self, url: str) -> Optional[dict]:
        """Fetch JSON from an API endpoint with retry logic."""
        for attempt in range(self.max_retries):
            try:
                async with self.session.get(url, timeout=self.timeout) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    else:
                        logger.warning(
                            f"API returned {resp.status} for {url} "
                            f"(attempt {attempt + 1}/{self.max_retries})"
                        )
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.warning(
                    f"Request failed for {url}: {e} "
                    f"(attempt {attempt + 1}/{self.max_retries})"
                )

            if attempt < self.max_retries - 1:
                delay = self.retry_delay * (2 ** attempt)  # Exponential backoff
                await asyncio.sleep(delay)

        logger.error(f"All {self.max_retries} retries failed for {url}")
        return None

    async def _download_image(self, url: str) -> Optional[bytes]:
        """Download a camera image with retry logic."""
        for attempt in range(self.max_retries):
            try:
                async with self.session.get(url, timeout=self.timeout) as resp:
                    if resp.status == 200:
                        return await resp.read()
                    else:
                        logger.warning(
                            f"Image download returned {resp.status} for {url}"
                        )
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.warning(f"Image download failed: {e}")

            if attempt < self.max_retries - 1:
                delay = self.retry_delay * (2 ** attempt)
                await asyncio.sleep(delay)

        return None

    async def fetch_traffic_images(self) -> Optional[dict]:
        """Fetch all traffic camera data."""
        url = self.config["apis"]["traffic_images"]["url"]
        return await self._fetch_json(url)

    async def fetch_taxi_availability(self) -> Optional[dict]:
        """Fetch taxi GPS positions."""
        url = self.config["apis"]["taxi_availability"]["url"]
        return await self._fetch_json(url)

    async def fetch_weather(self) -> Optional[dict]:
        """Fetch air temperature readings."""
        url = self.config["apis"]["air_temperature"]["url"]
        return await self._fetch_json(url)

    async def fetch_weather_forecast(self) -> Optional[dict]:
        """Fetch 24-hour weather forecast."""
        url = self.config["apis"]["weather_forecast"]["url"]
        return await self._fetch_json(url)

    async def fetch_pm25(self) -> Optional[dict]:
        """Fetch PM2.5 air quality readings."""
        url = self.config["apis"]["pm25"]["url"]
        return await self._fetch_json(url)


# =============================================================================
# Data Processors
# =============================================================================

def compute_image_hash(image_bytes: bytes) -> str:
    """Compute SHA-256 hash of image bytes for deduplication."""
    return hashlib.sha256(image_bytes).hexdigest()


def extract_weather_condition(forecast_data: Optional[dict]) -> str:
    """Extract current weather condition from forecast API response."""
    if not forecast_data:
        return "unknown"

    try:
        items = forecast_data.get("items", [])
        if items:
            # Get the general forecast
            general = items[0].get("general", {})
            return general.get("forecast", "unknown")
    except (IndexError, KeyError, TypeError):
        pass

    return "unknown"


def extract_temperature(weather_data: Optional[dict]) -> Optional[float]:
    """Extract mean temperature from weather station readings."""
    if not weather_data:
        return None

    try:
        items = weather_data.get("items", [])
        if items:
            readings = items[0].get("readings", [])
            if readings:
                temps = [r["value"] for r in readings if "value" in r]
                return round(sum(temps) / len(temps), 1) if temps else None
    except (IndexError, KeyError, TypeError):
        pass

    return None


def extract_pm25(pm25_data: Optional[dict]) -> Optional[dict]:
    """Extract PM2.5 readings per region."""
    if not pm25_data:
        return None

    try:
        items = pm25_data.get("items", [])
        if items:
            return items[0].get("readings", {}).get("pm25_one_hourly", None)
    except (IndexError, KeyError, TypeError):
        pass

    return None


def count_nearby_taxis(
    taxi_data: Optional[dict],
    camera_lat: float,
    camera_lng: float,
    radius_km: float = 5.0,
) -> int:
    """Count taxis within radius_km of a camera location.

    Uses simple Euclidean approximation — sufficient for Singapore's
    small geographic area (~50km across).
    """
    if not taxi_data:
        return 0

    try:
        features = taxi_data.get("features", [])
        if not features:
            return 0

        coordinates = features[0]["geometry"]["coordinates"]
        count = 0
        # 1 degree ≈ 111km at equator
        radius_deg = radius_km / 111.0

        for lng, lat in coordinates:
            if (abs(lat - camera_lat) < radius_deg and
                    abs(lng - camera_lng) < radius_deg):
                count += 1

        return count
    except (IndexError, KeyError, TypeError):
        return 0


# =============================================================================
# Collector
# =============================================================================

class DataCollector:
    """Main data collection orchestrator."""

    def __init__(self, config: dict):
        self.config = config
        self.output_dir = Path(config["collection"]["output_dir"])
        self.max_concurrent = config["collection"]["max_concurrent_downloads"]
        self.camera_filter = config.get("camera_filter", {})

        # Stats tracking
        self.stats = {
            "cycles_completed": 0,
            "images_saved": 0,
            "images_failed": 0,
            "cameras_responding": 0,
            "cameras_failed": 0,
            "errors": [],
        }

    def _should_collect_camera(self, camera: dict) -> bool:
        """Check if camera passes filter criteria."""
        # Filter by camera ID list
        allowed_ids = self.camera_filter.get("camera_ids")
        if allowed_ids and camera["camera_id"] not in allowed_ids:
            return False

        # Filter by minimum resolution
        min_width = self.camera_filter.get("min_resolution_width")
        if min_width and camera.get("image_metadata", {}).get("width", 0) < min_width:
            return False

        return True

    def _get_image_path(self, camera_id: str, timestamp: datetime) -> Path:
        """Get the filesystem path for saving a camera image."""
        date_str = timestamp.strftime("%Y-%m-%d")
        time_str = timestamp.strftime("%H-%M-%S")
        return self.output_dir / date_str / camera_id / f"{time_str}.jpg"

    def _get_metadata_path(self, camera_id: str, timestamp: datetime) -> Path:
        """Get the path for the camera's daily metadata JSONL file."""
        date_str = timestamp.strftime("%Y-%m-%d")
        return self.output_dir / date_str / camera_id / "metadata.jsonl"

    async def _collect_single_camera(
        self,
        camera: dict,
        client: SingaporeAPIClient,
        weather_condition: str,
        temperature: Optional[float],
        pm25_readings: Optional[dict],
        taxi_data: Optional[dict],
        collection_time: datetime,
    ) -> bool:
        """Download image and save metadata for a single camera."""
        camera_id = camera["camera_id"]
        image_url = camera["image"]
        lat = camera["location"]["latitude"]
        lng = camera["location"]["longitude"]
        width = camera.get("image_metadata", {}).get("width", 0)
        height = camera.get("image_metadata", {}).get("height", 0)

        # Download image
        image_bytes = await client._download_image(image_url)
        if image_bytes is None:
            logger.warning(f"Failed to download image for camera {camera_id}")
            return False

        # Compute hash
        image_hash = compute_image_hash(image_bytes)

        # Save image
        image_path = self._get_image_path(camera_id, collection_time)
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(image_bytes)

        # Count nearby taxis
        taxi_count = count_nearby_taxis(taxi_data, lat, lng, radius_km=5.0)

        # Determine region for PM2.5 (simple heuristic based on longitude)
        pm25_value = None
        if pm25_readings:
            if lng < 103.75:
                pm25_value = pm25_readings.get("west")
            elif lng > 103.9:
                pm25_value = pm25_readings.get("east")
            elif lat > 1.38:
                pm25_value = pm25_readings.get("north")
            elif lat < 1.3:
                pm25_value = pm25_readings.get("south")
            else:
                pm25_value = pm25_readings.get("central")

        # Build metadata record
        metadata = {
            "timestamp": collection_time.isoformat(),
            "camera_id": camera_id,
            "latitude": lat,
            "longitude": lng,
            "image_path": str(image_path),
            "image_hash_sha256": image_hash,
            "image_width": width,
            "image_height": height,
            "image_size_bytes": len(image_bytes),
            "weather_condition": weather_condition,
            "temperature_celsius": temperature,
            "pm25_reading": pm25_value,
            "taxi_count_nearby_5km": taxi_count,
        }

        # Append metadata to JSONL file
        metadata_path = self._get_metadata_path(camera_id, collection_time)
        with open(metadata_path, "a") as f:
            f.write(json.dumps(metadata) + "\n")

        return True

    async def run_collection_cycle(self, client: SingaporeAPIClient) -> dict:
        """Run one complete collection cycle across all cameras and sensors."""
        cycle_start = time.time()
        collection_time = datetime.now(SGT)

        logger.info(
            f"Starting collection cycle at {collection_time.strftime('%H:%M:%S')}"
        )

        # Fetch all data sources in parallel
        traffic_task = client.fetch_traffic_images()
        weather_task = client.fetch_weather()
        forecast_task = client.fetch_weather_forecast()
        pm25_task = client.fetch_pm25()
        taxi_task = client.fetch_taxi_availability()

        traffic_data, weather_data, forecast_data, pm25_data, taxi_data = (
            await asyncio.gather(
                traffic_task, weather_task, forecast_task, pm25_task, taxi_task
            )
        )

        if not traffic_data:
            logger.error("Failed to fetch traffic camera data — skipping cycle")
            return {"success": False, "error": "No traffic data"}

        # Extract multi-modal context
        weather_condition = extract_weather_condition(forecast_data)
        temperature = extract_temperature(weather_data)
        pm25_readings = extract_pm25(pm25_data)

        # Get cameras
        cameras = traffic_data["items"][0]["cameras"]
        filtered_cameras = [c for c in cameras if self._should_collect_camera(c)]

        logger.info(
            f"Collecting {len(filtered_cameras)}/{len(cameras)} cameras | "
            f"Weather: {weather_condition} | "
            f"Temp: {temperature}°C | "
            f"PM2.5: {pm25_readings}"
        )

        # Download images with concurrency limit
        semaphore = asyncio.Semaphore(self.max_concurrent)
        success_count = 0
        fail_count = 0

        async def bounded_collect(camera: dict) -> bool:
            async with semaphore:
                return await self._collect_single_camera(
                    camera, client, weather_condition, temperature,
                    pm25_readings, taxi_data, collection_time,
                )

        results = await asyncio.gather(
            *[bounded_collect(cam) for cam in filtered_cameras],
            return_exceptions=True,
        )

        for result in results:
            if result is True:
                success_count += 1
            else:
                fail_count += 1

        cycle_duration = time.time() - cycle_start

        # Update stats
        self.stats["cycles_completed"] += 1
        self.stats["images_saved"] += success_count
        self.stats["images_failed"] += fail_count
        self.stats["cameras_responding"] = success_count
        self.stats["cameras_failed"] = fail_count

        cycle_result = {
            "success": True,
            "timestamp": collection_time.isoformat(),
            "cameras_success": success_count,
            "cameras_failed": fail_count,
            "cameras_total": len(filtered_cameras),
            "weather": weather_condition,
            "temperature": temperature,
            "duration_seconds": round(cycle_duration, 2),
        }

        logger.info(
            f"Cycle complete: {success_count}/{len(filtered_cameras)} cameras | "
            f"{cycle_duration:.1f}s | "
            f"Total images: {self.stats['images_saved']}"
        )

        return cycle_result

    def print_stats(self):
        """Print collection health dashboard to console."""
        print("\n" + "=" * 60)
        print("  📊 COLLECTION HEALTH DASHBOARD")
        print("=" * 60)
        print(f"  Cycles completed:    {self.stats['cycles_completed']}")
        print(f"  Total images saved:  {self.stats['images_saved']}")
        print(f"  Total images failed: {self.stats['images_failed']}")
        print(f"  Last cycle cameras:  {self.stats['cameras_responding']} ok / "
              f"{self.stats['cameras_failed']} failed")

        if self.stats['images_saved'] + self.stats['images_failed'] > 0:
            success_rate = (
                self.stats['images_saved'] /
                (self.stats['images_saved'] + self.stats['images_failed'])
                * 100
            )
            print(f"  Overall success rate: {success_rate:.1f}%")

        print("=" * 60 + "\n")


# =============================================================================
# Main Entry Point
# =============================================================================

async def run_collector(
    config: dict,
    duration_hours: float,
    interval_seconds: int,
):
    """Run the data collector for a specified duration."""
    collector = DataCollector(config)
    end_time = time.time() + (duration_hours * 3600)

    # Setup logging
    log_config = config.get("logging", {})
    log_level = getattr(logging, log_config.get("level", "INFO"))
    log_file = log_config.get("log_file")

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            *(
                [logging.FileHandler(log_file)]
                if log_file
                else []
            ),
        ],
    )

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"Starting collector — duration: {duration_hours}h, "
        f"interval: {interval_seconds}s"
    )

    async with aiohttp.ClientSession() as session:
        client = SingaporeAPIClient(config, session)

        while time.time() < end_time:
            cycle_start = time.time()

            try:
                result = await collector.run_collection_cycle(client)

                if result.get("success"):
                    collector.print_stats()
                else:
                    logger.error(f"Cycle failed: {result.get('error')}")

            except Exception as e:
                logger.exception(f"Unexpected error during collection cycle: {e}")

            # Wait for next cycle
            elapsed = time.time() - cycle_start
            sleep_time = max(0, interval_seconds - elapsed)

            if time.time() + sleep_time < end_time:
                logger.debug(f"Sleeping {sleep_time:.0f}s until next cycle")
                await asyncio.sleep(sleep_time)
            else:
                break

    logger.info("Collection finished.")
    collector.print_stats()


@click.command()
@click.option(
    "--config",
    default="configs/collection_config.yaml",
    help="Path to collection config YAML",
)
@click.option(
    "--duration",
    default=24.0,
    type=float,
    help="Collection duration in hours (e.g. 0.1 for ~6 min test)",
)
@click.option(
    "--interval",
    default=None,
    type=int,
    help="Override collection interval in seconds (default: from config)",
)
def main(config: str, duration: float, interval: Optional[int]):
    """Singapore Smart City — Data Collector

    Collects traffic camera images and multi-modal metadata from
    Singapore's data.gov.sg APIs.

    Examples:
        # Quick 6-minute test
        python -m src.ingestion.collector --duration 0.1

        # Collect for 24 hours
        python -m src.ingestion.collector --duration 24

        # Collect for 1 week
        python -m src.ingestion.collector --duration 168
    """
    cfg = load_config(config)
    interval_seconds = interval or cfg["collection"]["interval_seconds"]

    print(f"\n🇸🇬 Singapore Smart City — Data Collector")
    print(f"   Duration: {duration} hours")
    print(f"   Interval: {interval_seconds} seconds")
    print(f"   Output:   {cfg['collection']['output_dir']}/\n")

    asyncio.run(run_collector(cfg, duration, interval_seconds))


if __name__ == "__main__":
    main()
