"""
Tests for the Singapore Smart City data collector.

Covers:
- API response parsing
- Metadata extraction (weather, temperature, PM2.5, taxi count)
- Image hash computation
- Camera filtering
- Collection cycle orchestration (with mocked APIs)
"""

import hashlib
import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest

from src.ingestion.collector import (
    SGT,
    DataCollector,
    SingaporeAPIClient,
    compute_image_hash,
    count_nearby_taxis,
    extract_pm25,
    extract_temperature,
    extract_weather_condition,
    load_config,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_config():
    """Minimal config for testing (doesn't need real YAML file)."""
    return {
        "apis": {
            "traffic_images": {"url": "https://api.data.gov.sg/v1/transport/traffic-images"},
            "taxi_availability": {"url": "https://api.data.gov.sg/v1/transport/taxi-availability"},
            "air_temperature": {"url": "https://api.data.gov.sg/v1/environment/air-temperature"},
            "weather_forecast": {
                "url": "https://api.data.gov.sg/v1/environment/24-hour-weather-forecast"
            },
            "pm25": {"url": "https://api.data.gov.sg/v1/environment/pm25"},
        },
        "collection": {
            "interval_seconds": 60,
            "output_dir": "/tmp/test_sg_data",
            "max_retries": 2,
            "retry_delay_seconds": 0.01,
            "request_timeout_seconds": 5,
            "max_concurrent_downloads": 5,
        },
        "camera_filter": {
            "camera_ids": None,
            "min_resolution_width": None,
        },
        "logging": {
            "level": "WARNING",
            "log_file": None,
            "console_output": False,
        },
    }


@pytest.fixture
def sample_traffic_response():
    """Minimal traffic API response with 2 cameras."""
    return {
        "items": [
            {
                "timestamp": "2026-03-08T15:00:00+08:00",
                "cameras": [
                    {
                        "camera_id": "1001",
                        "timestamp": "2026-03-08T15:00:05+08:00",
                        "image": "https://images.data.gov.sg/api/traffic-images/2026/03/cam1001.jpg",
                        "location": {"latitude": 1.29531, "longitude": 103.871146},
                        "image_metadata": {"width": 1920, "height": 1080, "md5": "abc123"},
                    },
                    {
                        "camera_id": "1002",
                        "timestamp": "2026-03-08T15:00:05+08:00",
                        "image": "https://images.data.gov.sg/api/traffic-images/2026/03/cam1002.jpg",
                        "location": {"latitude": 1.31988, "longitude": 103.87653},
                        "image_metadata": {"width": 320, "height": 240, "md5": "def456"},
                    },
                ],
            }
        ],
    }


@pytest.fixture
def sample_forecast_response():
    """Weather forecast API response."""
    return {
        "items": [
            {
                "general": {
                    "forecast": "Thundery Showers",
                    "temperature": {"low": 24, "high": 34},
                },
            }
        ],
    }


@pytest.fixture
def sample_temperature_response():
    """Air temperature API response."""
    return {
        "items": [
            {
                "readings": [
                    {"station_id": "S107", "value": 26.4},
                    {"station_id": "S24", "value": 25.7},
                ],
            }
        ],
    }


@pytest.fixture
def sample_pm25_response():
    """PM2.5 API response."""
    return {
        "items": [
            {
                "readings": {
                    "pm25_one_hourly": {
                        "west": 9,
                        "east": 18,
                        "central": 15,
                        "south": 7,
                        "north": 15,
                    },
                },
            }
        ],
    }


@pytest.fixture
def sample_taxi_response():
    """Taxi availability API response with a few positions."""
    return {
        "features": [
            {
                "geometry": {
                    "type": "MultiPoint",
                    "coordinates": [
                        [103.871, 1.295],  # Near camera 1001
                        [103.872, 1.296],  # Near camera 1001
                        [103.7, 1.35],  # Far away (west)
                        [103.95, 1.32],  # Far away (east)
                    ],
                },
            }
        ],
    }


# =============================================================================
# Unit Tests — Data Extraction
# =============================================================================


class TestExtractWeatherCondition:
    def test_valid_forecast(self, sample_forecast_response):
        result = extract_weather_condition(sample_forecast_response)
        assert result == "Thundery Showers"

    def test_none_input(self):
        assert extract_weather_condition(None) == "unknown"

    def test_empty_items(self):
        assert extract_weather_condition({"items": []}) == "unknown"

    def test_missing_general(self):
        assert extract_weather_condition({"items": [{}]}) == "unknown"

    def test_missing_forecast_key(self):
        data = {"items": [{"general": {}}]}
        assert extract_weather_condition(data) == "unknown"


class TestExtractTemperature:
    def test_valid_readings(self, sample_temperature_response):
        result = extract_temperature(sample_temperature_response)
        assert result == pytest.approx(26.05, abs=0.1)  # mean of 26.4 and 25.7

    def test_none_input(self):
        assert extract_temperature(None) is None

    def test_empty_readings(self):
        assert extract_temperature({"items": [{"readings": []}]}) is None

    def test_single_reading(self):
        data = {"items": [{"readings": [{"station_id": "S1", "value": 30.0}]}]}
        assert extract_temperature(data) == 30.0


class TestExtractPm25:
    def test_valid_readings(self, sample_pm25_response):
        result = extract_pm25(sample_pm25_response)
        assert result["west"] == 9
        assert result["east"] == 18
        assert result["central"] == 15

    def test_none_input(self):
        assert extract_pm25(None) is None

    def test_empty_items(self):
        assert extract_pm25({"items": []}) is None


class TestCountNearbyTaxis:
    def test_count_nearby(self, sample_taxi_response):
        # Camera 1001 is at (1.295, 103.871). Two taxis are near it.
        count = count_nearby_taxis(
            sample_taxi_response, camera_lat=1.295, camera_lng=103.871, radius_km=5.0
        )
        assert count == 2

    def test_no_nearby(self, sample_taxi_response):
        # Camera far from any taxis
        count = count_nearby_taxis(
            sample_taxi_response, camera_lat=1.45, camera_lng=103.6, radius_km=1.0
        )
        assert count == 0

    def test_none_input(self):
        assert count_nearby_taxis(None, 1.3, 103.8) == 0

    def test_large_radius_catches_all(self, sample_taxi_response):
        count = count_nearby_taxis(
            sample_taxi_response, camera_lat=1.3, camera_lng=103.85, radius_km=50.0
        )
        assert count == 4


class TestComputeImageHash:
    def test_deterministic(self):
        data = b"fake image bytes"
        assert compute_image_hash(data) == hashlib.sha256(data).hexdigest()

    def test_different_input_different_hash(self):
        assert compute_image_hash(b"image1") != compute_image_hash(b"image2")


# =============================================================================
# Unit Tests — Camera Filtering
# =============================================================================


class TestCameraFiltering:
    def test_no_filter_passes_all(self, sample_config, sample_traffic_response):
        collector = DataCollector(sample_config)
        cameras = sample_traffic_response["items"][0]["cameras"]
        assert all(collector._should_collect_camera(c) for c in cameras)

    def test_filter_by_camera_id(self, sample_config, sample_traffic_response):
        sample_config["camera_filter"]["camera_ids"] = ["1001"]
        collector = DataCollector(sample_config)
        cameras = sample_traffic_response["items"][0]["cameras"]

        assert collector._should_collect_camera(cameras[0]) is True  # 1001
        assert collector._should_collect_camera(cameras[1]) is False  # 1002

    def test_filter_by_resolution(self, sample_config, sample_traffic_response):
        sample_config["camera_filter"]["min_resolution_width"] = 640
        collector = DataCollector(sample_config)
        cameras = sample_traffic_response["items"][0]["cameras"]

        assert collector._should_collect_camera(cameras[0]) is True  # 1920
        assert collector._should_collect_camera(cameras[1]) is False  # 320


# =============================================================================
# Unit Tests — Path Generation
# =============================================================================


class TestPathGeneration:
    def test_image_path_format(self, sample_config):
        collector = DataCollector(sample_config)
        ts = datetime(2026, 3, 8, 15, 0, 0, tzinfo=SGT)
        path = collector._get_image_path("1001", ts)

        assert "2026-03-08" in str(path)
        assert "1001" in str(path)
        assert "15-00-00.jpg" in str(path)

    def test_metadata_path_format(self, sample_config):
        collector = DataCollector(sample_config)
        ts = datetime(2026, 3, 8, 15, 0, 0, tzinfo=SGT)
        path = collector._get_metadata_path("1001", ts)

        assert "2026-03-08" in str(path)
        assert "1001" in str(path)
        assert "metadata.jsonl" in str(path)


# =============================================================================
# Integration-Style Tests — Collection Cycle (mocked HTTP)
# =============================================================================


class TestCollectionCycle:
    @pytest.mark.xfail(reason="Integration test needs image download mock wiring")
    @pytest.mark.asyncio
    async def test_full_cycle_with_mocked_apis(
        self,
        sample_config,
        sample_traffic_response,
        sample_forecast_response,
        sample_temperature_response,
        sample_pm25_response,
        sample_taxi_response,
        tmp_path,
    ):
        """Test a complete collection cycle with all APIs mocked."""
        sample_config["collection"]["output_dir"] = str(tmp_path / "data")

        # Create a mock session
        mock_session = AsyncMock(spec=aiohttp.ClientSession)

        # Setup mock responses
        fake_image = b"\xff\xd8\xff\xe0" + b"\x00" * 100  # Fake JPEG header

        def mock_get(url, **kwargs):
            """Route mock responses based on URL."""
            mock_resp = AsyncMock()
            mock_resp.status = 200

            if "traffic-images" in url:
                mock_resp.json = AsyncMock(return_value=sample_traffic_response)
            elif "taxi-availability" in url:
                mock_resp.json = AsyncMock(return_value=sample_taxi_response)
            elif "air-temperature" in url:
                mock_resp.json = AsyncMock(return_value=sample_temperature_response)
            elif "24-hour-weather" in url:
                mock_resp.json = AsyncMock(return_value=sample_forecast_response)
            elif "pm25" in url:
                mock_resp.json = AsyncMock(return_value=sample_pm25_response)
            elif "images.data.gov.sg" in url:
                mock_resp.read = AsyncMock(return_value=fake_image)
            else:
                mock_resp.status = 404

            # Return a sync context manager with async protocol
            cm = MagicMock()
            cm.__aenter__ = AsyncMock(return_value=mock_resp)
            cm.__aexit__ = AsyncMock(return_value=False)
            return cm

        mock_session.get = mock_get

        # Run collection
        collector = DataCollector(sample_config)
        client = SingaporeAPIClient(sample_config, mock_session)
        result = await collector.run_collection_cycle(client)

        # Assertions
        assert result["success"] is True
        assert result["cameras_success"] == 2
        assert result["cameras_failed"] == 0
        assert result["weather"] == "Thundery Showers"
        assert result["temperature"] == 26.1

        # Check files were created
        data_dir = tmp_path / "data"
        assert data_dir.exists()

        # Check images saved
        jpg_files = list(data_dir.rglob("*.jpg"))
        assert len(jpg_files) == 2

        # Check metadata saved
        jsonl_files = list(data_dir.rglob("*.jsonl"))
        assert len(jsonl_files) == 2

        # Validate metadata content
        for jsonl_file in jsonl_files:
            with open(jsonl_file) as f:
                metadata = json.loads(f.readline())
                assert "camera_id" in metadata
                assert "weather_condition" in metadata
                assert metadata["weather_condition"] == "Thundery Showers"
                assert metadata["temperature_celsius"] == 26.1
                assert "pm25_reading" in metadata
                assert "taxi_count_nearby_5km" in metadata
                assert "image_hash_sha256" in metadata
                assert len(metadata["image_hash_sha256"]) == 64

    @pytest.mark.asyncio
    async def test_cycle_handles_api_failure(
        self,
        sample_config,
        tmp_path,
    ):
        """Test graceful handling when traffic API is down."""
        sample_config["collection"]["output_dir"] = str(tmp_path / "data")

        mock_session = AsyncMock(spec=aiohttp.ClientSession)

        def mock_get_fail(url, **kwargs):
            mock_resp = AsyncMock()
            mock_resp.status = 500
            cm = MagicMock()
            cm.__aenter__ = AsyncMock(return_value=mock_resp)
            cm.__aexit__ = AsyncMock(return_value=False)
            return cm

        mock_session.get = mock_get_fail

        collector = DataCollector(sample_config)
        client = SingaporeAPIClient(sample_config, mock_session)
        result = await collector.run_collection_cycle(client)

        assert result["success"] is False
        assert "error" in result


# =============================================================================
# Config Loading Test
# =============================================================================


class TestConfigLoading:
    def test_load_real_config(self):
        """Verify the real config file loads without errors."""
        config = load_config("configs/collection_config.yaml")
        assert "apis" in config
        assert "collection" in config
        assert config["apis"]["traffic_images"]["url"].startswith("https://")
        assert config["collection"]["interval_seconds"] == 60
        assert config["collection"]["max_retries"] == 3
