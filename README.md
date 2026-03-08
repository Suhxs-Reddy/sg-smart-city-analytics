# 🇸🇬 Singapore Smart City Analytics

Real-time urban intelligence platform processing all 90 of Singapore's LTA public traffic cameras with computer vision, multi-object tracking, and predictive analytics.

## Overview

This system ingests live traffic camera feeds from Singapore's Land Transport Authority (LTA) via the [data.gov.sg](https://data.gov.sg) API and applies a multi-stage ML pipeline to extract actionable urban insights:

1. **Detection** — Fine-tuned YOLOv11 for Singapore traffic conditions
2. **Tracking** — BoT-SORT with Re-ID for persistent vehicle/pedestrian tracking
3. **Analytics** — Congestion prediction, anomaly detection, speed estimation
4. **Multi-Modal Correlation** — Cross-referencing traffic with weather, taxi demand, and air quality
5. **Visualization** — Geographic heatmap dashboard with island-wide coverage

## Architecture

```
┌─────────────────┐    ┌──────────────┐    ┌──────────────────┐
│  Data Ingestion │───▶│  Detection   │───▶│    Tracking       │
│  (90 LTA Cams)  │    │ (YOLOv11 INT8│    │  (BoT-SORT+ReID) │
│  + Weather API  │    │  TensorRT)   │    │                  │
│  + Taxi API     │    └──────────────┘    └────────┬─────────┘
└─────────────────┘                                 │
                                                    ▼
┌─────────────────┐    ┌──────────────┐    ┌──────────────────┐
│   Dashboard     │◀───│     API      │◀───│   Analytics      │
│ (React+Leaflet) │    │  (FastAPI)   │    │ (LSTM+Autoencoder│
│  Geographic Map │    │  WebSocket   │    │  Anomaly Detect) │
└─────────────────┘    └──────────────┘    └──────────────────┘
```

## Data Sources

| Source | API | Refresh Rate |
|---|---|---|
| Traffic Cameras | `data.gov.sg/v1/transport/traffic-images` | 20 seconds |
| Taxi Availability | `data.gov.sg/v1/transport/taxi-availability` | 30 seconds |
| Air Temperature | `data.gov.sg/v1/environment/air-temperature` | 1 minute |
| Weather Forecast | `data.gov.sg/v1/environment/24-hour-weather-forecast` | Periodic |
| PM2.5 Air Quality | `data.gov.sg/v1/environment/pm25` | 1 hour |

## Project Structure

```
sg-smart-city-analytics/
├── src/
│   ├── ingestion/       # API clients for LTA cameras, weather, taxi data
│   ├── detection/       # YOLOv11 inference, fine-tuning scripts
│   ├── tracking/        # BoT-SORT/StrongSORT integration via BoxMOT
│   ├── analytics/       # Congestion prediction, anomaly detection, speed estimation
│   └── api/             # FastAPI server, WebSocket real-time feed
├── dashboard/           # React + Leaflet geographic heatmap UI
├── data/
│   ├── raw/             # Collected camera snapshots
│   └── processed/       # Aggregated analytics data
├── models/              # Trained model weights (gitignored)
├── configs/             # YAML configs for cameras, model params, API keys
├── tests/               # Unit and integration tests
├── scripts/             # Training, evaluation, deployment scripts
├── docs/                # Architecture docs, benchmarks, experiment logs
├── docker-compose.yml   # Service orchestration
└── requirements.txt     # Python dependencies
```

## Tech Stack

- **Detection**: YOLOv11 (Ultralytics) with TensorRT INT8 quantization
- **Tracking**: BoxMOT (BoT-SORT + OSNet Re-ID)
- **Prediction**: PyTorch (LSTM / Transformer for congestion forecasting)
- **Anomaly Detection**: Variational Autoencoder (VAE)
- **API**: FastAPI + WebSocket
- **Dashboard**: React + Leaflet.js
- **MLOps**: MLflow for experiment tracking
- **Infrastructure**: Docker, cloud GPU (Vast.ai / RunPod)

## Getting Started

```bash
# Clone the repo
git clone <repo-url>
cd sg-smart-city-analytics

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Test the API connection
python -m src.ingestion.test_connection
```

## License

MIT
