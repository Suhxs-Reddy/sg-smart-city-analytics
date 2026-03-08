# 🇸🇬 Singapore Smart City Analytics

Real-time urban intelligence platform processing all 90 of Singapore's LTA public traffic cameras with computer vision, multi-object tracking, and predictive analytics.

## Overview

This system ingests live traffic camera feeds from Singapore's Land Transport Authority (LTA) via the [data.gov.sg](https://data.gov.sg) API and applies a multi-stage ML pipeline to extract actionable urban insights:

1. **Detection** — Fine-tuned YOLOv11s for Singapore traffic conditions (target: 92% mAP)
2. **Tracking** — BoT-SORT with OSNet Re-ID for persistent vehicle identity
3. **Prediction** — Spatial-Temporal Graph Transformer for multi-camera congestion forecasting
4. **Failure Analysis** — 6-category failure taxonomy with per-camera reliability scorecards
5. **Drift Monitoring** — Statistical (PSI, KS-test) model health monitoring
6. **Multi-Modal Fusion** — Traffic × Weather × Taxi × Air Quality correlation

## Architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│                         SINGAPORE SMART CITY PIPELINE                      │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────────────┐       │
│  │   COLLECT     │────▶│   DETECT     │────▶│      TRACK           │       │
│  │ 90 cameras    │     │ YOLOv11s     │     │ BoT-SORT + OSNet    │       │
│  │ + weather     │     │ 6 classes    │     │ Re-ID               │       │
│  │ + taxi GPS    │     │ auto-label   │     │ trajectories        │       │
│  │ + PM2.5       │     │              │     │ congestion score    │       │
│  └──────────────┘     └──────────────┘     └──────────┬───────────┘       │
│                                                        │                   │
│                                                        ▼                   │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────────────┐       │
│  │  DASHBOARD   │◀────│    API       │◀────│     ANALYZE          │       │
│  │ Leaflet map  │     │ FastAPI      │     │ Failure analysis     │       │
│  │ heatmap      │     │ 10 endpoints │     │ Drift detection      │       │
│  │ alerts       │     │ WebSocket    │     │ LSTM + Graph predict │       │
│  └──────────────┘     └──────────────┘     └──────────────────────┘       │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

## Pipeline

Run the full pipeline or individual stages:

```bash
# Full pipeline (detect → track → analyze → label → format)
python -m src.pipeline --mode full --model models/yolo11s_traffic.pt

# Individual stages
python -m src.pipeline --mode detect --input data/raw/2026-03-08
python -m src.pipeline --mode track
python -m src.pipeline --mode analyze

# Data collection (runs as long-running service on Azure/Colab)
python -m src.ingestion.collector --duration 24 --interval 60
```

## Data Sources

| Source | API | Refresh | Status |
|---|---|---|---|
| Traffic Cameras (90) | `data.gov.sg/v1/transport/traffic-images` | 20s | ✅ Verified |
| Taxi Availability (~1,450) | `data.gov.sg/v1/transport/taxi-availability` | 30s | ✅ Verified |
| Air Temperature | `data.gov.sg/v1/environment/air-temperature` | 60s | ✅ Verified |
| Weather Forecast | `data.gov.sg/v1/environment/24-hour-weather-forecast` | 5min | ✅ Verified |
| PM2.5 Air Quality | `data.gov.sg/v1/environment/pm25` | 1hr | ✅ Verified |

## Project Structure

```
sg-smart-city-analytics/
├── src/
│   ├── ingestion/
│   │   ├── collector.py         # Async 90-camera scraper + multi-modal metadata
│   │   └── dataset_formatter.py # Kaggle dataset formatter (stratified splits)
│   ├── detection/
│   │   └── detector.py          # YOLOv11 wrapper + auto-labeling
│   ├── tracking/
│   │   └── tracker.py           # BoT-SORT tracking + congestion scoring
│   ├── analytics/
│   │   ├── predictor.py         # LSTM + Spatial-Temporal Graph Transformer
│   │   ├── failure_analyzer.py  # 6-category failure taxonomy
│   │   ├── drift_monitor.py     # PSI + KS-test drift detection
│   │   └── benchmark.py         # Model comparison suite
│   ├── api/
│   │   └── server.py            # FastAPI REST endpoints
│   └── pipeline.py              # End-to-end pipeline orchestrator
├── configs/
│   ├── collection_config.yaml   # API endpoints + collection params
│   ├── training_config.yaml     # YOLO fine-tuning hyperparams
│   └── traffic_dataset.yaml     # 6-class dataset definition
├── tests/
│   ├── test_collector.py        # Data collector tests (25 tests)
│   ├── test_detector.py         # Detection + label generation tests
│   ├── test_analytics.py        # Failure analysis + drift tests
│   └── test_predictor.py        # Prediction model + feature tests
├── data/
│   ├── raw/                     # Collected camera snapshots
│   └── processed/               # Detection, tracking, analytics outputs
├── models/                      # Trained weights (gitignored)
├── reports/                     # Benchmark reports, fleet reports
└── requirements.txt
```

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| Detection | YOLOv11s (Ultralytics) | 9.4M params, 46.6% COCO mAP, 6ms/frame on T4 |
| Tracking | BoxMOT (BoT-SORT + OSNet x0.25) | Camera motion compensation, motion+appearance fusion |
| Prediction | PyTorch (LSTM → GAT + Transformer) | Per-camera baseline → spatial-temporal graph upgrade |
| Failure Analysis | Custom (6-category taxonomy) | Per-camera reliability scorecards |
| Drift Detection | SciPy (PSI + KS-test) | Statistical, no ML — trustworthy and explainable |
| API | FastAPI | 10 endpoints for cameras, congestion, failures, drift |
| Experiment Tracking | MLflow | Training run comparison |

## Compute Strategy

| Task | Where | Cost |
|---|---|---|
| Development | Local (VS Code) | Free |
| GPU Training | Kaggle (T4, 30hr/week) + Colab (T4) | Free |
| Data Collection | Azure B1s VM | ~$8 from $100 credits |
| Dashboard Hosting | Azure App Service (free tier) | Free |

**Total: $0 out of pocket**

## Getting Started

```bash
# Clone and setup
git clone <repo-url>
cd sg-smart-city-analytics
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Quick data collection test (6 minutes)
python -m src.ingestion.collector --duration 0.1 --interval 60
```

## License

MIT
