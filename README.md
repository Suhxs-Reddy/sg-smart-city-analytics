# Singapore Smart City Traffic Analytics

**CATI — Context-Aware Traffic Intelligence**

A novel traffic detection and analytics platform built on Singapore's 90 LTA traffic cameras. The core contribution is **CATI**, a FiLM-conditioned YOLOv11 detector that adapts to environmental conditions (weather, time-of-day, camera viewpoint) using real-time metadata from Singapore's national APIs.

![Python](https://img.shields.io/badge/python-3.11+-blue)
![PyTorch](https://img.shields.io/badge/pytorch-2.1+-red)
![License](https://img.shields.io/badge/license-MIT-green)
![CI](https://img.shields.io/github/actions/workflow/status/Suhxs-Reddy/sg-smart-city-analytics/ci.yml?label=CI)

---

## The Problem

Generic object detectors (YOLO, Faster R-CNN) treat every frame identically — a clear daytime highway image and a rain-soaked night image from a 320x240 camera receive the exact same feature extraction. But in Singapore's fixed-camera traffic network, we **know things at inference time** that generic detectors ignore:

| Signal | Source | Why It Matters |
|--------|--------|----------------|
| Camera ID | Fixed deployment | Same viewpoint always — learnable spatial priors |
| Weather | data.gov.sg API | Rain/haze degrades features differently than clear sky |
| Time | Timestamp | Lighting, shadows, traffic density vary |
| Resolution | Camera spec | 78 cameras @ 1080p, 11 @ 320x240 |
| PM2.5 | Air quality API | Haze reduces visibility and contrast |

**No published traffic detector uses environmental metadata to modulate the detection backbone.**

## Architecture

CATI injects **Feature-wise Linear Modulation (FiLM)** layers into YOLOv11's backbone. FiLM ([Perez et al., AAAI 2018](https://arxiv.org/abs/1709.07871)) learns channel-wise affine transforms conditioned on an external signal:

```
feature_out = γ ⊙ feature_in + β
```

where `γ` (scale) and `β` (shift) are predicted by a context encoder that processes environmental metadata.

```
CONTEXT BRANCH                    VISION BRANCH

┌────────────────┐               ┌──────────────┐
│ Context Vector │               │ Camera Frame │
│ • weather_id   │               │ (RGB Image)  │
│ • temperature  │               └──────┬───────┘
│ • hour_sin/cos │                      │
│ • camera_id    │               ┌──────▼───────┐
│ • resolution   │               │ YOLO Backbone│
│ • pm25         │               │ P3 → FiLM(γ₁,β₁)
└───────┬────────┘               │ P4 → FiLM(γ₂,β₂)
        │                        │ P5 → FiLM(γ₃,β₃)
 ┌──────▼───────┐               └──────┬───────┘
 │ContextEncoder│                      │
 │ (MLP → γ,β)  │──── FiLM ──────────>│
 └──────────────┘               ┌──────▼───────┐
                                │ Detection    │
                                │ Head (6 cls) │
                                └──────────────┘
```

### Key Design Decisions

- **FiLM init = identity**: γ=1, β=0 at initialization, so the model starts equivalent to vanilla YOLO
- **Per-camera embeddings**: Each of 90 cameras gets a learned 16-dim embedding, capturing viewpoint priors
- **Cyclical time encoding**: sin/cos encoding avoids midnight discontinuity
- **~130K overhead**: CATI adds ~130K parameters to YOLO's 9.4M — 1.4% overhead, negligible inference cost

### Training Strategy

**Phase 1 — Context Module Only** (backbone frozen):
- Train ContextEncoder + FiLM layers only
- YOLO backbone weights from COCO pretrain stay frozen
- LR: 1e-3, 50 epochs

**Phase 2 — End-to-End Fine-tuning**:
- Unfreeze backbone with lower LR (1e-4)
- Context modules at 1e-3
- 30 epochs with cosine annealing

## Project Structure

```
src/
├── models/                    # Novel CATI architecture
│   ├── film.py                # FiLM conditioning layer
│   ├── context_encoder.py     # Environmental metadata encoder
│   └── cati_detector.py       # Full CATI detector + inference pipeline
├── ingestion/                 # Data collection
│   ├── collector.py           # Async Singapore API data collector
│   └── dataset_formatter.py   # Kaggle dataset formatter
├── detection/
│   └── detector.py            # YOLOv11 detection wrapper
├── tracking/
│   └── tracker.py             # BoT-SORT multi-object tracking
├── analytics/
│   ├── predictor.py           # LSTM + GAT + Transformer prediction
│   ├── failure_analyzer.py    # 6-category quality taxonomy
│   ├── drift_monitor.py       # PSI + KS-test data drift
│   └── benchmark.py           # Model comparison suite
├── training/
│   └── train_cati.py          # Two-phase CATI training pipeline
├── api/
│   └── server.py              # FastAPI endpoints
└── pipeline.py                # Pipeline orchestrator
```

## Data Collection

Images are collected from Singapore's [LTA Traffic Images API](https://data.gov.sg/datasets/d_62ff3afe7f0f43ceab65e7431dd4415d/view) every 60 seconds, along with metadata from:

- **Weather**: 24-hour forecast + air temperature
- **Air Quality**: PM2.5 readings by region
- **Taxi GPS**: ~30,000 taxi positions for traffic proxy

```bash
# Quick 6-minute test
python -m src.ingestion.collector --duration 0.1

# 24-hour collection
python -m src.ingestion.collector --duration 24
```

## Development

```bash
# Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run tests
pytest tests/ -v --ignore=tests/test_predictor.py

# Run ML tests (requires torch)
pytest tests/test_models.py tests/test_predictor.py -v

# Lint
ruff check src/ tests/
ruff format src/ tests/
```

## CI/CD

Single clean GitHub Actions workflow:
- **lint-and-test**: Ruff linting + pytest (no torch dependency)
- **test-ml**: PyTorch-dependent model tests with CPU torch

## License

MIT
