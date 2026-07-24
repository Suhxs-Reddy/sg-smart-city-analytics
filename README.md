# 🇸🇬 Singapore Smart City Traffic Analytics

**CATI — Context-Aware Traffic Intelligence**

A novel traffic detection and analytics platform built on Singapore's 90 LTA traffic cameras. CATI is a FiLM-conditioned YOLOv11 detector that adapts to environmental conditions (weather, time-of-day, camera viewpoint) using real-time metadata from Singapore's national APIs.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-HF%20Space-yellow)](https://huggingface.co/spaces/SuhxsReddy/SingaporeAnalytics)
[![Dataset](https://img.shields.io/badge/Dataset-190k%2B%20records-blue)](https://huggingface.co/datasets/SuhxsReddy/cati-singapore-dataset)
[![Python](https://img.shields.io/badge/python-3.11+-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.1+-red)](https://pytorch.org/)
[![CI](https://img.shields.io/github/actions/workflow/status/Suhxs-Reddy/sg-smart-city-analytics/ci.yml?label=CI)](https://github.com/Suhxs-Reddy/sg-smart-city-analytics/actions)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## What This Is

A live analytics system running continuously against Singapore's 90 expressway cameras. Every 90 seconds it:

1. Pulls fresh images from all 90 LTA cameras via [data.gov.sg](https://data.gov.sg)
2. Runs YOLO-based vehicle detection conditioned on real-time weather, PM2.5, and time signals
3. Classifies traffic flow by direction on each road (CTE, PIE, ECP, AYE, TPE, BKE, KJE, SLE, MCE)
4. Pushes a new row to a [public HF dataset](https://huggingface.co/datasets/SuhxsReddy/cati-singapore-dataset)

**After 70+ days of continuous collection: 190,000+ detection records across all 90 cameras.**

---

## The Research Contribution: CATI

Generic object detectors treat every frame identically — a clear daytime highway image and a rain-soaked night image receive the exact same feature extraction. In Singapore's fixed-camera network, we know things at inference time that generic detectors ignore:

| Signal | Source | Why It Matters |
|--------|--------|----------------|
| Camera ID | Fixed deployment | Same viewpoint always — learnable spatial priors |
| Weather | data.gov.sg API | Rain/haze degrades features differently than clear sky |
| Time | Timestamp | Lighting, shadows, traffic density vary |
| Resolution | Camera spec | 78 cameras @ 1080p, 11 @ 320x240 |
| PM2.5 | Air quality API | Haze reduces visibility and contrast |

**No published traffic detector uses environmental metadata to modulate the detection backbone.**

CATI injects **Feature-wise Linear Modulation (FiLM)** layers into YOLOv11's backbone. FiLM ([Perez et al., AAAI 2018](https://arxiv.org/abs/1709.07871)) learns channel-wise affine transforms conditioned on an external signal:

```
feature_out = γ ⊙ feature_in + β
```

where `γ` and `β` are predicted by a context encoder processing real-time environmental metadata.

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

- **FiLM init = identity**: γ=1, β=0 at init, so the model starts equivalent to vanilla YOLO
- **Per-camera embeddings**: Each of 90 cameras gets a learned 16-dim embedding, capturing viewpoint priors
- **Cyclical time encoding**: sin/cos encoding avoids midnight discontinuity
- **~130K overhead**: CATI adds ~130K parameters to YOLO's 9.4M — 1.4% overhead, negligible inference cost

### Training Strategy (Two-Phase)

**Phase 1 — Context Modules Only** (backbone frozen):
- Train ContextEncoder + FiLM layers only using cached P3/P4/P5 features
- LR: 1e-3, 50 epochs

**Phase 2 — End-to-End Fine-tuning**:
- Unfreeze backbone (LR: 1e-4), context modules at 1e-3
- 30 epochs with cosine annealing + AMP + EMA

---

## Live Dashboard

The [HF Space](https://huggingface.co/spaces/SuhxsReddy/SingaporeAnalytics) shows:
- Folium map with all 90 camera locations, colour-coded by congestion level
- Per-road KPI cards (CTE, PIE, ECP, AYE, etc.) with live vehicle counts
- Directional split: vehicles heading each way on each expressway
- Weather overlay from NEA API

---

## Dataset

**[SuhxsReddy/cati-singapore-dataset](https://huggingface.co/datasets/SuhxsReddy/cati-singapore-dataset)**

- 190,000+ rows, collected April–June 2026
- 25 columns: timestamp, camera_id, road, lat/lon, weather, per-class counts (car/motorcycle/bus/truck/van/lorry), directional split, conf/iou/imgsz, model_version
- Updated every 90 seconds while the Space is running
- Annotated detection images for all 90 cameras included

---

## Camera Road Network

`src/network/` contains a ground-truth camera-to-road mapping derived from LTA text labels and OCR:

- `camera_config.json` — authoritative camera metadata (road, lat/lon, direction anchors)
- `camera_network.py` — road assignment with SLE heuristic and MCE override
- `visibility.py` — N-direction visibility analysis (v7: head-on y-anchors + signpost filter)
- `lane_detector.py` — per-lane directional counting via 2-frame IoU tracking

---

## Project Structure

```
app.py                     # Streamlit dashboard (continuous inference + live map)
server.py                  # FastAPI backend
src/
├── models/                # Novel CATI architecture
│   ├── film.py            # FiLM conditioning layer
│   ├── context_encoder.py # Environmental metadata encoder
│   ├── attention.py       # SE-Attention, CBAM, Adaptive Gating
│   └── cati_detector.py   # Full CATI detector + inference pipeline
├── network/               # Singapore camera road network
│   ├── camera_config.json # Ground-truth camera metadata
│   ├── camera_network.py  # Road assignment
│   ├── visibility.py      # Direction visibility analysis
│   └── lane_detector.py   # Per-lane counting
├── training/
│   ├── train_cati.py      # Two-phase training (AMP + EMA + stratified val)
│   └── feature_extractor.py
├── ingestion/
│   ├── collector.py       # Async LTA + weather + PM2.5 collector
│   └── dataset_formatter.py
├── detection/
│   └── detector.py        # YOLOv11 wrapper
├── tracking/
│   └── tracker.py         # BoT-SORT multi-object tracking
├── analytics/
│   ├── predictor.py       # LSTM + GAT congestion forecasting
│   ├── failure_analyzer.py
│   └── drift_monitor.py   # PSI + KS-test model health
└── api/
    └── server.py          # FastAPI endpoints
notebooks/
├── analyse_cameras.ipynb  # Ground-truth camera config derivation
├── train_yolo.ipynb       # Kaggle YOLO training
└── upload_to_hf.ipynb     # Dataset upload utilities
```

---

## Development

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Run tests (no GPU needed)
pytest tests/ -v --ignore=tests/test_predictor.py

# ML tests (requires torch)
pytest tests/test_models.py tests/test_predictor.py -v

# Lint
ruff check src/ tests/ && ruff format src/ tests/
```

## CI/CD

GitHub Actions on every push:
- **lint-and-test**: Ruff + pytest (no torch)
- **test-ml**: PyTorch CPU model tests

## License

MIT
