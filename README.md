# 🇸🇬 Singapore Smart City — CATI Traffic Intelligence

**Context-Aware Traffic Intelligence: a FiLM-conditioned YOLOv11 detector that adapts to environment at inference time, deployed live against all 90 Singapore LTA expressway cameras.**

[![Live Demo](https://img.shields.io/badge/Live%20Demo-HF%20Space-yellow)](https://huggingface.co/spaces/SuhxsReddy/SingaporeAnalytics)
[![Dataset](https://img.shields.io/badge/Dataset-190k%2B%20records-blue)](https://huggingface.co/datasets/SuhxsReddy/cati-singapore-dataset)
[![Model](https://img.shields.io/badge/Model-HF%20Hub-orange)](https://huggingface.co/SuhxsReddy/cati-singapore)
[![CI](https://img.shields.io/github/actions/workflow/status/Suhxs-Reddy/sg-smart-city-analytics/ci.yml?label=CI)](https://github.com/Suhxs-Reddy/sg-smart-city-analytics/actions)
[![Python](https://img.shields.io/badge/python-3.11+-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## What's Running

A Streamlit dashboard deployed on Hugging Face Spaces runs continuous inference against Singapore's full expressway network. Every 90 seconds:

1. All 90 LTA camera images are fetched from [data.gov.sg](https://data.gov.sg)
2. CATI runs vehicle detection conditioned on live weather, PM2.5, and time-of-day
3. Per-camera directional counts are computed (vehicles heading each way per road)
4. Results are appended to a public HF dataset and rendered on a live Folium map

**190,910 detection records collected across 70+ days (April – June 2026).** Schema: 25 columns including per-class vehicle counts (car, motorcycle, bus, truck, van, lorry), directional split, weather condition, road assignment, and model versioning metadata.

---

## The Research Problem

Generic object detectors — YOLO, Faster R-CNN, DETR — treat every input frame identically. The same convolutional filters extract features from a clear midday highway shot and a rain-soaked 3 AM image from a degraded 320×240 camera. For a general-purpose detector this is necessary; it has no choice.

**Singapore's LTA camera network is not general-purpose.** It is 90 fixed cameras at known locations, on known roads, in a tropical city with predictable weather patterns. At inference time, the system knows:

| Signal | Source | Information content |
|--------|--------|---------------------|
| Camera ID | Fixed deployment | Viewpoint, road geometry, typical scene composition |
| Weather | NEA API (real-time) | Rain attenuates contrast; haze flattens textures |
| PM2.5 | NEA air quality API | Quantified visibility reduction |
| Time of day | Timestamp | Lighting regime, shadow angles, traffic density priors |
| Resolution | Camera spec | 78 cameras @ 1080p, 11 @ 320×240 |

Standard approach: ignore all of this at inference time and run COCO-pretrained weights unchanged.

**CATI uses all of it.**

---

## Architecture: CATI

CATI injects **Feature-wise Linear Modulation (FiLM)** ([Perez et al., AAAI 2018](https://arxiv.org/abs/1709.07871)) into YOLOv11's backbone at the P3, P4, and P5 feature pyramid levels. FiLM applies a learned channel-wise affine transform:

```
FiLM(feature) = γ ⊙ feature + β
```

`γ` (scale) and `β` (shift) are not fixed — they are predicted fresh at every inference step by a **Context Encoder** that processes the environmental metadata for that specific camera at that specific moment.

```
CONTEXT BRANCH                        VISION BRANCH

┌───────────────────┐                ┌────────────────┐
│   Context Vector  │                │  Camera Frame  │
│                   │                │   (RGB image)  │
│  • weather_id     │                └───────┬────────┘
│  • temperature    │                        │
│  • hour_sin/cos   │                ┌───────▼────────┐
│  • cam_embed[16]  │                │  YOLOv11s      │
│  • resolution     │                │  Backbone      │
│  • pm25           │                │                │
└────────┬──────────┘                │  P3 ──► FiLM₁ │◄─ γ₁,β₁
         │                           │  P4 ──► FiLM₂ │◄─ γ₂,β₂
  ┌──────▼──────┐                    │  P5 ──► FiLM₃ │◄─ γ₃,β₃
  │   Context   │                    └───────┬────────┘
  │   Encoder   │──── (γ₁,β₁,γ₂,β₂,γ₃,β₃)─►│
  │   (MLP)     │                    ┌───────▼────────┐
  └─────────────┘                    │  Detection     │
                                     │  Head (6 cls)  │
                                     └────────────────┘
```

### The Adaptive Gating Problem

Naive FiLM has a failure mode: it conditions unconditionally. On a perfectly clear afternoon when the YOLO backbone is already performing well, forcing a context-modulated feature transform just adds noise. The model needs a way to ask *how much should I trust this context signal right now?*

**Adaptive Gating** solves this. Each FiLM layer learns a scalar gate α ∈ [0, 1] alongside γ and β:

```
output = α · FiLM(feature) + (1 − α) · feature
```

α is itself context-dependent — predicted by the same encoder. In practice:
- **Heavy rain / haze**: α → 1. The backbone features are degraded. Context modulation actively corrects for it.
- **Clear day, high-res camera**: α → 0. The backbone is already doing its job. Don't disturb it.

This means the model learns *when* to apply conditioning, not just *how*. Without this gate, FiLM risks hurting performance on easy conditions while helping on hard ones.

The gate is also paired with **Squeeze-Excitation attention** at each FiLM site — channel recalibration that lets the network further sharpen which features to amplify before the affine transform is applied.

### Parameter Overhead

YOLOv11s has **9.4M parameters**. CATI adds **~130K** — **1.4% overhead**, with negligible inference latency impact.

| Component | Parameters | Notes |
|-----------|-----------|-------|
| Per-camera embeddings | 1,440 | 90 cameras × 16-dim learned vectors |
| Context Encoder (MLP) | ~18K | weather + time + GPS + PM2.5 → 256-dim |
| FiLM Generator × 3 | ~96K | One per pyramid level (P3/P4/P5); outputs γ, β per channel |
| Adaptive Gates × 3 | ~3K | Scalar α per level, conditioned on context |
| SE-Attention × 3 | ~12K | Channel recalibration before each FiLM application |
| **Total CATI overhead** | **~130K** | |

**Initialisation**: γ = 1, β = 0, α = 0 at all FiLM sites. This means CATI starts as an exact copy of vanilla YOLOv11s and only deviates from it as training reveals that context is useful. There is no risk of the conditioning destabilising early training.

### Context Encoder Design

```
Input: [weather_id, temperature, hour_sin, hour_cos, cam_embedding(16), resolution_flag, pm25]
         └──────────────────── ~23 dims ─────────────────────────────────────────────────┘
                                       │
                              Linear(23 → 128) + GELU
                              LayerNorm
                              Linear(128 → 256) + GELU
                                       │
                          ┌────────────┼────────────┐
                     FiLM Generator  Gate      SE-Attn
                     (γ, β per level) (α)     weights
```

**Cyclical time encoding**: hour is encoded as `(sin(2π·h/24), cos(2π·h/24))` — avoids the discontinuity between 23:59 and 00:00 that would confuse a raw hour value.

**Tropical weather conditioning**: weather states include `heavy_rain`, `thundery_showers`, `haze`, `night` — Singapore-specific labels that carry real signal for feature modulation.

---

## Dataset

**[SuhxsReddy/cati-singapore-dataset](https://huggingface.co/datasets/SuhxsReddy/cati-singapore-dataset)**

| Field | Detail |
|-------|--------|
| Records | 190,910 (v2) + 22,018 (v1) = **~213,000 total** |
| Date range | April 15 – June 27, 2026 |
| Cameras | All 90 LTA expressway cameras |
| Collection interval | 90 seconds per full sweep |
| Schema version | v2: ground-truth direction anchors, N-direction visibility |

**25-column schema:**

```
timestamp, camera_id, road, lat, lon, weather,
total_vehicles, dir_a, dir_b, dir_a_label, dir_b_label,
is_ramp, is_junction_camera, n_visible_directions, lane_counts,
car, motorcycle, bus, truck, van, lorry,
conf_threshold, iou_threshold, imgsz, model_version
```

The dataset also contains annotated detection images from the first inference sweep — one JPEG per camera showing bounding boxes and class labels. These serve as ground-truth spot-checks and qualitative evidence of model behaviour on real Singapore expressway footage.

---

## Camera Road Network

Mapping LTA camera IDs to roads, directions, and geographic positions required more than a lookup table. `src/network/` encodes a full directed graph:

- **`camera_config.json`** — authoritative ground truth derived by OCR-reading LTA's own text overlays from actual camera images. Covers all 90 cameras with road name, direction labels (e.g. "towards Changi" / "towards City"), lane anchor coordinates, junction flags.
- **`camera_network.py`** — builds a `networkx` DiGraph where nodes are cameras and directed edges encode road flow direction. Uses PCA on lat/lon to determine the principal road axis (handles E-W roads like PIE and AYE correctly, where naive N-S sorting fails).
- **`visibility.py`** — determines how many traffic directions are visible per camera. v7 adds head-on camera y-anchors and a signpost filter to suppress false direction counts from visible road signs.
- **`lane_detector.py`** — 2-frame IoU tracking to assign individual vehicle detections to directional lanes without full re-ID, keeping it lightweight enough to run every sweep.

---

## Training Strategy

### Why Two Phases?

Training CATI end-to-end from scratch is expensive and risky — the FiLM layers start at identity, so gradients flowing back to the context encoder are initially tiny. The two-phase strategy separates concerns:

**Phase 1 — Context module training only (backbone frozen)**

The YOLO backbone is frozen at its COCO-pretrained weights. P3/P4/P5 feature tensors are pre-extracted and cached to disk (`src/training/feature_extractor.py`). Training only touches the Context Encoder, FiLM generators, and gates.

- Loss: context prediction loss (not trivial MSE — the loss penalises predictions that would push features away from what they'd need to be to improve detection)
- LR: 1e-3, 50 epochs, linear warmup
- Runs on CPU — feature extraction was the GPU-bound step
- Stratified validation: accuracy reported separately for Clear / Rain / Night conditions

**Phase 2 — End-to-end fine-tuning**

The backbone is unfrozen with a deliberately low LR (1e-4) to prevent catastrophic forgetting of COCO pretraining. Context modules continue at 1e-3.

- AMP (Automatic Mixed Precision): halves VRAM usage, speeds training ~1.4×
- EMA (Exponential Moving Average): maintains a shadow copy of weights with τ=0.9999 — EMA weights are used for evaluation, giving more stable mAP measurements
- Cosine annealing over 30 epochs
- Early stopping on stratified val mAP (average across Clear/Rain/Night)

---

## Live Dashboard

**[suhxsreddy-singaporeanalytics.hf.space](https://huggingface.co/spaces/SuhxsReddy/SingaporeAnalytics)**

- Folium map with all 90 cameras, markers colour-coded by road (CTE purple, PIE blue, ECP cyan, AYE green, etc.)
- Per-road KPI panels: total vehicles, directional split (e.g. "towards Changi: 142 / towards City: 89"), per-class breakdown
- Real-time weather from NEA API
- Dataset record counter (live row count from HF)
- Auto-refreshes every 90 seconds in sync with the inference loop

*The Space runs on HF free tier (CPU-basic) and sleeps after inactivity. Visit the URL to wake it.*

---

## Project Structure

```
app.py                          # Streamlit dashboard — inference loop + live map
Dockerfile                      # HF Spaces deployment (CPU torch, ~800MB image)
src/
├── models/                     # CATI architecture
│   ├── film.py                 # FiLM conditioning layer + Adaptive Gate
│   ├── context_encoder.py      # Environmental metadata encoder (MLP)
│   ├── attention.py            # SE-Attention, CBAM, Adaptive Gating
│   └── cati_detector.py        # Full CATI detector — YOLOv11 + FiLM hooks
├── network/                    # Singapore camera road network
│   ├── camera_config.json      # Ground-truth metadata for all 90 cameras
│   ├── camera_network.py       # Directed road graph (NetworkX)
│   ├── visibility.py           # Per-camera direction visibility (v7)
│   └── lane_detector.py        # 2-frame IoU directional lane counting
├── training/
│   ├── train_cati.py           # Two-phase trainer (AMP + EMA + stratified val)
│   └── feature_extractor.py    # Cached P3/P4/P5 features for Phase 1
├── ingestion/
│   ├── collector.py            # Async LTA + NEA weather + PM2.5 collector
│   └── dataset_formatter.py    # Structures raw collections into training sets
├── detection/
│   └── detector.py             # YOLOv11s wrapper (6 Singapore vehicle classes)
├── tracking/
│   └── tracker.py              # ByteTrack multi-object tracking
├── analytics/
│   ├── predictor.py            # LSTM + GAT congestion forecasting
│   ├── failure_analyzer.py     # 6-category camera failure taxonomy
│   └── drift_monitor.py        # PSI + KS-test model health monitoring
└── api/
    └── server.py               # FastAPI backend (10 REST endpoints)
notebooks/                      # Full pipeline in execution order
├── 01_collect_data.ipynb       # Colab: collect images from all 90 cameras
├── 02_analyse_cameras.ipynb    # Derive ground-truth camera config via OCR
├── 03_prepare_dataset.ipynb    # Build YOLO-format training set
├── 04_train_yolo_baseline.ipynb# Kaggle T4: YOLOv11s domain adaptation
├── 05_train_cati_phase1.ipynb  # Phase 1: context modules, frozen backbone
├── 06_train_cati_phase2.ipynb  # Phase 2: end-to-end with neck FiLM
├── 07_evaluate_model.ipynb     # Holdout eval, stratified by condition
├── 08_upload_to_hf.ipynb       # Push weights + dataset to HF Hub
└── 09_demo.ipynb               # End-to-end inference on live LTA frames
```

---

## Current Limitation & Phase 3: Singapore Vehicle Taxonomy

### The Oversight

The current model uses **COCO-pretrained weights with 6 generic classes** (`car, motorcycle, bus, truck, van, lorry`). This produces systematic miscounts because COCO's taxonomy was not designed for Singapore's expressway vehicle mix:

| What's on the road | What COCO calls it | Error |
|---|---|---|
| Container truck (articulated) | `truck` | Road load severely underestimated — 2× footprint ignored |
| Prime mover (no container) | `truck` or `car` | Often missed or misclassified |
| Tipper / construction truck | `truck` | Conflated with container trucks |
| Scooter / moped | `motorcycle` | Different behaviour, different lane discipline |
| Taxi | `car` | High stop-start behaviour invisible to analytics |
| School / shuttle bus | `truck` | Misclassified |

Near MCE, AYE, and Tuas — roads that carry heavy port freight — the container truck misclassification alone means congestion scores are materially wrong.

### Phase 3: Singapore-Specific Vehicle Taxonomy

**10-class taxonomy designed for LTA expressway camera angles:**

```
0  car              sedan, hatchback, SUV, MPV
1  motorcycle       standard motorcycle
2  scooter          moped, small scooter
3  bus              double-decker and single-decker
4  van              panel van, minivan, delivery van
5  lorry            light lorry, pickup truck
6  container_truck  articulated lorry with shipping container
7  prime_mover      tractor unit only (no container)
8  tipper_truck     tipper, dump truck, concrete mixer
9  taxi             Singapore taxi (distinct livery)
```

### Labelling Pipeline: Grounding DINO + Human Review

Rather than manual annotation from scratch, the pipeline uses **Grounding DINO** ([Liu et al., 2023](https://arxiv.org/abs/2303.05499)) — a zero-shot open-set detector — to auto-label the raw images collected from 90 LTA cameras, followed by targeted human review of low-confidence predictions.

```
Google Drive (raw LTA images)
         │
         ▼
  Sample ~200 images per camera condition
  (clear day / rain / night / haze)
         │
         ▼
  Grounding DINO inference
  Text query: "car . motorcycle . scooter . bus . van .
               lorry . container truck . prime mover .
               tipper truck . taxi ."
         │
         ├──► High-confidence detections → YOLO labels (auto-accept)
         └──► Low-confidence / ambiguous → review queue (human label)
                        │
                        ▼
              Label Studio / CVAT human review
                        │
                        ▼
              Clean YOLO-format dataset
              (train/val split, stratified by condition)
                        │
                        ▼
              Fine-tune YOLOv11s on Singapore data
              (backbone init from COCO, new head for 10 classes)
                        │
                        ▼
              CATI Phase 1 + 2 training on Singapore-labelled data
```

Notebook `10_label_sg_vehicles.ipynb` (Colab) runs the full Grounding DINO pass on Drive images and outputs YOLO labels + annotated review JPEGs.

---

## Development

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Core tests (no GPU)
pytest tests/ -v --ignore=tests/test_models.py --ignore=tests/test_predictor.py --ignore=tests/test_training.py

# ML tests (requires torch)
pytest tests/test_models.py tests/test_training.py tests/test_predictor.py -v

# Lint + format
ruff check src/ tests/ app.py
ruff format src/ tests/ app.py
```

## CI

GitHub Actions on every push to `main`:

| Job | What it does |
|-----|-------------|
| `lint` | Ruff lint + format check |
| `test-core` | pytest, no torch — collector, detector, analytics |
| `test-ml` | pytest with CPU torch — CATI model, FiLM layers, training |
| `docker` | Build + push to GHCR (version tags only) |

## License

MIT
