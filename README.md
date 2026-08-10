# 🇸🇬 Singapore Smart City — CATI Traffic Intelligence

**Context-Aware Traffic Intelligence: a FiLM-conditioned YOLOv11 detector that adapts to environment at inference time, deployed live against Singapore's LTA camera network.**

[![Live Demo](https://img.shields.io/badge/Live%20Demo-HF%20Space-yellow)](https://huggingface.co/spaces/SuhxsReddy/SingaporeAnalytics)
[![Dataset](https://img.shields.io/badge/Dataset-213k%2B%20records-blue)](https://huggingface.co/datasets/SuhxsReddy/cati-singapore-dataset)
[![Model](https://img.shields.io/badge/Model-HF%20Hub-orange)](https://huggingface.co/SuhxsReddy/cati-singapore)
[![CI](https://img.shields.io/github/actions/workflow/status/Suhxs-Reddy/sg-smart-city-analytics/ci.yml?label=CI)](https://github.com/Suhxs-Reddy/sg-smart-city-analytics/actions)
[![Python](https://img.shields.io/badge/python-3.11+-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## What's Running

A Streamlit dashboard deployed on Hugging Face Spaces runs continuous inference against Singapore's LTA camera network. Every 90 seconds:

1. Live LTA camera images are fetched from [data.gov.sg](https://data.gov.sg)
2. CATI runs vehicle detection conditioned on live weather, PM2.5, and time-of-day
3. Per-camera directional counts are computed (vehicles heading each way per road)
4. Results are appended to a public HF dataset and rendered on a live Folium map

**Two distinct data phases:**

| Phase | Cameras | Period | Coverage | Records |
|-------|---------|--------|----------|---------|
| Historical | 90 LTA expressway cameras (320×240) | April 2026 — 15-day run | Full Singapore road network | 213K+ |
| Current live | 8 LTA checkpoint cameras (1920×1080) | July 2026 → ongoing | Woodlands, Tuas, Sentosa Gateway | Growing |

From 30 June 2026, LTA decommissioned the 320×240 expressway camera feed and retained only 8 high-resolution cameras at Singapore's two land border crossings (Woodlands Checkpoint and Tuas Second Link) and Sentosa Gateway. These are three of the highest-traffic chokepoints in Singapore — Woodlands and Tuas handle over 500,000 daily crossings between Singapore and Malaysia. The live feed now delivers deep checkpoint analytics: cross-border vehicle flow, heavy goods vehicle classification at Tuas, and tourist traffic patterns at Sentosa.

**Phase 3 in progress.** Initial retraining on a 10-class Singapore-specific taxonomy (car, motorcycle, scooter, bus, van, lorry, container truck, prime mover, tipper truck, taxi) complete — CATI mAP50=0.572 vs ablation 0.541; in pre-dawn frames (05:00–06:00) CATI produced ~7.5% more detections than the ablation at higher average confidence (0.634 vs 0.602) — consistent with the gate opening in low light; ground-truthed night accuracy in progress. Now collecting adversarial data at the 8 checkpoint cameras (night/sunrise/rain) for targeted fine-tuning on border traffic. New weights pushed to HF and app.py updated to 10-class schema once fine-tuning is done.

---

## The Research Problem

Generic object detectors — YOLO, Faster R-CNN, DETR — treat every input frame identically. The same convolutional filters extract features from a clear midday highway shot and a rain-soaked 3 AM image from a degraded 320×240 camera. For a general-purpose detector this is necessary; it has no choice.

**Singapore's LTA camera network is not general-purpose.** These are fixed cameras at known locations, on known roads, in a tropical city with predictable weather patterns. At inference time, the system knows:

| Signal | Source | Information content |
|--------|--------|---------------------|
| Camera ID | Fixed deployment | Viewpoint, road geometry, typical scene composition |
| Weather | NEA API (real-time) | Rain attenuates contrast; haze flattens textures |
| PM2.5 | NEA air quality API | Quantified visibility reduction |
| Time of day | Timestamp | Lighting regime, shadow angles, traffic density priors |
| Resolution | Camera spec | All current cameras @ 1920×1080 |

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
                                     │  Head (10 cls) │
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
| Per-camera embeddings | 128 | 8 cameras × 16-dim learned vectors (expandable) |
| Context Encoder (MLP) | ~18K | weather + time + GPS + PM2.5 → 256-dim |
| FiLM Generator × 3 | ~96K | One per pyramid level (P3/P4/P5); outputs γ, β per channel |
| Adaptive Gates × 3 | ~3K | Scalar α per level, conditioned on context |
| SE-Attention × 3 | ~12K | Channel recalibration before each FiLM application |
| **Total CATI overhead** | **~130K** | |

**Initialisation**: γ = 1, β = 0, gate bias = −2 (α ≈ 0.12, near-off) at all FiLM sites. CATI starts near-identical to vanilla YOLOv11s — the gate is biased strongly toward the identity path and only opens where training demonstrates that context reduces detection loss. There is no risk of the conditioning destabilising early training.

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
| Records | **213,000+** (growing — live collection every 90s) |
| Historical dataset | April 2026, 15-day run — 90 LTA expressway cameras, full city coverage |
| Current live feed | July 2026 → ongoing — 8 checkpoint cameras (Woodlands × 3, Tuas × 3, Sentosa × 2) |
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

- **`camera_config.json`** — authoritative ground truth derived by OCR-reading LTA's own text overlays from actual camera images. Covers all cameras with road name, direction labels (e.g. "towards JB" / "towards City"), lane anchor coordinates, junction flags.
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

- Folium map with all active cameras, markers colour-coded by checkpoint location
- Per-camera KPI panels: total vehicles, directional split (e.g. "towards JB: 142 / towards SG: 89"), per-class breakdown
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

## Phase 3: Singapore Vehicle Taxonomy

### The Problem (Phase 2)

The Phase 2 model used **COCO-pretrained weights with 6 generic classes** (`car, motorcycle, bus, truck, van, lorry`). This produced systematic miscounts because COCO's taxonomy was not designed for Singapore's vehicle mix:

| What's on the road | What COCO called it | Error |
|---|---|---|
| Container truck (articulated) | `truck` | Road load severely underestimated — 2× footprint ignored |
| Prime mover (no container) | `truck` or `car` | Often missed or misclassified |
| Tipper / construction truck | `truck` | Conflated with container trucks |
| Scooter / moped | `motorcycle` | Different behaviour, different lane discipline |
| Taxi | `car` | High stop-start behaviour invisible to analytics |

At Tuas Checkpoint — Singapore's primary heavy freight border crossing — the container truck misclassification alone meant congestion and load scores were materially wrong.

### Phase 3: Singapore-Specific Vehicle Taxonomy (Complete)

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

Notebook `10_label_sg_vehicles.ipynb` (Colab) runs the full Grounding DINO pass on Drive images and outputs YOLO labels + annotated review JPEGs. **GDino labelling complete** — ~1,800 images. Initial Phase 3 training done: CATI mAP50=0.572, ablation mAP50=0.541, true CATI contribution +0.031 on clean val; in pre-dawn frames (05:00–06:00 SGT) CATI produced ~7.5% more detections than the ablation at higher average confidence (0.634 vs 0.602) — consistent with the gate opening in low light; ground-truthed accuracy on a stratified night val set is in progress. Adversarial data collection at checkpoint cameras ongoing — fine-tuning and production deployment pending.

---

## Phase 4: Night-Aware Detection (Planned)

### The FiLM Limitation at Night

FiLM modulates existing feature map activations — it scales and shifts what the backbone already computed. This works well when degraded conditions (rain, haze, glare) suppress signal that is still structurally present. At night it hits a harder wall: the COCO-pretrained backbone's convolutional filters learned to respond to edges, textures, and colour gradients. In total darkness those signals collapse. FiLM applied to near-zero activations cannot recover information that was never extracted.

The relevant visual cues at night are structurally different: headlight pair spacing identifies vehicle width, headlight height identifies truck vs car, tail light density indicates queue depth. These are point-light-source features, not texture features.

### Planned Architecture

**1. Dark baseline hard negatives**

Run the collector 3–5am SGT when Woodlands and Tuas are at minimum traffic. These frames have the full streetlight configuration visible with few or no vehicles. Training on these as empty-label negatives teaches the model that lit scene geometry alone — streetlights, road markings, ambient glow — does not indicate a vehicle. Without this, a night-trained model risks firing on streetlight patterns.

**2. Context-conditioned bright-region attention**

Insert a lightweight attention module before the backbone that activates when the context encoder detects dark conditions (hour encoding + mean pixel intensity). In dark frames, this generates a saliency mask over high-intensity point sources (headlight pairs, tail lights) and upweights those regions in the early backbone feature maps before FiLM applies. During daylight the module gates off entirely. This is distinct from Retinex-style illumination normalisation — Retinex would suppress streetlights along with ambient light, destroying the very point-light signals that carry vehicle information at night.

**3. Frequency-domain reweighting**

At night, low-frequency components (large bright blobs) dominate; high-frequency texture features collapse. A frequency reweighting layer conditioned on the context encoder upweights low-frequency channels in dark frames, directing the backbone's attention toward headlight blob features rather than absent texture edges.

**Expected outcome**: the +0.031 mAP on clean val and the ~7.5% dets/img increase in pre-dawn frames are both from a model where the FiLM gates barely opened (25 epochs, 1,133 images, no night ground truth) — the signal is real but the mechanism was mostly inactive. Phase 4 trains on a substantially larger adversarial dataset with hard negatives, more epochs, and explicit night-aware architecture — the condition-stratified mAP gap (CATI vs ablation on night-only val) is expected to be significantly larger than the aggregate +0.031.

---

## Branch Organisation

This repo has **two branches with no common ancestor** — they serve completely different purposes and cannot be merged.

| Branch | Purpose | What it contains |
|--------|---------|-----------------|
| `fresh` | Research, training, data pipeline | Full CATI architecture (`src/`), all notebooks, data collection scripts, training configs, GDino labelling pipeline, Phase 3/4 work. This is where all ML development happens. |
| `main` | HF Space deployment | `app.py` (Streamlit dashboard), `Dockerfile`, `server.py` (FastAPI), `requirements.txt`. Picks up model weights from HF Hub. No training code. |

**Why separate?** `main` was initialised independently when the HF Space was set up. `fresh` carries the full research history (90-camera network, graph construction, training experiments). Merging would surface hundreds of training-only dependencies that break the Space's slim deployment image and confuse the HF Spaces build system.

**Workflow:** train on `fresh` → push weights to [HF Model Hub](https://huggingface.co/SuhxsReddy/cati-singapore) → `main`'s `app.py` loads weights from HF at Space startup. The two branches are linked through HF, not through git.

**If you're here to understand the model:** start on `fresh`. Read the notebooks in order (`01_` → `10_`), then `src/models/` for the architecture.

**If you're here to run the live demo:** see [main](https://github.com/Suhxs-Reddy/sg-smart-city-analytics/tree/main) or visit the [HF Space](https://huggingface.co/spaces/SuhxsReddy/SingaporeAnalytics) directly.

### `fresh` Branch: Key Scripts

```
scripts/
├── collect_data_colab.py          # Async LTA collector — normal daytime collection
├── collect_night_baseline_colab.py# Colab notebook version of night collector (4 cells)
├── collect_night_standalone.py    # Pure-Python night collector for `colab exec --file`
│                                  # Phase-aware (dusk/night/pre_dawn/dawn), per-frame
│                                  # quality metrics (blur, brightness, contrast, blob_count)
│                                  # Flags: is_challenging, is_candidate_neg
│                                  # Output: Drive/sg_smart_city/data/raw_night_baseline/night_YYYY-MM-DD/
├── run_night_collect.sh           # Cron wrapper — opens Colab session, mounts Drive,
│                                  # uploads script, runs with 6h timeout, logs to logs/
└── label_sg_vehicles.py           # GDino auto-labelling pipeline (CONF_AUTO threshold)
```

**Night collection automation:** `run_night_collect.sh` is invoked by a macOS crontab (IST timezone) at 18:30 SGT and 00:30 SGT nightly. `caffeinate -i` prevents Mac sleep during the 6-hour collection window. Logs in `logs/night-YYYYMMDD-HHMM.log`.

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
