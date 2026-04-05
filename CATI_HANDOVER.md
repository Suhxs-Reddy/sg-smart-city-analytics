# 🇸🇬 Project Handover: Singapore Smart City CATI Traffic Intelligence

To **Claude** (or the next AI agent):

We are building a world-class traffic analytics platform for Singapore using 90 LTA cameras. The project has just undergone a major **ML Engineering Elevation** from a functional sketch to a production-grade architecture.

## 🚀 The Core Innovation: CATI
**CATI (Context-Aware Traffic Intelligence)** is a novel system that modulates a YOLOv11 backbone in real-time based on environmental metadata.

### Why it's special:
- **Environmental Conditioning**: Uses real-time weather (data.gov.sg), PM2.5, time, and camera-specific priors to adapt feature extraction.
- **FiLM + Attention**: Uses *Feature-wise Linear Modulation* with **Adaptive Gating** and **Squeeze-Excitation Attention**. 
- **Adaptive Gating**: The model learns a context-dependent scalar α ∈ [0,1] to decide *how much* to trust the conditioning (e.g., α → 1 in heavy rain, α → 0 on a clear day).
- **GPS Spatial Priors**: Camera locations are encoded using sinusoidal positional encodings, allowing the model to learn spatial relationships between cameras.

## 🏗️ Architecture Status
The following modules are **fully implemented, linted, and unit-tested**:
- `src/models/attention.py`: SE-Attention, CBAM, and Adaptive Gating logic.
- `src/models/film.py`: Upgraded FiLM layers with residual pathways.
- `src/models/context_encoder.py`: Metadata encoder with GPS and tropical weather augmentation.
- `src/models/cati_detector.py`: The `CATIDetector` (FiLM pathway) and `CATIBackboneWrapper` (YOLO hook integration).
- `src/training/train_cati.py`: Production trainer with **AMP (Mixed Precision)**, **EMA (Exponential Moving Average)**, and **Stratified Validation**.
- `src/training/feature_extractor.py`: Optimized bridge to extract features from frozen YOLO and cache them to disk/Drive for Phase 1 training.

## 🛠️ How to Proceed (Next Steps)

### 1. Data Preparation
The user has raw data in Google Drive (`/content/drive/MyDrive/sg_smart_city/data/raw`).
- **Objective**: Run `src/training/feature_extractor.py` to convert raw images into CATI-ready JSON/Feature samples.
- **Tip**: This must be done on a GPU-enabled environment (Colab) because it runs a frozen YOLOv11s to extract P3/P4/P5 features.

### 2. Phase 1 Training (Context Modules Only)
Run `python -m src.training.train_cati --phase 1`.
- This trains only the ContextEncoder and FiLMGenerator. 
- It's extremely fast and VRAM-efficient because it uses cached features.
- **Target**: Train until the `val_loss` (reconstruction/modulation MSE) stabilizes.

### 3. Phase 2 Training (End-to-End Fine-tuning)
Run `python -m src.training.train_cati --phase 2`.
- Unfreezes the YOLO backbone with a low learning rate (1e-5).
- Jointly optimizes the detection heads and the conditioning pathway.

### 4. Hook Integration Verification
The `CATIBackboneWrapper` in `src/models/cati_detector.py` uses PyTorch forward hooks. 
- **Check**: Ensure the layer indices `[4, 6, 9]` in `HOOK_LAYER_NAMES` match the specific YOLOv11 version being used.

## 🖇️ CI/CD & Hygiene
- **CI**: `.github/workflows/ci.yml` is a multi-stage pipeline: Lint → Core Tests → ML Tests (torch-cpu) → Docker.
- **Quality**: The codebase is **Ruff clean** and fully formatted. Keep it this way.
- **Tests**: Run `pytest tests/` (requires torch) or `pytest tests/ -m "not ml"` (core only).

> [!IMPORTANT]
> **Singapore Focus**: Always emphasize that this is for Singapore. Use tropical weather labels (`heavy_rain`, `haze`) and LTA camera IDs. The model must be robust to torrential rain and night-time glare.

Good luck! This is a high-level project with serious engineering depth. Keep the standards high.
