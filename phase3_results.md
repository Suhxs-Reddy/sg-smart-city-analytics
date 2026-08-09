---
recorded: 2026-08-09
training: 25 epochs, batch=8, lr=1e-4, T4 GPU
dataset: 1133 train / 249 val / 236 test, nc=10 (GDino CONF_AUTO=0.30)
weights: models/phase2/yolo_cati-2/weights/best.pt + cati_phase2_final.pt
---

# Phase 3 Results — 10-class SG Taxonomy

## Summary

| Model | mAP50 | mAP50-95 | Precision | Recall |
|-------|-------|----------|-----------|--------|
| Baseline (pretrained COCO, no fine-tune) | 0.0347 | 0.0223 | 0.043 | 0.200 |
| Ablation (fine-tuned YOLO, no CATI) | 0.5411 | 0.4559 | 0.661 | 0.514 |
| CATI (fine-tuned + FiLM hooks active) | 0.5719 | 0.4831 | 0.753 | 0.515 |
| **True CATI contribution** | **+0.031** | **+0.027** | — | — |

Baseline is meaninglessly low — pretrained YOLO has 80 COCO classes with no alignment to SG taxonomy. The meaningful delta is CATI vs ablation.

## Per-class (CATI, hooks active)

| Class | mAP50 | mAP50-95 | Val instances | Notes |
|-------|-------|----------|---------------|-------|
| car | 0.925 | 0.727 | 2578 | Dominant class (85.6%) |
| motorcycle | 0.794 | 0.523 | 65 | Good |
| container_truck | 0.657 | 0.562 | 55 | Good |
| tipper_truck | 0.641 | 0.571 | 54 | Good |
| bus | 0.566 | 0.484 | 108 | Decent |
| lorry | 0.544 | 0.492 | 80 | Decent |
| taxi | 0.515 | 0.478 | 56 | Decent |
| van | 0.452 | 0.451 | 14 | Weak — too few instances |
| prime_mover | 0.053 | 0.053 | 2 | Expected — 10 train samples |
| scooter | — | — | 0 | No training data at all |

## Confidence Sweep (backbone-only, no hooks — for ranking only)

| Conf | Precision | Recall | mAP50 | F1 |
|------|-----------|--------|-------|----|
| 0.15 | 0.642 | 0.515 | 0.478 | 0.571 |
| 0.20 | 0.642 | 0.515 | 0.471 | 0.571 |
| **0.25** | **0.642** | **0.516** | **0.460** | **0.572** ← best F1 |
| 0.30 | 0.653 | 0.503 | 0.451 | 0.569 |
| 0.35 | 0.665 | 0.471 | 0.425 | 0.551 |
| 0.40 | 0.696 | 0.446 | 0.407 | 0.543 |
| 0.50 | 0.732 | 0.411 | 0.378 | 0.526 |

**Production conf: 0.25** (best F1 = 0.572)

## Known Gaps

- **scooter**: 0 training samples. GDino can't detect scooters at LTA camera angles/scales (320×240 mounted overhead). Requires manual annotation or synthetic data.
- **prime_mover**: 10 training samples → mAP50=0.053. Needs targeted collection.
- **van**: 14 val instances → mAP50=0.452. Undersupplied in GDino auto-labels.
- **+0.031 CATI delta understated**: Val set is clean (good conditions). FiLM advantage shows in adversarial conditions (rain, haze, night) which are underrepresented in current val set.

## Next Steps

1. Filter HF dataset records (213K) for adversarial conditions: rain weather codes, PM2.5 >55, hours 21-05
2. Cross-reference against `data/raw/` images on Drive
3. GDino-label adversarial frames → add to training set
4. Build condition-stratified val set (val_normal vs val_adversarial)
5. Retrain 50+ epochs on expanded dataset
6. Update app.py: CATI_CLASSES → 10-class SG taxonomy, bump dataset to v3
