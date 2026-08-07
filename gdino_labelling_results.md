# Grounding DINO Labelling Results — Phase 3

**Run date:** 2026-08-07  
**Notebook:** 10_label_sg_vehicles.ipynb  
**CONF_AUTO:** 0.45  **CONF_REVIEW:** 0.25

## Cell 6 — Run Summary

| Metric | Value |
|--------|-------|
| Total images | 1,800 |
| Auto-labelled (≥0.45) | 1,295 |
| Review queue (0.25–0.45) | 1,659 |
| No detections | 127 |
| Total detections | 26,860 |
| Auto-accept rate | 26.1% (7,023 / 26,860) |

## Cell 7 — Detection Counts by Class

| Class | Detections | Mean Conf | Above 0.45? |
|-------|-----------|-----------|-------------|
| car | 22,398 | 0.396 | No |
| lorry | 1,064 | 0.313 | No |
| bus | 908 | 0.366 | No |
| tipper_truck | 685 | 0.312 | No |
| taxi | 669 | 0.322 | No |
| container_truck | 628 | 0.307 | No |
| motorcycle | 271 | 0.351 | No |
| van | 227 | 0.309 | No |
| prime_mover | 10 | 0.309 | No |
| **scooter** | **0** | — | — |

## Key Observations

- **Scooter (class 2): zero detections** — GDino never fired on a scooter.
- **All class mean confidences are below CONF_AUTO (0.45)** — majority of labels sit in review queue, not auto-accepted.
- **26% auto-accept rate** means training on auto-only labels uses 7,023 detections across 1,295 images.
- **Prime mover: only 10 detections** — very rare in the dataset, will likely underperform.
- **Car dominates at 83.4%** of all detections — class imbalance is significant.
