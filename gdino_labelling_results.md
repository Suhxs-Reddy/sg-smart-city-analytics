# Grounding DINO Labelling Results — Phase 3

**Notebook:** 10_label_sg_vehicles.ipynb  
**CONF_REVIEW:** 0.25 (fixed)

---

## Final Run — CONF_AUTO = 0.30

**Run date:** 2026-08-08

| Metric | Value |
|--------|-------|
| Total images | 1,800 |
| Auto-labelled (≥0.30) | 1,618 |
| Review queue (0.25–0.30) | 1,527 |
| No detections | 127 |
| Total detections | 26,860 |
| Auto-accepted detections | 19,639 (73.1%) |
| Review queue detections | 7,221 |

## Detection Counts by Class

| Class | Detections | Mean Conf |
|-------|-----------|-----------|
| car | 22,398 | 0.396 |
| lorry | 1,064 | 0.313 |
| bus | 908 | 0.366 |
| tipper_truck | 685 | 0.312 |
| taxi | 669 | 0.322 |
| container_truck | 628 | 0.307 |
| motorcycle | 271 | 0.351 |
| van | 227 | 0.309 |
| prime_mover | 10 | 0.309 |
| **scooter** | **0** | — |

## vs First Run (CONF_AUTO = 0.45)

| Metric | 0.45 | 0.30 | Delta |
|--------|------|------|-------|
| Auto-labelled images | 1,295 | 1,618 | +323 |
| Auto-accepted detections | 7,023 | 19,639 | +12,616 |
| Review queue detections | 19,837 | 7,221 | −12,616 |

## Known Gaps

- **Scooter (class 2): zero detections at any threshold** — GDino cannot detect scooters at these camera angles/resolutions. Model will not predict this class.
- **Prime mover: 10 detections** — insufficient for reliable training. Expect poor performance on this class.
- **Car dominates at 83.4%** — significant class imbalance. YOLO's class-weighted loss will help but rare classes will still underperform.
