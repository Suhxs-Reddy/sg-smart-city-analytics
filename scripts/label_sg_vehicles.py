"""
Singapore Vehicle Labelling — Grounding DINO Auto-Label Pipeline

Processes raw LTA camera images (from Google Drive collection runs) and produces:
  1. YOLO-format .txt annotation files for high-confidence detections
  2. Annotated JPEG previews in review/ for human inspection
  3. manifest.csv — per-detection confidence scores to prioritise review queue

Run on Colab (GPU recommended) or locally with CUDA.

Usage:
    python scripts/label_sg_vehicles.py \
        --images_dir /path/to/raw/images \
        --output_dir /path/to/output \
        --conf_auto 0.45 \
        --conf_review 0.25

    Images below conf_review are discarded.
    Images between conf_review and conf_auto go to the human review queue.
    Images above conf_auto are auto-accepted as YOLO labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

# ── Singapore vehicle taxonomy ────────────────────────────────────────────────

SG_CLASSES = [
    "car",
    "motorcycle",
    "scooter",
    "bus",
    "van",
    "lorry",
    "container_truck",
    "prime_mover",
    "tipper_truck",
    "taxi",
]

# Grounding DINO text query — each token is one class (separated by ' . ')
# Multi-word synonyms help the model generalise across camera conditions.
GDINO_QUERY = (
    "car . "
    "motorcycle . "
    "scooter . moped . "
    "bus . double decker bus . "
    "van . delivery van . "
    "lorry . light truck . pickup truck . "
    "container truck . articulated lorry . "
    "prime mover . tractor unit . "
    "tipper truck . dump truck . construction truck . "
    "taxi . cab ."
)

# Map Grounding DINO output text → class index
_TOKEN_TO_CLASS: dict[str, int] = {
    "car": 0,
    "motorcycle": 1, "motorbike": 1,
    "scooter": 2, "moped": 2,
    "bus": 3, "double decker bus": 3,
    "van": 4, "delivery van": 4,
    "lorry": 5, "light truck": 5, "pickup truck": 5,
    "container truck": 6, "articulated lorry": 6,
    "prime mover": 7, "tractor unit": 7,
    "tipper truck": 8, "dump truck": 8, "construction truck": 8,
    "taxi": 9, "cab": 9,
}

# Colours per class for review images
_CLASS_COLORS = [
    "#4fc3f7", "#ff8a65", "#ce93d8", "#a5d6a7",
    "#fff176", "#ffab91", "#ef9a9a", "#80cbc4",
    "#bcaaa4", "#f48fb1",
]


def _load_gdino():
    """Load Grounding DINO from HuggingFace transformers."""
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    model_id = "IDEA-Research/grounding-dino-base"
    log.info(f"Loading Grounding DINO from {model_id} ...")
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    log.info(f"Model on {device}")
    return processor, model, device


def _infer(processor, model, device, image: Image.Image) -> list[dict]:
    """Run Grounding DINO on one PIL image. Returns list of detections."""
    inputs = processor(
        images=image,
        text=GDINO_QUERY,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.20,       # low — we filter by class below
        text_threshold=0.20,
        target_sizes=[image.size[::-1]],
    )[0]

    detections = []
    for box, score, label in zip(
        results["boxes"].cpu().tolist(),
        results["scores"].cpu().tolist(),
        results["labels"],
        strict=False,
    ):
        text = label.strip().lower()
        cls_idx = _TOKEN_TO_CLASS.get(text)
        if cls_idx is None:
            # Try partial match
            for token, idx in _TOKEN_TO_CLASS.items():
                if token in text or text in token:
                    cls_idx = idx
                    break
        if cls_idx is None:
            continue
        x1, y1, x2, y2 = box
        detections.append({
            "cls": cls_idx,
            "label": SG_CLASSES[cls_idx],
            "conf": round(float(score), 4),
            "box_xyxy": [x1, y1, x2, y2],
        })
    return detections


def _to_yolo(det: dict, img_w: int, img_h: int) -> str:
    """Convert xyxy box to YOLO normalised cx cy w h string."""
    x1, y1, x2, y2 = det["box_xyxy"]
    cx = (x1 + x2) / 2 / img_w
    cy = (y1 + y2) / 2 / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return f"{det['cls']} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def _draw_review(image: Image.Image, detections: list[dict], auto_threshold: float) -> Image.Image:
    """Draw bounding boxes on a copy of the image for human review."""
    img = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except OSError:
        font = ImageFont.load_default()

    for det in detections:
        x1, y1, x2, y2 = det["box_xyxy"]
        color = _CLASS_COLORS[det["cls"]]
        border = 2 if det["conf"] >= auto_threshold else 1
        draw.rectangle([x1, y1, x2, y2], outline=color, width=border)
        tag = f"{det['label']} {det['conf']:.2f}"
        draw.text((x1 + 2, y1 + 2), tag, fill=color, font=font)
    return img


def process_directory(
    images_dir: Path,
    output_dir: Path,
    conf_auto: float = 0.45,
    conf_review: float = 0.25,
) -> None:
    labels_dir = output_dir / "labels"
    review_dir = output_dir / "review"
    labels_dir.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        p for p in images_dir.rglob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    log.info(f"Found {len(image_paths)} images in {images_dir}")

    processor, model, device = _load_gdino()

    manifest_rows = []
    auto_count = review_count = skip_count = 0

    for i, img_path in enumerate(image_paths):
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            log.warning(f"Skipping {img_path.name}: {e}")
            continue

        detections = _infer(processor, model, device, image)

        # Filter below minimum confidence
        detections = [d for d in detections if d["conf"] >= conf_review]

        if not detections:
            skip_count += 1
            log.info(f"[{i+1}/{len(image_paths)}] {img_path.name} — no detections above threshold")
            continue

        # Split into auto-accept and review queues
        auto_dets = [d for d in detections if d["conf"] >= conf_auto]
        review_dets = [d for d in detections if d["conf"] < conf_auto]

        # Write YOLO label file (auto-accepted detections only)
        if auto_dets:
            label_path = labels_dir / f"{img_path.stem}.txt"
            with open(label_path, "w") as f:
                for det in auto_dets:
                    f.write(_to_yolo(det, image.width, image.height) + "\n")
            auto_count += 1

        # Write review image (all detections — auto in bold, review in thin border)
        if review_dets or auto_dets:
            review_img = _draw_review(image, detections, conf_auto)
            review_img.save(review_dir / img_path.name)
            review_count += (1 if review_dets else 0)

        # Manifest row
        for det in detections:
            manifest_rows.append({
                "image": img_path.name,
                "class": det["label"],
                "cls_idx": det["cls"],
                "conf": det["conf"],
                "auto_accepted": det["conf"] >= conf_auto,
                "box": json.dumps([round(v, 1) for v in det["box_xyxy"]]),
            })

        if (i + 1) % 50 == 0:
            log.info(f"[{i+1}/{len(image_paths)}] processed — auto: {auto_count}  review: {review_count}")

    # Write manifest
    manifest_path = output_dir / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "class", "cls_idx", "conf", "auto_accepted", "box"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    # Write class names file (for YOLO dataset config)
    (output_dir / "classes.txt").write_text("\n".join(SG_CLASSES) + "\n")

    log.info("─" * 60)
    log.info(f"Done.  {len(image_paths)} images processed")
    log.info(f"  Auto-labelled : {auto_count}")
    log.info(f"  Review queue  : {review_count}  (check {review_dir}/)")
    log.info(f"  No detections : {skip_count}")
    log.info(f"  Manifest      : {manifest_path}")
    log.info(f"  Classes       : {output_dir / 'classes.txt'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Grounding DINO auto-label for Singapore vehicles")
    parser.add_argument("--images_dir", required=True, type=Path, help="Directory of raw LTA camera images")
    parser.add_argument("--output_dir", required=True, type=Path, help="Where to write labels + review images")
    parser.add_argument("--conf_auto", type=float, default=0.45, help="Confidence above which detections are auto-accepted")
    parser.add_argument("--conf_review", type=float, default=0.25, help="Confidence below which detections are discarded")
    args = parser.parse_args()
    process_directory(args.images_dir, args.output_dir, args.conf_auto, args.conf_review)


if __name__ == "__main__":
    main()
