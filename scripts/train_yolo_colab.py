"""
Singapore Smart City — YOLOv11 Fine-Tuning Script for Colab/Kaggle

This script is designed to be uploaded to Google Colab or Kaggle Notebooks
and run on a free T4 GPU. It handles:
1. Environment setup (clone repo, install deps)
2. Dataset download (UA-DETRAC from Roboflow)
3. YOLOv11s fine-tuning with traffic-optimized hyperparams
4. Evaluation and export to ONNX
5. Saving results to Google Drive / Kaggle output

Usage on Colab:
    1. Upload this file or paste into a Colab notebook
    2. Set runtime to GPU (T4)
    3. Run all cells

Usage on Kaggle:
    1. Create new notebook, attach GPU
    2. Paste this script
    3. Run
"""

import os
import subprocess
import sys
from pathlib import Path


# =============================================================================
# Step 1: Environment Setup
# =============================================================================

def setup_environment():
    """Install dependencies and clone the project repo."""
    print("=" * 60)
    print("  Step 1: Environment Setup")
    print("=" * 60)

    # Check GPU
    gpu_check = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                                 "--format=csv,noheader"], capture_output=True, text=True)
    if gpu_check.returncode == 0:
        print(f"  GPU: {gpu_check.stdout.strip()}")
    else:
        print("  ⚠️  No GPU detected — training will be very slow")

    # Install dependencies
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "ultralytics>=8.3.0", "roboflow", "onnx", "onnxruntime"],
                   check=True)

    # Clone project (for configs)
    if not Path("sg-smart-city-analytics").exists():
        subprocess.run(["git", "clone",
                        "https://github.com/Suhxs-Reddy/sg-smart-city-analytics.git"],
                       check=True)

    print("  ✅ Environment ready\n")


# =============================================================================
# Step 2: Dataset Download
# =============================================================================

def download_dataset(roboflow_api_key: str = None):
    """Download UA-DETRAC dataset from Roboflow.

    If no API key is provided, uses alternative download.
    """
    print("=" * 60)
    print("  Step 2: Dataset Download (UA-DETRAC)")
    print("=" * 60)

    dataset_dir = Path("datasets/ua_detrac")

    if dataset_dir.exists() and list(dataset_dir.rglob("*.jpg")):
        print(f"  Dataset already exists at {dataset_dir}")
        return str(dataset_dir)

    if roboflow_api_key:
        from roboflow import Roboflow
        rf = Roboflow(api_key=roboflow_api_key)
        project = rf.workspace().project("ua-detrac-yvknh")
        version = project.version(1)
        ds = version.download("yolov11", location=str(dataset_dir))
        print(f"  ✅ Downloaded to {ds.location}")
        return ds.location
    else:
        print("  ⚠️  No Roboflow API key provided")
        print("  To get one: https://app.roboflow.com → Settings → API Key")
        print("  Then re-run: download_dataset('your_api_key')")
        return None


# =============================================================================
# Step 3: Fine-Tune YOLOv11s
# =============================================================================

def train_yolo(
    dataset_yaml: str,
    epochs: int = 100,
    batch_size: int = 16,
    img_size: int = 640,
    run_name: str = "sg_traffic_v1",
):
    """Fine-tune YOLOv11s on the traffic dataset.

    Uses hyperparameters optimized for Singapore traffic cameras on T4 GPU.
    """
    print("=" * 60)
    print(f"  Step 3: Fine-Tuning YOLOv11s ({epochs} epochs)")
    print("=" * 60)

    from ultralytics import YOLO

    model = YOLO("yolo11s.pt")

    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        patience=15,

        # Optimizer
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=3,

        # Augmentation (traffic-optimized)
        hsv_h=0.015,
        hsv_s=0.7,       # Weather variation
        hsv_v=0.4,       # Day/night variation
        degrees=0.0,     # No rotation (fixed cameras)
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        close_mosaic=10,

        # Hardware
        device=0,
        workers=2,
        amp=True,        # Mixed precision on T4

        # Output
        project="runs/train",
        name=run_name,
        save_period=10,
        exist_ok=True,
        verbose=True,
        plots=True,
    )

    print(f"\n  ✅ Training complete")
    print(f"  Best weights: runs/train/{run_name}/weights/best.pt")

    return results


# =============================================================================
# Step 4: Evaluate
# =============================================================================

def evaluate_model(
    model_path: str = "runs/train/sg_traffic_v1/weights/best.pt",
    dataset_yaml: str = None,
):
    """Evaluate fine-tuned model and print metrics."""
    print("=" * 60)
    print("  Step 4: Evaluation")
    print("=" * 60)

    from ultralytics import YOLO

    model = YOLO(model_path)
    metrics = model.val(data=dataset_yaml)

    print(f"\n  Results:")
    print(f"    mAP@50:     {metrics.box.map50:.4f}")
    print(f"    mAP@50-95:  {metrics.box.map:.4f}")
    print(f"    Precision:   {metrics.box.mp:.4f}")
    print(f"    Recall:      {metrics.box.mr:.4f}")

    return metrics


# =============================================================================
# Step 5: Export to ONNX
# =============================================================================

def export_model(
    model_path: str = "runs/train/sg_traffic_v1/weights/best.pt",
):
    """Export fine-tuned model to ONNX format for deployment."""
    print("=" * 60)
    print("  Step 5: Export to ONNX")
    print("=" * 60)

    from ultralytics import YOLO

    model = YOLO(model_path)
    onnx_path = model.export(format="onnx", imgsz=640, simplify=True)

    print(f"  ✅ Exported to: {onnx_path}")
    return onnx_path


# =============================================================================
# Step 6: Save to Google Drive (Colab) or Output (Kaggle)
# =============================================================================

def save_results(run_name: str = "sg_traffic_v1"):
    """Save training results to persistent storage."""
    print("=" * 60)
    print("  Step 6: Saving Results")
    print("=" * 60)

    src = Path(f"runs/train/{run_name}")

    # Try Google Drive (Colab)
    drive_path = Path("/content/drive/MyDrive/sg_smart_city")
    if drive_path.parent.exists():
        import shutil
        dst = drive_path / run_name
        dst.mkdir(parents=True, exist_ok=True)
        # Copy weights
        for w in (src / "weights").glob("*.pt"):
            shutil.copy2(w, dst / w.name)
        # Copy metrics
        for f in src.glob("*.csv"):
            shutil.copy2(f, dst / f.name)
        for f in src.glob("*.png"):
            shutil.copy2(f, dst / f.name)
        print(f"  ✅ Saved to Google Drive: {dst}")
        return

    # Try Kaggle output
    kaggle_path = Path("/kaggle/working")
    if kaggle_path.exists():
        import shutil
        dst = kaggle_path / run_name
        shutil.copytree(src, dst, dirs_exist_ok=True)
        print(f"  ✅ Saved to Kaggle output: {dst}")
        return

    print(f"  Results remain at: {src}")


# =============================================================================
# Main — Run everything
# =============================================================================

def main():
    """Run the complete fine-tuning pipeline.

    Modify ROBOFLOW_API_KEY and other parameters below as needed.
    """
    # ---- CONFIGURE THESE ----
    ROBOFLOW_API_KEY = None           # Set your Roboflow API key here
    EPOCHS = 100                       # Start with 50 for a quick test
    RUN_NAME = "sg_traffic_v1"
    # -------------------------

    print("\n🇸🇬 Singapore Smart City — YOLOv11 Fine-Tuning\n")

    # Step 1
    setup_environment()

    # Step 2
    dataset_location = download_dataset(ROBOFLOW_API_KEY)
    if not dataset_location:
        print("\n⚠️  Set ROBOFLOW_API_KEY and re-run to continue\n")
        return

    # Find dataset YAML
    dataset_yaml = None
    for yml in Path(dataset_location).rglob("*.yaml"):
        dataset_yaml = str(yml)
        break

    if not dataset_yaml:
        print("❌ No dataset YAML found")
        return

    print(f"  Using dataset: {dataset_yaml}\n")

    # Step 3
    train_yolo(dataset_yaml, epochs=EPOCHS, run_name=RUN_NAME)

    # Step 4
    evaluate_model(
        f"runs/train/{RUN_NAME}/weights/best.pt",
        dataset_yaml,
    )

    # Step 5
    export_model(f"runs/train/{RUN_NAME}/weights/best.pt")

    # Step 6
    save_results(RUN_NAME)

    print("\n" + "=" * 60)
    print("  🎉 FINE-TUNING COMPLETE")
    print("  Next: Download best.pt and use it with the pipeline")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
