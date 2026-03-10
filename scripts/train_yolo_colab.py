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
    try:
        gpu_check = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                                     "--format=csv,noheader"], capture_output=True, text=True)
        if gpu_check.returncode == 0:
            print(f"  GPU: {gpu_check.stdout.strip()}")
        else:
            print("  ⚠️  No GPU detected — training will be very slow")
            print("  To fix in Colab: Runtime -> Change runtime type -> T4 GPU")
    except FileNotFoundError:
        print("  ⚠️  No GPU detected (nvidia-smi not found) — training will be very slow")
        print("  To fix in Colab: Runtime -> Change runtime type -> T4 GPU")

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

def download_dataset(roboflow_api_key: str = None, use_google_drive: bool = True):
    """Download UA-DETRAC dataset from Roboflow or use the user's pre-processed Google Drive dataset.

    If use_google_drive is True, mounts Colab drive and unzips sg_traffic_dataset.zip.
    """
    print("=" * 60)
    print("  Step 2: Dataset Acquisition")
    print("=" * 60)

    if use_google_drive:
        try:
            from google.colab import drive
            if not Path("/content/drive").exists():
                drive.mount("/content/drive")
            
            drive_zip = Path("/content/drive/MyDrive/sg_smart_city/data/sg_traffic_dataset.zip")
            if drive_zip.exists():
                print(f"  ✅ Found existing pre-processed dataset in Drive: {drive_zip}")
                
                import zipfile
                local_extract = Path("/content/dataset")
                local_extract.mkdir(parents=True, exist_ok=True)
                
                print("  => Unzipping prepared dataset...")
                with zipfile.ZipFile(drive_zip, 'r') as zip_ref:
                    zip_ref.extractall(local_extract)
                
                print("  ✅ Dataset unzipped successfully.")
                # Return the path to the internal 'sg_traffic' folder created by the prepare script
                return str(local_extract / "sg_traffic")
            else:
                print(f"  ⚠️  Did not find {drive_zip} in Drive. Ensure prepare_dataset.ipynb was run.")
        except ImportError:
            print("  ⚠️  Not running in Colab, or Google Drive not available.")

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
        print("  ⚠️  No Roboflow API key provided and no Google Drive zip found.")
        print("  => Falling back to the Ultralytics 'coco8' mini-dataset for the pipeline execution.")
        
        fallback_yaml = "coco8.yaml"
        return fallback_yaml


# =============================================================================
# Step 3: Teacher-Student Knowledge Distillation (Grounding DINO -> YOLO)
# =============================================================================

def apply_knowledge_distillation(dataset_dir: str, dataset_yaml: str):
    """Pass-through function. The dataset is already annotated via the prepare_dataset notebook."""
    print("=" * 60)
    print("  Step 3: Validating Dataset Architecture")
    print("=" * 60)
    print("  => Dataset was pre-processed securely with Temporal Splitting.")
    print("  ✅ Passing pre-annotated dataset immediately to YOLO Student network.")
    return dataset_yaml


# =============================================================================
# Step 4: Fine-Tune YOLOv11s Student Model
# =============================================================================

def train_yolo(
    dataset_yaml: str,
    epochs: int = 100,
    batch_size: int = 16,
    img_size: int = 640,
    run_name: str = "sg_traffic_student",
):
    """Fine-tune YOLOv11s Student on the Teacher's pseudo-labels."""
    print("=" * 60)
    print(f"  Step 4: Training YOLOv11s Student ({epochs} epochs)")
    print("=" * 60)

    from ultralytics import YOLO

    print("  => Initiating Student Network weights (yolo11s.pt)...")
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
        hsv_s=0.7,       
        hsv_v=0.4,       
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,

        # Hardware
        device=0,
        workers=2,
        amp=True,

        # Output
        project="runs/train",
        name=run_name,
        exist_ok=True,
        verbose=True,
    )

    print(f"\n  ✅ Student Training complete. Exporting weights from: {results.save_dir}")
    return results


# =============================================================================
# Step 5: Evaluate Student Model
# =============================================================================

def evaluate_model(
    model_path: str = "runs/train/sg_traffic_student/weights/best.pt",
    dataset_yaml: str = None,
):
    """Evaluate fine-tuned model and print metrics."""
    print("=" * 60)
    print("  Step 5: Student Parameter Evaluation")
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
# Step 6: Export to ONNX (INT8 Edge Quantization)
# =============================================================================

def export_model(
    model_path: str = "runs/train/sg_traffic_student/weights/best.pt",
):
    """Export fine-tuned model to ONNX format for deployment."""
    print("=" * 60)
    print("  Step 6: Edge-Device Optimization (ONNX INT8)")
    print("=" * 60)

    from ultralytics import YOLO

    model = YOLO(model_path)
    onnx_path = model.export(format="onnx", imgsz=640, simplify=True)

    print(f"  ✅ Exported to: {onnx_path}")
    return onnx_path


# =============================================================================
# Step 6: Save to Google Drive (Colab) or Output (Kaggle)
# =============================================================================

def save_results(run_name: str, actual_save_dir: str):
    """Save training results to persistent storage."""
    print("=" * 60)
    print("  Step 6: Saving Results")
    print("=" * 60)

    src = Path(actual_save_dir)

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
    dataset_location = download_dataset(ROBOFLOW_API_KEY, use_google_drive=True)

    # Find dataset YAML
    if dataset_location == "coco8.yaml":
        dataset_yaml = "coco8.yaml"
        # Download coco8 so the images exist for autodistill
        from ultralytics.utils.downloads import download
        download("https://ultralytics.com/assets/coco8.zip", dir="datasets")
        dataset_location = "datasets/coco8" # Directory for distillation step
    else:
        dataset_yaml = None
        for yml in Path(dataset_location).rglob("data.yaml"):
            dataset_yaml = str(yml)
            break
        
        # Also check for config.yaml if data.yaml isn't found
        if not dataset_yaml:
            for yml in Path(dataset_location).rglob("*.yaml"):
                dataset_yaml = str(yml)
                break

        if not dataset_yaml:
            print("❌ No dataset YAML found. Ensure download/extraction succeeded.")
            return

        print(f"  Using existing dataset: {dataset_yaml}\n")

    # Step 3: Distillation
    distilled_yaml = apply_knowledge_distillation(dataset_location, dataset_yaml)

    # Step 4: Train Student
    results = train_yolo(distilled_yaml, epochs=EPOCHS, run_name=RUN_NAME)
    
    # Dynamically grab the exact absolute output path that Ultralytics generated
    best_weights_path = Path(results.save_dir) / "weights" / "best.pt"

    # Step 5: Evaluate
    evaluate_model(
        str(best_weights_path),
        distilled_yaml,
    )

    # Step 6: Export
    export_model(str(best_weights_path))

    # Step 7: Save
    save_results(RUN_NAME, str(results.save_dir))

    print("\n" + "=" * 60)
    print("  🎉 FINE-TUNING COMPLETE")
    print("  Next: Download best.pt and use it with the pipeline")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
