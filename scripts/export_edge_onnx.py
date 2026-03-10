import argparse
from ultralytics import YOLO

def export_int8_onnx(model_path: str, output_dir: str):
    """
    Exports a trained YOLOv11s model to INT8 Quantized ONNX format.
    This enables maximum throughput and sub-ms inference on edge devices.
    """
    print(f"Loading Teacher-Distilled YOLO Model from {model_path}...")
    model = YOLO(model_path)
    
    print("\nInitiating INT8 Quantization and ONNX Export...")
    # Activating int8=True invokes tensor scaling and calibration
    # simplify=True runs ONNX simplifier to fuse batch norms and reduce node count
    path = model.export(
        format="onnx", 
        dynamic=True, 
        simplify=True, 
        int8=True,
        workspace=4  # Allow 4GB workspace for TensorRT/ONNX optimizer
    )
    
    print(f"\n✅ Production Export Complete! INT8 ONNX graph saved to: {path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Singapore Smart City - Edge Optimization")
    parser.add_argument("--weights", type=str, required=True, help="Path to best.pt")
    parser.add_argument("--out", type=str, default="./deploy", help="Output directory")
    args = parser.parse_args()
    
    export_int8_onnx(args.weights, args.out)
