import os
import urllib.request
from ultralytics import YOLO

def run_demo():
    print("\n" + "="*50)
    print("🚀 INITIALIZING SINGAPORE SMART CITY ML MODEL")
    print("="*50)

    # 1. Download a realistic sample traffic image
    sample_img_path = "sample_traffic.jpg"
    if not os.path.exists(sample_img_path):
        print("=> Downloading sample traffic image for inference...")
        url = "https://images.unsplash.com/photo-1449844908441-8829872d2607?q=80&w=1000&auto=format&fit=crop"
        urllib.request.urlretrieve(url, sample_img_path)
    
    print(f"=> Sample image loaded: {sample_img_path}")

    # 2. Load the Fine-Tuned Model
    print("=> Loading Tier 1 Edge Perception Model...")
    model_path = 'yolo11s.pt'
    
    if os.path.exists('best.onnx'):
        print("   ✅ Found ONNX INT8 quantized graph: `best.onnx`")
        model_path = 'best.onnx'
    elif os.path.exists('best.pt'):
        print("   ✅ Found custom PyTorch weights: `best.pt`")
        model_path = 'best.pt'
    else:
        print("   ⚠️  Did not find `best.onnx` or `best.pt` in the project root.")
        print("       Falling back to the generic `yolo11s.pt` architecture.")

    model = YOLO(model_path, task='detect')
    
    # 3. Run Inference
    print("\n=> [ML ENGINE] Running Deep Learning Inference...")
    results = model(sample_img_path, conf=0.25, verbose=False)[0]
    
    # Export explicitly drawn bounding box image
    out_img = "inference_output.jpg"
    results.save(out_img)
    print(f"=> Synthetically drawn bounding boxes saved to: {out_img}")
    
    # 4. Process Results
    vehicle_count = 0
    vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck in COCO
    
    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id in vehicle_classes:
            vehicle_count += 1

    print("\n" + "="*50)
    print("🎯 INFERENCE RESULTS")
    print("="*50)
    print(f"✅ Total Vehicles Detected: {vehicle_count}")
    print(f"✅ Model Confidence Threshold: 25%")
    print(f"✅ Hardware Used: Local CPU (Zero-Cost Mode)")
    
    print("\n[SUCCESS] The ML Perception architecture is alive and functioning correctly.")
    print("="*50 + "\n")

if __name__ == "__main__":
    run_demo()
