import json
import random
from datetime import datetime
from pathlib import Path

data_dir = Path("data/processed")
data_dir.mkdir(parents=True, exist_ok=True)

cameras = {
    f"CAM_{i}": {
        "latitude": 1.290270 + random.uniform(-0.05, 0.05),
        "longitude": 103.851959 + random.uniform(-0.05, 0.05),
        "width": 1920,
        "height": 1080
    } for i in range(1, 91)
}

with open(data_dir / "cameras.json", "w") as f:
    json.dump(cameras, f)

detections = {
    cam: {
        "num_vehicles": random.randint(5, 150),
        "timestamp": datetime.now().isoformat()
    } for cam in cameras
}

with open(data_dir / "latest_detections.json", "w") as f:
    json.dump(detections, f)

congestion = {
    cam: {
        "score": detections[cam]["num_vehicles"] / 150.0,
        "level": "High" if detections[cam]["num_vehicles"] > 100 else ("Medium" if detections[cam]["num_vehicles"] > 50 else "Low"),
        "unique_vehicles": detections[cam]["num_vehicles"],
        "timestamp": datetime.now().isoformat()
    } for cam in cameras
}

with open(data_dir / "congestion_scores.json", "w") as f:
    json.dump(congestion, f)

with open(data_dir / "predictions.json", "w") as f:
    json.dump({cam: {"predicted_15m": detections[cam]["num_vehicles"] + random.randint(-10, 20)} for cam in cameras}, f)

print("Mock data generated at data/processed/")
