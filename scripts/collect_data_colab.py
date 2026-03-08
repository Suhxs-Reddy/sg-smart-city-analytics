"""
Singapore Smart City — Data Collection Script for Google Colab

Run data collection from Google Colab when you don't have
an Azure VM ready. Saves data to Google Drive for persistence.

Usage:
    1. Open in Google Colab (no GPU needed)
    2. Mount Google Drive
    3. Run the script
    4. Data is saved to Google Drive

One collection cycle captures:
    - 90 traffic camera images
    - Weather, temperature, PM2.5, taxi metadata
    - ~50 MB per hour at 1-minute intervals
"""

import os
import subprocess
import sys


def setup():
    """Install dependencies and clone repo."""
    print("Setting up environment...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "aiohttp", "Pillow", "imagehash", "pyyaml", "click", "pandas"],
                   check=True)

    if not os.path.exists("sg-smart-city-analytics"):
        subprocess.run(["git", "clone",
                        "https://github.com/Suhxs-Reddy/sg-smart-city-analytics.git"],
                       check=True)

    os.chdir("sg-smart-city-analytics")
    print("✅ Ready\n")


def mount_drive():
    """Mount Google Drive for persistent storage."""
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        drive_dir = "/content/drive/MyDrive/sg_smart_city/data/raw"
        os.makedirs(drive_dir, exist_ok=True)
        print(f"✅ Drive mounted, data will be saved to: {drive_dir}")
        return drive_dir
    except ImportError:
        print("Not running on Colab, using local storage")
        os.makedirs("data/raw", exist_ok=True)
        return "data/raw"


def collect(duration_hours: float = 6, interval_seconds: int = 60):
    """Run data collection.

    Args:
        duration_hours: How long to collect (6 hours ≈ 360 cycles ≈ 300 MB)
        interval_seconds: Seconds between collection cycles
    """
    data_dir = mount_drive()

    print(f"\n🇸🇬 Starting data collection")
    print(f"  Duration: {duration_hours} hours")
    print(f"  Interval: {interval_seconds}s")
    print(f"  Estimated cycles: {int(duration_hours * 3600 / interval_seconds)}")
    print(f"  Estimated size: ~{int(duration_hours * 50)} MB\n")

    # Override the default output directory
    os.environ["SG_DATA_DIR"] = data_dir

    subprocess.run([
        sys.executable, "-m", "src.ingestion.collector",
        "--duration", str(duration_hours),
        "--interval", str(interval_seconds),
        "--output", data_dir,
    ], check=True)


if __name__ == "__main__":
    setup()
    collect(duration_hours=6, interval_seconds=60)
