"""
Singapore Smart City — Data Collection Script for Google Colab

Usage (in Colab):
    1. Open in Google Colab (CPU runtime, no GPU needed)
    2. Run Cell 1 (setup), then Cell 2 (collect)

OUTPUT_TAG controls where images are saved:
    'raw'                  → data/raw/                   (normal collection)
    'raw_adversarial'      → data/raw_adversarial/       (night/sunrise/rain/sunset)
    'raw_dark_baseline'    → data/raw_dark_baseline/     (3–5am, near-zero traffic,
                                                           streetlights only — hard negatives)

COLLECTION MODES:
    Normal          OUTPUT_TAG='raw',             DURATION_HOURS=6,  INTERVAL_SECS=90
    Adversarial     OUTPUT_TAG='raw_adversarial', DURATION_HOURS=3,  INTERVAL_SECS=90
    Dark baseline   OUTPUT_TAG='raw_dark_baseline',DURATION_HOURS=2, INTERVAL_SECS=90
                    → Run 3am–5am SGT. Captures empty checkpoint frames with full
                      streetlight configuration and no/minimal vehicles. Used as
                      hard negatives in training so the model learns that streetlights
                      alone do not indicate a vehicle.
"""

# ── Cell 1: Setup ─────────────────────────────────────────────────────────────
# Run this once per Colab session

from google.colab import drive
drive.mount('/content/drive')

import os, subprocess, sys, yaml
from pathlib import Path

subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                'aiohttp', 'Pillow', 'imagehash', 'pyyaml', 'click', 'pandas'],
               check=True)

REPO_DIR = '/content/sg-smart-city-analytics'
if not os.path.exists(REPO_DIR):
    subprocess.run(['git', 'clone', '-b', 'fresh',
                    'https://github.com/Suhxs-Reddy/sg-smart-city-analytics.git',
                    REPO_DIR], check=True)

os.chdir(REPO_DIR)
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

print('Ready. Repo:', os.getcwd())
print('Branch:', subprocess.check_output(['git', 'branch', '--show-current']).decode().strip())


# ── Cell 2: Collect ───────────────────────────────────────────────────────────
# Set OUTPUT_TAG, DURATION_HOURS, then run.
# For dark baseline: start this cell at 3am SGT, let it run until 5am.

OUTPUT_TAG     = 'raw_adversarial'    # see docstring above for options
DURATION_HOURS = 3
INTERVAL_SECS  = 90

DRIVE_DATA = Path('/content/drive/MyDrive/sg_smart_city/data')
OUT_DIR    = DRIVE_DATA / OUTPUT_TAG
OUT_DIR.mkdir(parents=True, exist_ok=True)
Path('logs').mkdir(exist_ok=True)

# Patch config
cfg_path = Path('configs/collection_config.yaml')
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)
cfg['collection']['output_dir'] = str(OUT_DIR)
cfg['logging']['log_file'] = None
with open(cfg_path, 'w') as f:
    yaml.dump(cfg, f)

MODE_NOTES = {
    'raw':               'normal collection',
    'raw_adversarial':   'night / sunrise / sunset / rain — adversarial training data',
    'raw_dark_baseline': '3–5am near-zero traffic — streetlight hard negatives',
}
print(f'Mode    : {MODE_NOTES.get(OUTPUT_TAG, OUTPUT_TAG)}')
print(f'Output  : {OUT_DIR}')
print(f'Duration: {DURATION_HOURS}h  |  Interval: {INTERVAL_SECS}s')
print(f'Cycles  : ~{int(DURATION_HOURS * 3600 / INTERVAL_SECS)}  |  Cameras per cycle: ~8\n')

subprocess.run([
    sys.executable, '-m', 'src.ingestion.collector',
    '--duration', str(DURATION_HOURS),
    '--interval', str(INTERVAL_SECS),
], check=True)
