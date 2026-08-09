"""
Singapore Smart City — Data Collection Script for Google Colab

Usage (in Colab):
    1. Open in Google Colab (CPU runtime, no GPU needed)
    2. Run Cell 1 (setup), then Cell 2 (collect)

OUTPUT_TAG controls where images are saved:
    'raw'            → data/raw/                 (normal collection)
    'raw_adversarial'→ data/raw_adversarial/     (night/sunrise/rain runs)
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
# Edit OUTPUT_TAG and DURATION_HOURS, then run

OUTPUT_TAG     = 'raw_adversarial'   # 'raw' for normal runs, 'raw_adversarial' for night/sunrise/rain
DURATION_HOURS = 3
INTERVAL_SECS  = 90

DRIVE_DATA = Path('/content/drive/MyDrive/sg_smart_city/data')
OUT_DIR    = DRIVE_DATA / OUTPUT_TAG
OUT_DIR.mkdir(parents=True, exist_ok=True)
Path('logs').mkdir(exist_ok=True)

# Patch config to point at the right output dir
cfg_path = Path('configs/collection_config.yaml')
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)
cfg['collection']['output_dir'] = str(OUT_DIR)
cfg['logging']['log_file'] = None
with open(cfg_path, 'w') as f:
    yaml.dump(cfg, f)

print(f'Output  : {OUT_DIR}')
print(f'Duration: {DURATION_HOURS}h  |  Interval: {INTERVAL_SECS}s')
print(f'Cycles  : ~{int(DURATION_HOURS * 3600 / INTERVAL_SECS)}  |  Cameras per cycle: ~90\n')

subprocess.run([
    sys.executable, '-m', 'src.ingestion.collector',
    '--duration', str(DURATION_HOURS),
    '--interval', str(INTERVAL_SECS),
], check=True)
