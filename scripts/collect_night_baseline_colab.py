"""
Singapore Smart City — Full-Night Baseline Collector (Google Colab)

Runs unattended from sunset to sunrise SGT. Start once per night, let it run.
Organises by night_date so different nights never mix in train/val splits.
Tags every frame with quality metrics for negative mining — no GPU required.

PHASES (SGT, approximate year-round for Singapore ~1°N):
  dusk       18:30 – 20:30   lingering daylight, lens flare, camera adaptation
  night      20:30 – 04:30   full dark; 3–4:30am = near-zero checkpoint traffic
  pre_dawn   04:30 – 06:00   sky brightens, blue-hour glow
  dawn       06:00 – 07:15   sunrise artifacts; script self-terminates at 07:15

QUALITY FLAGS (computed without GPU):
  blur_score          Laplacian variance — low = wet lens / fog / haze
  brightness          Mean grayscale — context for glare vs deep dark
  contrast            Std grayscale — low = flat overcast / uniform scene
  blob_count          Connected bright regions (>200 px value, >50px area) —
                      proxy for headlights / street lamps / vehicles
  is_challenging      blur<150 OR contrast<15 OR (night + brightness>90)
  is_candidate_neg    blob_count<8 AND phase in {night, pre_dawn}

DRIVE STRUCTURE:
  data/raw_night_baseline/
    night_YYYY-MM-DD/           ← evening date the night started
      {camera_id}/
        HH-MM-SS.jpg
      night_YYYY-MM-DD_manifest.csv
"""

# ── Cell 1: Setup ─────────────────────────────────────────────────────────────
# Run once per Colab session. Mounts Drive and clones repo.

from google.colab import drive
drive.mount('/content/drive')

import os, subprocess, sys
from pathlib import Path

subprocess.run([
    sys.executable, '-m', 'pip', 'install', '-q',
    'aiohttp', 'Pillow', 'pyyaml', 'numpy', 'opencv-python-headless', 'pandas',
], check=True)

REPO_DIR = '/content/sg-smart-city-analytics'
if not os.path.exists(REPO_DIR):
    subprocess.run(['git', 'clone', '-b', 'fresh',
                    'https://github.com/Suhxs-Reddy/sg-smart-city-analytics.git',
                    REPO_DIR], check=True)

os.chdir(REPO_DIR)
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

print('Ready. Branch:', subprocess.check_output(['git', 'branch', '--show-current']).decode().strip())


# ── Cell 2: Collect (self-terminates at 07:15 SGT) ───────────────────────────
# Start this cell any time after ~18:30 SGT or earlier — it will skip forward
# to the first active phase and stop after dawn.
# Can also start at midnight if you missed the evening — it picks up in 'night'.

import asyncio, csv, hashlib, io, json, time
import datetime
import aiohttp
import numpy as np
from PIL import Image
import cv2

# ── Constants ─────────────────────────────────────────────────────────────────
SGT      = datetime.timezone(datetime.timedelta(hours=8))
CAMERAS  = ['2701', '2702', '2704', '4703', '4712', '4713', '4798', '4799']
INTERVAL = 90   # seconds — matches camera hardware refresh rate

# Phase boundaries as decimal hours (AM hours shifted by +24 to keep night contiguous)
# e.g. 01:30 SGT → 25.5  so the 20:30–28:5 "night" band is one continuous range
PHASES = [
    (18.5,  20.5, 'dusk'),
    (20.5,  28.5, 'night'),
    (28.5,  30.0, 'pre_dawn'),
    (30.0,  31.25,'dawn'),
]
STOP_HOUR = 31.25   # 07:15 SGT — self-terminate

# Quality thresholds
BLUR_CHALLENGING   = 150    # Laplacian var below this = blurry/wet lens
CONTRAST_FLAT      = 15     # std below this = overcast/uniform scene
BRIGHTNESS_GLARE   = 90     # bright at night = unusual glare
BLOB_AREA_MIN      = 50     # px — ignore salt noise
BLOB_BRIGHT_THRESH = 200    # pixel value threshold for bright-region detection
BLOB_NEG_MAX       = 8      # fewer blobs than this = candidate negative frame

# API
TRAFFIC_URL  = 'https://api.data.gov.sg/v1/transport/traffic-images'
WEATHER_URL  = 'https://api.data.gov.sg/v1/environment/24-hour-weather-forecast'
PM25_URL     = 'https://api.data.gov.sg/v1/environment/pm25'
TEMP_URL     = 'https://api.data.gov.sg/v1/environment/air-temperature'

# Drive output
DRIVE_DATA = Path('/content/drive/MyDrive/sg_smart_city/data')

MANIFEST_FIELDS = [
    'camera_id', 'night_date', 'timestamp_sgt', 'phase',
    'blur_score', 'brightness', 'contrast', 'blob_count',
    'weather_condition', 'pm25_value', 'temperature_celsius',
    'is_challenging', 'is_candidate_neg',
    'img_hash', 'filename',
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def sgt_hour(dt: datetime.datetime) -> float:
    """Decimal hour in SGT, with AM hours shifted +24 so night is contiguous."""
    h = dt.astimezone(SGT).hour + dt.astimezone(SGT).minute / 60
    return h + 24 if h < 12 else h


def get_phase(h: float) -> str | None:
    for start, end, label in PHASES:
        if start <= h < end:
            return label
    return None


def night_date_for(dt: datetime.datetime) -> str:
    """Return the evening date that started this night.
    Hours before noon belong to the previous calendar date's night."""
    sgt = dt.astimezone(SGT)
    if sgt.hour < 12:
        return (sgt - datetime.timedelta(days=1)).strftime('%Y-%m-%d')
    return sgt.strftime('%Y-%m-%d')


def quality_metrics(img: Image.Image) -> dict:
    arr  = np.array(img.convert('RGB'))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    contrast   = float(gray.std())

    _, thresh = cv2.threshold(gray, BLOB_BRIGHT_THRESH, 255, cv2.THRESH_BINARY)
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(thresh)
    blob_count = sum(
        1 for i in range(1, n_labels)
        if stats[i, cv2.CC_STAT_AREA] >= BLOB_AREA_MIN
    )

    return {
        'blur_score': round(blur_score, 1),
        'brightness': round(brightness, 1),
        'contrast':   round(contrast, 1),
        'blob_count': blob_count,
    }


def frame_flags(m: dict, phase: str) -> dict:
    is_challenging = (
        m['blur_score'] < BLUR_CHALLENGING or
        m['contrast']   < CONTRAST_FLAT    or
        (phase in ('night', 'pre_dawn') and m['brightness'] > BRIGHTNESS_GLARE)
    )
    is_candidate_neg = (
        m['blob_count'] < BLOB_NEG_MAX and
        phase in ('night', 'pre_dawn')
    )
    return {
        'is_challenging':   is_challenging,
        'is_candidate_neg': is_candidate_neg,
    }


async def fetch_json(session: aiohttp.ClientSession, url: str) -> dict | None:
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as r:
            return await r.json() if r.status == 200 else None
    except Exception:
        return None


async def fetch_image(session: aiohttp.ClientSession, url: str) -> bytes | None:
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as r:
            return await r.read() if r.status == 200 else None
    except Exception:
        return None


def extract_weather(forecast: dict | None) -> str:
    try:
        return forecast['items'][0]['general']['forecast']
    except (TypeError, KeyError, IndexError):
        return 'unknown'


def extract_pm25(pm25: dict | None, lng: float) -> float | None:
    try:
        readings = pm25['items'][0]['readings']['pm25_one_hourly']
        region   = 'west' if lng < 103.75 else ('east' if lng > 103.9 else 'central')
        return readings.get(region)
    except (TypeError, KeyError, IndexError):
        return None


def extract_temp(temp_data: dict | None) -> float | None:
    try:
        vals = [r['value'] for r in temp_data['items'][0]['readings'] if 'value' in r]
        return round(sum(vals) / len(vals), 1) if vals else None
    except (TypeError, KeyError, IndexError):
        return None


# ── Main collector loop ───────────────────────────────────────────────────────

now        = datetime.datetime.now(SGT)
nd         = night_date_for(now)
night_dir  = DRIVE_DATA / 'raw_night_baseline' / f'night_{nd}'
night_dir.mkdir(parents=True, exist_ok=True)
manifest_p = night_dir / f'night_{nd}_manifest.csv'

# Camera location cache (fetched once from traffic API)
cam_locs: dict[str, tuple[float, float]] = {}

seen_hashes: dict[str, str] = {}
total_saved  = 0
cycle        = 0

print(f'Night date  : {nd}')
print(f'Output      : {night_dir}')
print(f'Interval    : {INTERVAL}s')
print()
print('Phases (SGT):')
for s, e, label in PHASES:
    sh, sm = divmod(int((s % 24) * 60), 60)
    eh, em = divmod(int((e % 24) * 60), 60)
    print(f'  {sh:02d}:{sm:02d} – {eh:02d}:{em:02d}  {label}')
print(f'  Stops at 07:15 SGT')
print()

with open(manifest_p, 'w', newline='') as mf:
    writer = csv.DictWriter(mf, fieldnames=MANIFEST_FIELDS)
    writer.writeheader()

    async def run():
        nonlocal total_saved, cycle

        async with aiohttp.ClientSession() as http:
            while True:
                now   = datetime.datetime.now(SGT)
                h_now = sgt_hour(now)
                phase = get_phase(h_now)

                if h_now >= STOP_HOUR:
                    print(f'\n[{now.strftime("%H:%M")} SGT] Sunrise passed — collection complete.')
                    break

                if phase is None:
                    # Before first phase starts — wait
                    next_start = 18.5
                    wait_mins  = int((next_start - (h_now % 24)) * 60)
                    print(f'[{now.strftime("%H:%M")} SGT] Waiting for dusk (~18:30). '
                          f'~{wait_mins} min remaining.')
                    await asyncio.sleep(300)
                    continue

                cycle += 1

                # Parallel: traffic images + weather context
                traffic_raw, forecast_raw, pm25_raw, temp_raw = await asyncio.gather(
                    fetch_json(http, TRAFFIC_URL),
                    fetch_json(http, WEATHER_URL),
                    fetch_json(http, PM25_URL),
                    fetch_json(http, TEMP_URL),
                )

                weather   = extract_weather(forecast_raw)
                temp_c    = extract_temp(temp_raw)

                if not traffic_raw:
                    print(f'[{now.strftime("%H:%M")} SGT] Traffic API failed — skipping cycle')
                    await asyncio.sleep(INTERVAL)
                    continue

                cameras = {
                    c['camera_id']: c
                    for c in traffic_raw['items'][0]['cameras']
                    if c['camera_id'] in CAMERAS
                }

                # Cache lat/lng once
                for cid, c in cameras.items():
                    if cid not in cam_locs:
                        cam_locs[cid] = (
                            c['location']['latitude'],
                            c['location']['longitude'],
                        )

                # Download all 8 cameras in parallel
                async def _noop(): return None
                raw_bytes = await asyncio.gather(*[
                    fetch_image(http, cameras[cid]['image'])
                    if cid in cameras else _noop()
                    for cid in CAMERAS
                ])

                ts_str    = now.strftime('%H-%M-%S')
                saved_now = 0

                for cid, raw in zip(CAMERAS, raw_bytes):
                    if raw is None:
                        continue

                    img_hash = hashlib.sha256(raw).hexdigest()[:16]
                    if seen_hashes.get(cid) == img_hash:
                        continue   # camera hasn't refreshed yet
                    seen_hashes[cid] = img_hash

                    try:
                        img = Image.open(io.BytesIO(raw)).convert('RGB')
                    except Exception:
                        continue

                    m     = quality_metrics(img)
                    flags = frame_flags(m, phase)

                    lat, lng = cam_locs.get(cid, (1.35, 103.82))
                    pm25_val = extract_pm25(pm25_raw, lng)

                    cam_dir = night_dir / cid
                    cam_dir.mkdir(exist_ok=True)
                    fname = f'{ts_str}.jpg'
                    img.save(cam_dir / fname, quality=92)

                    writer.writerow({
                        'camera_id':          cid,
                        'night_date':         nd,
                        'timestamp_sgt':      now.strftime('%H:%M:%S'),
                        'phase':              phase,
                        **m,
                        'weather_condition':  weather,
                        'pm25_value':         pm25_val,
                        'temperature_celsius':temp_c,
                        **flags,
                        'img_hash':           img_hash,
                        'filename':           f'{cid}/{fname}',
                    })
                    saved_now  += 1
                    total_saved += 1

                mf.flush()

                neg_this = sum(
                    1 for cid in CAMERAS
                    if seen_hashes.get(cid)  # was saved (not deduped)
                )
                print(
                    f'[{now.strftime("%H:%M")} SGT | {phase:9s}] '
                    f'cycle={cycle:4d}  saved={saved_now}/8  '
                    f'total={total_saved:5d}  weather={weather}'
                )

                # Exact interval accounting for collection time
                elapsed    = (datetime.datetime.now(SGT) - now).total_seconds()
                sleep_time = max(0, INTERVAL - elapsed)
                await asyncio.sleep(sleep_time)

    await run()

print(f'\nManifest saved: {manifest_p}')
print(f'Total frames  : {total_saved}')


# ── Cell 3: Per-night summary ─────────────────────────────────────────────────
# Run after collection or next morning to review what was captured.

import pandas as pd

df = pd.read_csv(manifest_p)

print(f'Night: {nd}  |  Total frames: {len(df)}')
print()

print('=== Frames by phase ===')
print(df['phase'].value_counts().reindex(['dusk', 'night', 'pre_dawn', 'dawn']))

print()
print('=== Candidate negatives (blob_count < 8, phase: night/pre_dawn) ===')
neg = df[df['is_candidate_neg']]
print(f'  Count: {len(neg)} frames across {neg["camera_id"].nunique()} cameras')
if len(neg):
    print(neg.groupby('camera_id')[['blur_score', 'brightness', 'blob_count']].mean().round(1))

print()
print('=== Challenging frames (wet lens / glare / flat scene) ===')
ch = df[df['is_challenging']]
print(f'  Count: {len(ch)} frames')
print(ch.groupby('phase')['is_challenging'].count())

print()
print('=== Quality by camera ===')
print(df.groupby('camera_id')[['blur_score', 'brightness', 'contrast', 'blob_count']].mean().round(1))

print()
print('=== Weather conditions seen ===')
print(df['weather_condition'].value_counts())


# ── Cell 4: Multi-night split helper ─────────────────────────────────────────
# Run this after you have 3+ nights. Assigns each night to train or val
# so no night appears on both sides of the split.

from pathlib import Path
import pandas as pd
import random

NIGHT_BASELINE_DIR = DRIVE_DATA / 'raw_night_baseline'

all_manifests = sorted(NIGHT_BASELINE_DIR.glob('*/night_*_manifest.csv'))
frames = pd.concat([pd.read_csv(p) for p in all_manifests], ignore_index=True)

nights = sorted(frames['night_date'].unique())
print(f'Nights available: {nights}')

# 80/20 split at night level — shuffle nights, assign first 80% to train
random.seed(42)
shuffled = nights.copy()
random.shuffle(shuffled)
split_idx   = max(1, int(len(shuffled) * 0.8))
train_nights = shuffled[:split_idx]
val_nights   = shuffled[split_idx:]

train = frames[frames['night_date'].isin(train_nights)]
val   = frames[frames['night_date'].isin(val_nights)]

print(f'\nTrain nights: {sorted(train_nights)}  → {len(train)} frames')
print(f'Val nights  : {sorted(val_nights)}    → {len(val)} frames')

neg_train = train[train['is_candidate_neg']]
neg_val   = val[val['is_candidate_neg']]
print(f'\nCandidate negatives — train: {len(neg_train)}  val: {len(neg_val)}')

# Save split manifest
split_out = NIGHT_BASELINE_DIR / 'night_split_manifest.csv'
frames['split'] = frames['night_date'].apply(
    lambda d: 'train' if d in train_nights else 'val'
)
frames.to_csv(split_out, index=False)
print(f'\nSplit manifest saved: {split_out}')
