"""
CATI Night Baseline Collector — standalone (no Jupyter, runs via colab run)

Usage:
    colab run --timeout 21600 scripts/collect_night_standalone.py [STOP_HOUR]

STOP_HOUR: decimal SGT hour to self-terminate (AM hours are +24).
    Default 31.25 = 07:15 SGT.
    Evening job passes 24.5 (= 00:30 SGT).
    Dawn job uses default.

Drive must be pre-mounted at /content/drive by the caller (colab drivemount).
"""

import subprocess, sys

subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                'aiohttp', 'Pillow', 'numpy', 'opencv-python-headless',
                'pandas', 'nest_asyncio'],
               check=True)

import asyncio, csv, hashlib, io, json
import datetime
from pathlib import Path

import aiohttp
import numpy as np
from PIL import Image
import cv2

# ── Config ────────────────────────────────────────────────────────────────────

SGT      = datetime.timezone(datetime.timedelta(hours=8))
CAMERAS  = ['2701', '2702', '2704', '4703', '4712', '4713', '4798', '4799']
INTERVAL = 90   # seconds

PHASES = [
    (18.5,  20.5, 'dusk'),
    (20.5,  28.5, 'night'),
    (28.5,  30.0, 'pre_dawn'),
    (30.0,  31.5, 'dawn'),
]

# Stop hour: read from env (set by GHA workflow) or argv or default
import os
STOP_HOUR = float(
    os.environ.get('CATI_STOP_HOUR')
    or (sys.argv[1] if len(sys.argv) > 1 else '31.5')
)

TRAFFIC_URL = 'https://api.data.gov.sg/v1/transport/traffic-images'
WEATHER_URL = 'https://api.data.gov.sg/v1/environment/24-hour-weather-forecast'
PM25_URL    = 'https://api.data.gov.sg/v1/environment/pm25'
TEMP_URL    = 'https://api.data.gov.sg/v1/environment/air-temperature'

DRIVE_DATA = Path('/content/drive/MyDrive/sg_smart_city/data')

BLOB_THRESH    = 200
BLOB_AREA_MIN  = 50
BLOB_NEG_MAX   = 8
BLUR_THRESH    = 150
CONTRAST_FLAT  = 15
GLARE_BRIGHT   = 90

MANIFEST_FIELDS = [
    'camera_id', 'night_date', 'timestamp_sgt', 'phase',
    'blur_score', 'brightness', 'contrast', 'blob_count',
    'weather_condition', 'pm25_value', 'temperature_celsius',
    'is_challenging', 'is_candidate_neg',
    'img_hash', 'filename',
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def sgt_hour(dt: datetime.datetime) -> float:
    h = dt.astimezone(SGT).hour + dt.astimezone(SGT).minute / 60
    return h + 24 if h < 12 else h


def get_phase(h: float) -> str | None:
    for start, end, label in PHASES:
        if start <= h < end:
            return label
    return None


def night_date_for(dt: datetime.datetime) -> str:
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
    _, thresh  = cv2.threshold(gray, BLOB_THRESH, 255, cv2.THRESH_BINARY)
    n, _, stats, _ = cv2.connectedComponentsWithStats(thresh)
    blob_count = sum(1 for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= BLOB_AREA_MIN)
    return {
        'blur_score': round(blur_score, 1),
        'brightness': round(brightness, 1),
        'contrast':   round(contrast, 1),
        'blob_count': blob_count,
    }


def frame_flags(m: dict, phase: str) -> dict:
    return {
        'is_challenging': (
            m['blur_score'] < BLUR_THRESH or
            m['contrast']   < CONTRAST_FLAT or
            (phase in ('night', 'pre_dawn') and m['brightness'] > GLARE_BRIGHT)
        ),
        'is_candidate_neg': (
            m['blob_count'] < BLOB_NEG_MAX and phase in ('night', 'pre_dawn')
        ),
    }


async def fetch_json(s: aiohttp.ClientSession, url: str) -> dict | None:
    try:
        async with s.get(url, timeout=aiohttp.ClientTimeout(total=10)) as r:
            return await r.json() if r.status == 200 else None
    except Exception:
        return None


async def fetch_image(s: aiohttp.ClientSession, url: str) -> bytes | None:
    try:
        async with s.get(url, timeout=aiohttp.ClientTimeout(total=15)) as r:
            return await r.read() if r.status == 200 else None
    except Exception:
        return None


def extract_weather(d: dict | None) -> str:
    try:    return d['items'][0]['general']['forecast']
    except: return 'unknown'


def extract_pm25(d: dict | None, lng: float) -> float | None:
    try:
        readings = d['items'][0]['readings']['pm25_one_hourly']
        region   = 'west' if lng < 103.75 else ('east' if lng > 103.9 else 'central')
        return readings.get(region)
    except: return None


def extract_temp(d: dict | None) -> float | None:
    try:
        vals = [r['value'] for r in d['items'][0]['readings'] if 'value' in r]
        return round(sum(vals) / len(vals), 1) if vals else None
    except: return None


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    now = datetime.datetime.now(SGT)
    nd  = night_date_for(now)

    night_dir  = DRIVE_DATA / 'raw_night_baseline' / f'night_{nd}'
    night_dir.mkdir(parents=True, exist_ok=True)
    manifest_p = night_dir / f'night_{nd}_manifest.csv'

    seen_hashes: dict[str, str] = {}
    cam_locs:    dict[str, tuple[float, float]] = {}
    total_saved  = 0
    cycle        = 0

    print(f'Night date : {nd}')
    print(f'Output     : {night_dir}')
    print(f'Stop hour  : {STOP_HOUR} SGT ({int(STOP_HOUR % 24):02d}:{int((STOP_HOUR % 1)*60):02d})')
    print(f'Interval   : {INTERVAL}s')

    # Resume manifest if this job overlaps a previous one for the same night
    write_header = not manifest_p.exists()

    with open(manifest_p, 'a', newline='') as mf:
        writer = csv.DictWriter(mf, fieldnames=MANIFEST_FIELDS)
        if write_header:
            writer.writeheader()

        async def _noop(): return None

        async with aiohttp.ClientSession() as http:
            while True:
                now   = datetime.datetime.now(SGT)
                h_now = sgt_hour(now)

                if h_now >= STOP_HOUR:
                    print(f'[{now.strftime("%H:%M")} SGT] Reached stop hour {STOP_HOUR} — done.')
                    break

                phase = get_phase(h_now)

                if phase is None:
                    # Before first phase — sleep and wait
                    mins_to_dusk = int((18.5 - (h_now % 24)) * 60) if h_now < 18.5 else 1
                    print(f'[{now.strftime("%H:%M")} SGT] Waiting for dusk in ~{mins_to_dusk} min.')
                    await asyncio.sleep(min(300, mins_to_dusk * 60))
                    continue

                cycle += 1
                loop_start = now

                traffic_raw, forecast_raw, pm25_raw, temp_raw = await asyncio.gather(
                    fetch_json(http, TRAFFIC_URL),
                    fetch_json(http, WEATHER_URL),
                    fetch_json(http, PM25_URL),
                    fetch_json(http, TEMP_URL),
                )

                weather = extract_weather(forecast_raw)
                temp_c  = extract_temp(temp_raw)

                if not traffic_raw:
                    print(f'[{now.strftime("%H:%M")} SGT] Traffic API unavailable — retrying')
                    await asyncio.sleep(INTERVAL)
                    continue

                cameras = {
                    c['camera_id']: c
                    for c in traffic_raw['items'][0]['cameras']
                    if c['camera_id'] in CAMERAS
                }

                for cid, c in cameras.items():
                    if cid not in cam_locs:
                        cam_locs[cid] = (c['location']['latitude'], c['location']['longitude'])

                raw_bytes = await asyncio.gather(*[
                    fetch_image(http, cameras[cid]['image']) if cid in cameras else _noop()
                    for cid in CAMERAS
                ])

                ts_str    = now.strftime('%H-%M-%S')
                saved_now = 0

                for cid, raw in zip(CAMERAS, raw_bytes):
                    if raw is None:
                        continue
                    img_hash = hashlib.sha256(raw).hexdigest()[:16]
                    if seen_hashes.get(cid) == img_hash:
                        continue
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
                    img.save(cam_dir / f'{ts_str}.jpg', quality=92)

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
                        'filename':           f'{cid}/{ts_str}.jpg',
                    })
                    saved_now  += 1
                    total_saved += 1

                mf.flush()

                print(
                    f'[{now.strftime("%H:%M")} SGT | {phase:9s}] '
                    f'cycle={cycle:4d}  saved={saved_now}/8  '
                    f'total={total_saved:5d}  wx={weather}'
                )

                elapsed    = (datetime.datetime.now(SGT) - loop_start).total_seconds()
                await asyncio.sleep(max(0, INTERVAL - elapsed))

    print(f'\nDone. {total_saved} frames  →  {night_dir}')


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except RuntimeError:
        # Jupyter kernel already has a running event loop
        import nest_asyncio
        nest_asyncio.apply()
        asyncio.get_event_loop().run_until_complete(main())
