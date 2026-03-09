"""
Singapore Smart City — Dataset Formatter

Formats collected data into a Kaggle-publishable dataset:
1. Stratified train/val/test splits (by camera, time, weather)
2. YOLO-format labels from auto-detection
3. Metadata CSV/Parquet with multi-modal features
4. Statistics and visualization generation
5. Dataset card generation

Output structure:
    sg_traffic_dataset/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── labels/
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── metadata/
    │   ├── train.parquet
    │   ├── val.parquet
    │   └── test.parquet
    ├── statistics/
    │   ├── class_distribution.json
    │   ├── camera_stats.json
    │   └── weather_distribution.json
    └── DATASET_CARD.md
"""

import json
import logging
import shutil
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class DatasetFormatter:
    """Formats collected data into a Kaggle-ready dataset."""

    def __init__(
        self,
        raw_data_dir: str,
        output_dir: str,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ):
        """
        Args:
            raw_data_dir: Directory with collected data (from collector).
            output_dir: Directory for the formatted dataset.
            train_ratio: Fraction for training split.
            val_ratio: Fraction for validation split.
            test_ratio: Fraction for test split.
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

        self.raw_dir = Path(raw_data_dir)
        self.output_dir = Path(output_dir)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

    def _load_metadata(self) -> pd.DataFrame:
        """Load all metadata JSONL files into a single DataFrame."""
        records = []
        jsonl_files = list(self.raw_dir.rglob("metadata.jsonl"))

        logger.info(f"Loading metadata from {len(jsonl_files)} JSONL files")

        for jsonl_path in jsonl_files:
            with open(jsonl_path) as f:
                for line in f:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        if not records:
            logger.warning("No metadata records found")
            return pd.DataFrame()

        df = pd.DataFrame(records)
        logger.info(f"Loaded {len(df)} metadata records")
        return df

    def _stratified_split(self, df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Split data with stratification by camera and time of day.

        Ensures each camera appears in train/val/test proportionally,
        and different times of day are represented in each split.
        """
        # Add time features for stratification
        df = df.copy()
        df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour
        df["time_bucket"] = pd.cut(
            df["hour"], bins=[0, 6, 12, 18, 24],
            labels=["night", "morning", "afternoon", "evening"],
        )

        # Create stratification key
        df["strat_key"] = df["camera_id"] + "_" + df["time_bucket"].astype(str)

        # Split per stratification group
        train_dfs, val_dfs, test_dfs = [], [], []

        for _key, group in df.groupby("strat_key"):
            n = len(group)
            if n < 3:
                train_dfs.append(group)
                continue

            # Shuffle within group
            group = group.sample(frac=1, random_state=42)

            n_train = max(1, int(n * self.train_ratio))
            n_val = max(1, int(n * self.val_ratio))

            train_dfs.append(group.iloc[:n_train])
            val_dfs.append(group.iloc[n_train:n_train + n_val])
            test_dfs.append(group.iloc[n_train + n_val:])

        splits = {
            "train": pd.concat(train_dfs, ignore_index=True) if train_dfs else pd.DataFrame(),
            "val": pd.concat(val_dfs, ignore_index=True) if val_dfs else pd.DataFrame(),
            "test": pd.concat(test_dfs, ignore_index=True) if test_dfs else pd.DataFrame(),
        }

        # Drop helper columns
        for split_name in splits:
            if not splits[split_name].empty:
                splits[split_name] = splits[split_name].drop(
                    columns=["hour", "time_bucket", "strat_key"],
                    errors="ignore",
                )

        logger.info(
            f"Split: train={len(splits['train'])}, "
            f"val={len(splits['val'])}, "
            f"test={len(splits['test'])}"
        )

        return splits

    def _copy_images(self, df: pd.DataFrame, split_name: str):
        """Copy images to the split directory."""
        img_dir = self.output_dir / "images" / split_name
        img_dir.mkdir(parents=True, exist_ok=True)

        copied = 0
        for _, row in df.iterrows():
            src = Path(row["image_path"])
            if src.exists():
                # Name: {camera_id}_{timestamp}.jpg for uniqueness
                ts_clean = row.get("timestamp", "").replace(":", "-").replace("T", "_")
                dst_name = f"{row['camera_id']}_{ts_clean}.jpg"
                dst = img_dir / dst_name
                shutil.copy2(src, dst)
                copied += 1

        logger.info(f"Copied {copied}/{len(df)} images to {split_name}/")

    def _copy_labels(self, df: pd.DataFrame, split_name: str, labels_dir: Path):
        """Copy YOLO label files to the split directory."""
        out_dir = self.output_dir / "labels" / split_name
        out_dir.mkdir(parents=True, exist_ok=True)

        copied = 0
        for _, row in df.iterrows():
            img_path = Path(row["image_path"])
            label_file = labels_dir / f"{img_path.stem}.txt"

            if label_file.exists():
                ts_clean = row.get("timestamp", "").replace(":", "-").replace("T", "_")
                dst_name = f"{row['camera_id']}_{ts_clean}.txt"
                shutil.copy2(label_file, out_dir / dst_name)
                copied += 1

        logger.info(f"Copied {copied} labels to {split_name}/")

    def _compute_statistics(self, splits: dict) -> dict:
        """Compute dataset statistics for the dataset card."""
        all_data = pd.concat(splits.values(), ignore_index=True)

        stats = {
            "total_images": len(all_data),
            "split_sizes": {k: len(v) for k, v in splits.items()},
            "num_cameras": int(all_data["camera_id"].nunique()),
            "cameras": sorted(all_data["camera_id"].unique().tolist()),
            "date_range": {
                "start": str(all_data["timestamp"].min()),
                "end": str(all_data["timestamp"].max()),
            },
        }

        # Resolution distribution
        if "image_width" in all_data.columns:
            res_dist = (
                all_data.groupby(["image_width", "image_height"])
                .size()
                .reset_index(name="count")
            )
            stats["resolution_distribution"] = res_dist.to_dict(orient="records")

        # Weather distribution
        if "weather_condition" in all_data.columns:
            weather_dist = all_data["weather_condition"].value_counts().to_dict()
            stats["weather_distribution"] = weather_dist

        # Temperature stats
        if "temperature_celsius" in all_data.columns:
            temp = all_data["temperature_celsius"].dropna()
            if len(temp) > 0:
                stats["temperature"] = {
                    "mean": round(float(temp.mean()), 1),
                    "min": round(float(temp.min()), 1),
                    "max": round(float(temp.max()), 1),
                }

        return stats

    def format_dataset(
        self,
        labels_dir: str | None = None,
        deduplicate: bool = True,
    ) -> dict:
        """Format the full dataset.

        Args:
            labels_dir: Directory containing YOLO .txt label files.
            deduplicate: Remove duplicate frames (same hash).

        Returns:
            Dataset statistics dict.
        """
        logger.info("Starting dataset formatting...")

        # Load metadata
        df = self._load_metadata()
        if df.empty:
            logger.error("No data to format")
            return {}

        # Deduplicate by image hash
        if deduplicate and "image_hash_sha256" in df.columns:
            before = len(df)
            df = df.drop_duplicates(subset=["image_hash_sha256"])
            logger.info(f"Deduplication: {before} → {len(df)} images")

        # Split
        splits = self._stratified_split(df)

        # Copy images
        for split_name, split_df in splits.items():
            if not split_df.empty:
                self._copy_images(split_df, split_name)

        # Copy labels if available
        if labels_dir:
            labels_path = Path(labels_dir)
            for split_name, split_df in splits.items():
                if not split_df.empty:
                    self._copy_labels(split_df, split_name, labels_path)

        # Save metadata as Parquet
        meta_dir = self.output_dir / "metadata"
        meta_dir.mkdir(parents=True, exist_ok=True)
        for split_name, split_df in splits.items():
            if not split_df.empty:
                split_df.to_parquet(
                    meta_dir / f"{split_name}.parquet",
                    index=False,
                )

        # Compute and save statistics
        stats = self._compute_statistics(splits)
        stats_dir = self.output_dir / "statistics"
        stats_dir.mkdir(parents=True, exist_ok=True)
        with open(stats_dir / "dataset_stats.json", "w") as f:
            json.dump(stats, f, indent=2, default=str)

        # Generate dataset card
        self._generate_dataset_card(stats)

        logger.info(f"Dataset formatted at {self.output_dir}")
        return stats

    def _generate_dataset_card(self, stats: dict):
        """Generate a Kaggle-compatible dataset card."""
        card = f"""# Singapore Smart City Traffic — Multi-Modal Dataset

## Overview
Temporally continuous traffic camera images from Singapore's {stats.get('num_cameras', 90)} government LTA cameras,
enriched with weather, taxi availability, and air quality metadata.

## Key Features
- **Multi-modal**: Every frame paired with weather, temperature, PM2.5, taxi density
- **Temporally continuous**: Sequential frames enabling temporal analysis
- **Multi-camera**: Island-wide coverage across Singapore's expressway network
- **Pre-labeled**: YOLO-format annotations for 6 traffic classes

## Statistics
- **Total images**: {stats.get('total_images', 'N/A'):,}
- **Cameras**: {stats.get('num_cameras', 'N/A')}
- **Splits**: Train {stats['split_sizes'].get('train', 0):,} / Val {stats['split_sizes'].get('val', 0):,} / Test {stats['split_sizes'].get('test', 0):,}
- **Date range**: {stats.get('date_range', {}).get('start', 'N/A')} to {stats.get('date_range', {}).get('end', 'N/A')}

## Classes
| ID | Class | Description |
|---|---|---|
| 0 | car | Passenger vehicles |
| 1 | motorcycle | Two-wheelers |
| 2 | bus | Public and private buses |
| 3 | truck | Commercial vehicles |
| 4 | person | Pedestrians |
| 5 | bicycle | Cyclists |

## Directory Structure
```
sg_traffic_dataset/
├── images/{{train,val,test}}/     # Camera images (.jpg)
├── labels/{{train,val,test}}/     # YOLO annotations (.txt)
├── metadata/{{train,val,test}}.parquet  # Multi-modal metadata
├── statistics/                    # Dataset statistics
└── DATASET_CARD.md               # This file
```

## Metadata Columns
| Column | Type | Description |
|---|---|---|
| camera_id | str | LTA camera identifier |
| timestamp | datetime | Collection timestamp (SGT, UTC+8) |
| latitude/longitude | float | Camera GPS coordinates |
| image_width/height | int | Image resolution |
| weather_condition | str | Current weather (e.g. "Thundery Showers") |
| temperature_celsius | float | Air temperature at nearest station |
| pm25_reading | int | PM2.5 air quality index for camera region |
| taxi_count_nearby_5km | int | Available taxis within 5km radius |
| image_hash_sha256 | str | Content hash for deduplication |

## Data Source
All data collected from Singapore's open government APIs:
- Traffic cameras: `data.gov.sg/v1/transport/traffic-images`
- Weather: `data.gov.sg/v1/environment/24-hour-weather-forecast`
- Temperature: `data.gov.sg/v1/environment/air-temperature`
- PM2.5: `data.gov.sg/v1/environment/pm25`
- Taxi: `data.gov.sg/v1/transport/taxi-availability`

## License
Data sourced from Singapore Open Data License: https://data.gov.sg/open-data-licence

## Citation
If you use this dataset, please cite:
```
@dataset{{sg_smart_city_traffic,
  title={{Singapore Smart City Traffic — Multi-Modal Dataset}},
  year={{2026}},
  url={{https://kaggle.com/datasets/sg-smart-city-traffic}},
}}
```
"""
        card_path = self.output_dir / "DATASET_CARD.md"
        card_path.write_text(card)
        logger.info(f"Dataset card written to {card_path}")
