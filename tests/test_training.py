"""
Unit tests for CATI Training Pipeline

Verifies:
    1. CATIDataset metadata parsing (no .pt file → features=None)
    2. CATIDataset with cached feature tensors
    3. cati_collate function
    4. CATITrainer initialization and optimizer groups
    5. Training loop (forward/backward/step) with synthetic fallback
    6. Checkpoint save/load logic
"""

import json

import pytest

torch = pytest.importorskip("torch", reason="PyTorch required for training tests")

from src.models.cati_detector import CATIConfig  # noqa: E402
from src.training.train_cati import CATIDataset, CATITrainer, cati_collate  # noqa: E402


@pytest.fixture
def dummy_data_dir(tmp_path):
    """Create a temporary directory with dummy CATI metadata JSONs (no .pt files)."""
    data_dir = tmp_path / "train"
    data_dir.mkdir()

    for i in range(5):
        meta = {
            "camera_id": str(100 + i),
            "camera_idx": i,
            "timestamp": f"2026-03-09T{10 + i}:00:00Z",
            "weather_condition": "clear",
            "temperature_celsius": 28.0 + i,
            "pm25_reading": 15.0,
            "hour": 10.0 + i,
            "image_width": 1920,
            "image_height": 1080,
            "camera_latitude": 1.35,
            "camera_longitude": 103.8,
        }
        with open(data_dir / f"sample_{i}.json", "w") as f:
            json.dump(meta, f)

    return data_dir


@pytest.fixture
def dummy_data_dir_with_features(tmp_path):
    """Metadata JSONs + synthetic cached .pt feature files."""
    data_dir = tmp_path / "train"
    data_dir.mkdir()

    channel_dims = [64, 128, 256]  # Minimal backbone channels for tests

    for i in range(4):
        meta = {
            "camera_id": str(100 + i),
            "camera_idx": i % 90,
            "timestamp": f"2026-03-09T{10 + i}:00:00Z",
            "weather_condition": "heavy_rain" if i % 2 == 0 else "clear",
            "temperature_celsius": 28.0,
            "pm25_reading": 30.0,
            "hour": 8.0 + i,
            "image_width": 1920,
            "image_height": 1080,
            "camera_latitude": 1.35 + i * 0.01,
            "camera_longitude": 103.8,
        }
        stem = f"sample_{i}"
        with open(data_dir / f"{stem}.json", "w") as f:
            json.dump(meta, f)

        # Save FP16 backbone features matching the expected format
        features = [
            torch.randn(1, c, 80 // (2**s), 80 // (2**s)).half()
            for s, c in enumerate(channel_dims)
        ]
        torch.save(features, data_dir / f"{stem}.pt")

    return data_dir


class TestCATIDataset:
    def test_dataset_loading_no_features(self, dummy_data_dir):
        ds = CATIDataset(str(dummy_data_dir))
        assert len(ds) == 5

        ctx, strat, features = ds[0]
        assert "weather_id" in ctx
        assert "camera_id" in ctx
        assert ctx["camera_id"] == 0
        assert strat["weather"] == "clear"
        assert features is None  # No .pt file → None

    def test_dataset_loading_with_features(self, dummy_data_dir_with_features):
        ds = CATIDataset(str(dummy_data_dir_with_features))
        assert len(ds) == 4

        _ctx, _strat, features = ds[0]
        assert features is not None
        assert len(features) == 3  # P3, P4, P5
        # Loaded as FP32 (converted from FP16 on disk)
        assert features[0].dtype == torch.float32

    def test_stratification(self, dummy_data_dir):
        ds = CATIDataset(str(dummy_data_dir))
        _, strat, _ = ds[0]
        # 10am is 'midday' category
        assert strat["time_of_day"] == "midday"

    def test_max_samples(self, dummy_data_dir):
        ds = CATIDataset(str(dummy_data_dir), max_samples=3)
        assert len(ds) == 3


class TestCATICollate:
    def test_collate_no_features(self, dummy_data_dir):
        ds = CATIDataset(str(dummy_data_dir))
        from torch.utils.data import DataLoader

        dl = DataLoader(ds, batch_size=3, collate_fn=cati_collate)
        ctx, _strat, features = next(iter(dl))

        assert ctx["weather_id"].shape == (3,)
        assert features is None  # All None → None

    def test_collate_with_features(self, dummy_data_dir_with_features):
        ds = CATIDataset(str(dummy_data_dir_with_features))
        from torch.utils.data import DataLoader

        dl = DataLoader(ds, batch_size=2, collate_fn=cati_collate)
        _ctx, _strat, features = next(iter(dl))

        assert features is not None
        assert len(features) == 3  # P3, P4, P5
        assert features[0].shape[0] == 2  # Batch size 2


class TestCATITrainer:
    def test_trainer_init(self):
        config = CATIConfig(num_cameras=90, context_dim=64)
        trainer = CATITrainer(config, learning_rate=1e-3, device="cpu")

        assert len(trainer.optimizer.param_groups) == 4  # context_encoder, film_gen, film_layers, prediction_head
        assert trainer.optimizer.param_groups[0]["lr"] == 1e-3  # context_encoder
        assert trainer.optimizer.param_groups[2]["lr"] == 5e-4  # film_layers (0.5×)

    def test_train_step_synthetic_fallback(self, dummy_data_dir):
        """Without .pt files the trainer should fall back to synthetic features."""
        config = CATIConfig(num_cameras=90, context_dim=32, backbone_channels=[64])
        trainer = CATITrainer(config, learning_rate=1e-3, device="cpu", use_amp=False)

        from torch.utils.data import DataLoader

        ds = CATIDataset(str(dummy_data_dir))
        dl = DataLoader(ds, batch_size=2, collate_fn=cati_collate)

        metrics = trainer.train_epoch(dl)
        assert "train_loss" in metrics
        assert metrics["train_loss"] >= 0
        assert trainer.current_epoch == 1
        assert metrics["synthetic_batches"] > 0  # Confirms synthetic fallback was used

    def test_train_step_real_features(self, dummy_data_dir_with_features):
        """With cached .pt files the trainer should use real backbone features."""
        config = CATIConfig(num_cameras=90, context_dim=32, backbone_channels=[64, 128, 256])
        trainer = CATITrainer(config, learning_rate=1e-3, device="cpu", use_amp=False)

        from torch.utils.data import DataLoader

        ds = CATIDataset(str(dummy_data_dir_with_features))
        dl = DataLoader(ds, batch_size=2, collate_fn=cati_collate)

        metrics = trainer.train_epoch(dl)
        assert "train_loss" in metrics
        assert metrics["synthetic_batches"] == 0  # No synthetic fallback

    def test_checkpoint_logic(self, dummy_data_dir, tmp_path):
        config = CATIConfig(num_cameras=90, context_dim=32)
        trainer = CATITrainer(config, device="cpu")

        checkpoint_path = tmp_path / "latest.pt"
        trainer.save_checkpoint(str(checkpoint_path), epoch=5, val_loss=0.123)

        new_trainer = CATITrainer(config, device="cpu")
        epoch = new_trainer.load_checkpoint(str(checkpoint_path))

        assert epoch == 5
        assert new_trainer.cati is not trainer.cati
        w1 = next(iter(trainer.cati.parameters()))
        w2 = next(iter(new_trainer.cati.parameters()))
        torch.testing.assert_close(w1, w2)
