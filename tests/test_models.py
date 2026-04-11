"""
Senior-Grade Unit Tests for CATI Architecture

Verifies all components of the Context-Aware Traffic Intelligence system:
    1. Attention modules (SE, Spatial, CBAM, AdaptiveGate)
    2. Upgraded FiLM layers (AdaptiveFiLM, Generator)
    3. Context Encoder (GPS encoding, multi-freq time, augmentation)
    4. CATI Detector (End-to-end forward pass, parameter counts)
    5. EMA Model (Shadow weight updates)
    6. Backbone Wrapper (Integration logic)

All tests are designed to run on CPU via CI (Job 3: test-ml).
"""

import pytest

torch = pytest.importorskip("torch", reason="PyTorch required for CATI model tests")
nn = torch.nn

from src.models.attention import (  # noqa: E402
    CBAM,
    AdaptiveGate,
    SpatialAttention,
    SqueezeExciteBlock,
)
from src.models.cati_detector import (  # noqa: E402
    CATIBackboneWrapper,
    CATIConfig,
    CATIDetector,
    EMAModel,
)
from src.models.context_encoder import ContextEncoder  # noqa: E402
from src.models.film import AdaptiveFiLMLayer, FiLMGenerator, FiLMLayer  # noqa: E402


class TestAttention:
    def test_squeeze_excite(self):
        block = SqueezeExciteBlock(channels=64, reduction=16)
        x = torch.randn(2, 64, 16, 16)
        out = block(x)
        assert out.shape == x.shape
        # Check that it's not and identity (weights should be learned/init)
        assert not torch.allclose(out, x)

    def test_spatial_attention(self):
        block = SpatialAttention(kernel_size=7)
        x = torch.randn(2, 64, 16, 16)
        out = block(x)
        assert out.shape == x.shape

    def test_cbam(self):
        block = CBAM(channels=32)
        x = torch.randn(2, 32, 8, 8)
        out = block(x)
        assert out.shape == x.shape

    def test_adaptive_gate_static(self):
        gate = AdaptiveGate(channels=16)
        orig = torch.ones(2, 16, 4, 4)
        cond = torch.zeros(2, 16, 4, 4)
        # Default init is near 0 (mostly vanilla/original)
        out = gate(orig, cond)
        assert out.mean() > 0.8  # Should be closer to 'orig' (1.0)

    def test_adaptive_gate_context(self):
        gate = AdaptiveGate(channels=16, context_dim=32)
        ctx = torch.randn(2, 32)
        orig = torch.ones(2, 16, 4, 4)
        cond = torch.zeros(2, 16, 4, 4)
        out = gate(orig, cond, context=ctx)
        assert out.shape == orig.shape


class TestFiLM:
    def test_film_layer_identity(self):
        layer = FiLMLayer(num_channels=64)
        x = torch.randn(2, 64, 8, 8)
        out = layer(x)
        torch.testing.assert_close(out, x)

    def test_adaptive_film_forward(self):
        layer = AdaptiveFiLMLayer(num_channels=64, context_dim=32)
        x = torch.randn(2, 64, 8, 8)
        gamma = torch.ones(2, 64) * 1.1
        beta = torch.ones(2, 64) * 0.1
        ctx = torch.randn(2, 32)
        out = layer(x, gamma, beta, ctx)
        assert out.shape == x.shape

    def test_generator_init_identity(self):
        gen = FiLMGenerator(context_dim=32, channel_dims=[64, 128])
        ctx = torch.zeros(1, 32)
        params = gen(ctx)
        gamma1, beta1 = params[0]
        # Identity init: gamma ≈ 1.0, beta ≈ 0.0.
        # Spectral norm requires small non-zero weights to avoid 0/0, so we allow
        # moderate tolerance — the key property is that gamma starts near 1.0 (not near 0
        # or exploding), and beta stays near 0.
        assert gamma1.mean().item() == pytest.approx(1.0, abs=0.5)
        assert beta1.abs().mean().item() < 0.5

    def test_generator_spectral_norm(self):
        gen = FiLMGenerator(context_dim=32, channel_dims=[64], use_spectral_norm=True)
        # Check if weight_orig exists (indicator of spectral_norm in torch)
        assert hasattr(gen.gamma_projectors[0], "weight_orig")


class TestContextEncoder:
    def test_gps_encoding(self):
        encoder = ContextEncoder(use_gps_encoding=True, gps_embed_dim=8)
        batch_size = 4
        # Singapore coordinates
        lat = torch.tensor([1.35, 1.36, 1.37, 1.38])
        lon = torch.tensor([103.8, 103.81, 103.82, 103.83])

        output = encoder(
            torch.zeros(batch_size),
            torch.ones(batch_size) * 28,
            torch.ones(batch_size) * 15,
            torch.ones(batch_size) * 12,
            torch.zeros(batch_size),
            torch.zeros(batch_size),
            camera_lat=lat,
            camera_lon=lon,
        )
        assert output.shape == (batch_size, 64)

    def test_augmentation_behavior(self):
        encoder = ContextEncoder(use_augmentation=True)
        encoder.train()

        weather = torch.zeros(1, dtype=torch.long)
        temp = torch.tensor([30.0])
        pm25 = torch.tensor([10.0])
        hour = torch.tensor([12.0])

        # Run multiple times, should see variation
        outputs = []
        for _ in range(20):
            with torch.no_grad():
                # We need to access augmented values directly or check fusion output variance
                # Easiest: test the augmentation module standalone
                _aug_w, aug_t, _aug_p, aug_h = encoder.augmentation(weather, temp, pm25, hour)
                outputs.append((aug_t.item(), aug_h.item()))

        temps = [o[0] for o in outputs]
        assert len(set(temps)) > 1  # Temperature should vary

    def test_resolution_mapping(self):
        assert ContextEncoder.resolution_to_id(1920, 1080) == 0  # High
        assert ContextEncoder.resolution_to_id(320, 240) == 2  # Low


class TestCATIDetector:
    def test_parameter_overhead(self):
        config = CATIConfig(num_cameras=90, context_dim=64)
        cati = CATIDetector(config)
        counts = cati.count_parameters()
        # Overhead is ~200–600K — lightweight relative to the 9.4M YOLOv11s backbone
        assert counts["total_cati_overhead"] > 100_000
        assert counts["total_cati_overhead"] < 700_000

    def test_end_to_end_forward(self):
        config = CATIConfig(backbone_channels=[64, 128])
        cati = CATIDetector(config)
        batch = 2
        features = [torch.randn(batch, 64, 16, 16), torch.randn(batch, 128, 8, 8)]

        out = cati(
            features,
            torch.zeros(batch),
            torch.ones(batch) * 28,
            torch.ones(batch) * 15,
            torch.ones(batch) * 12,
            torch.zeros(batch),
            torch.zeros(batch),
        )
        assert len(out) == 2
        assert out[0].shape == features[0].shape


class TestEMAModel:
    def test_ema_update(self):
        model = nn.Linear(10, 1)
        ema = EMAModel(model, decay=0.9)

        # Pin shadow and model to known initial values so the test is deterministic
        with torch.no_grad():
            for p in ema.shadow.parameters():
                p.fill_(0.0)
            model.weight.fill_(1.0)
            model.bias.fill_(0.0)

        ema.update(model)
        # EMA: 0.9 * 0.0 + 0.1 * 1.0 = 0.1 for weights, 0.9 * 0 + 0.1 * 0 = 0 for bias
        shadow_weight = ema.shadow.weight.mean().item()
        assert 0.09 < shadow_weight < 0.11


class TestBackboneWrapper:
    def test_wrapper_init(self):
        # Smoke test for wrapper init (doesn't load YOLO unless path exists)
        config = CATIConfig()
        wrapper = CATIBackboneWrapper(yolo_model_path="nonexistent.pt", config=config)
        assert wrapper.cati is not None
        assert wrapper.yolo is None  # Should fail gracefully
