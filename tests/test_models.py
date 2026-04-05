"""
Unit tests for the CATI (Context-Aware Traffic Intelligence) architecture.
"""

import pytest

torch = pytest.importorskip("torch", reason="PyTorch required for CATI model tests")

from src.models.cati_detector import CATIConfig, CATIDetector  # noqa: E402
from src.models.context_encoder import (  # noqa: E402
    RESOLUTION_BUCKETS,
    WEATHER_CONDITIONS,
    ContextEncoder,
)
from src.models.film import FiLMGenerator, FiLMLayer  # noqa: E402


class TestFiLMLayer:
    def test_identity_initialization(self):
        layer = FiLMLayer(num_channels=64)
        features = torch.randn(2, 64, 16, 16)
        output = layer(features)
        torch.testing.assert_close(output, features, atol=1e-6, rtol=1e-6)

    def test_scaling(self):
        layer = FiLMLayer(num_channels=4)
        features = torch.ones(1, 4, 2, 2)
        gamma = torch.tensor([[2.0, 3.0, 0.5, 1.0]])
        beta = torch.zeros(1, 4)
        output = layer(features, gamma, beta)
        assert output[0, 0, 0, 0].item() == pytest.approx(2.0)
        assert output[0, 1, 0, 0].item() == pytest.approx(3.0)

    def test_shifting(self):
        layer = FiLMLayer(num_channels=4)
        features = torch.zeros(1, 4, 2, 2)
        gamma = torch.ones(1, 4)
        beta = torch.tensor([[10.0, -5.0, 0.0, 1.0]])
        output = layer(features, gamma, beta)
        assert output[0, 0, 0, 0].item() == pytest.approx(10.0)
        assert output[0, 1, 0, 0].item() == pytest.approx(-5.0)

    def test_batch_independence(self):
        layer = FiLMLayer(num_channels=4)
        features = torch.ones(2, 4, 2, 2)
        gamma = torch.tensor([[2.0, 2.0, 2.0, 2.0], [0.5, 0.5, 0.5, 0.5]])
        beta = torch.zeros(2, 4)
        output = layer(features, gamma, beta)
        assert output[0, 0, 0, 0].item() == pytest.approx(2.0)
        assert output[1, 0, 0, 0].item() == pytest.approx(0.5)

    def test_output_shape(self):
        layer = FiLMLayer(num_channels=128)
        features = torch.randn(4, 128, 32, 32)
        output = layer(features)
        assert output.shape == (4, 128, 32, 32)


class TestFiLMGenerator:
    def test_output_count(self):
        gen = FiLMGenerator(context_dim=64, channel_dims=[128, 256, 512])
        context = torch.randn(2, 64)
        params = gen(context)
        assert len(params) == 3

    def test_output_shapes(self):
        dims = [128, 256, 512]
        gen = FiLMGenerator(context_dim=64, channel_dims=dims)
        context = torch.randn(4, 64)
        params = gen(context)
        for (gamma, beta), dim in zip(params, dims, strict=True):
            assert gamma.shape == (4, dim)
            assert beta.shape == (4, dim)

    def test_identity_initialization(self):
        gen = FiLMGenerator(context_dim=32, channel_dims=[64])
        context = torch.zeros(1, 32)
        params = gen(context)
        gamma, beta = params[0]
        assert gamma.mean().item() == pytest.approx(1.0, abs=0.01)
        assert beta.mean().item() == pytest.approx(0.0, abs=0.01)

    def test_gradient_flow(self):
        gen = FiLMGenerator(context_dim=32, channel_dims=[64])
        with torch.no_grad():
            for p in gen.parameters():
                p.add_(torch.randn_like(p) * 0.01)
        context = torch.randn(2, 32, requires_grad=True)
        params = gen(context)
        loss = sum(g.sum() + b.sum() for g, b in params)
        loss.backward()
        assert context.grad is not None
        assert context.grad.abs().sum() > 0


class TestContextEncoder:
    def test_forward_shape(self):
        encoder = ContextEncoder(num_cameras=90, context_dim=64)
        batch_size = 4
        output = encoder(
            torch.randint(0, len(WEATHER_CONDITIONS), (batch_size,)),
            torch.randn(batch_size) * 5 + 28,
            torch.randn(batch_size).abs() * 20,
            torch.rand(batch_size) * 24,
            torch.randint(0, 90, (batch_size,)),
            torch.randint(0, len(RESOLUTION_BUCKETS), (batch_size,)),
        )
        assert output.shape == (batch_size, 64)

    def test_cyclical_time_encoding(self):
        encoder = ContextEncoder()
        hours = torch.tensor([0.0, 6.0, 12.0, 18.0, 24.0])
        encoded = encoder.encode_time(hours)
        assert encoded.shape == (5, 2)
        torch.testing.assert_close(encoded[0], encoded[4], atol=1e-5, rtol=1e-5)
        norms = torch.norm(encoded, dim=-1)
        torch.testing.assert_close(norms, torch.ones(5), atol=1e-5, rtol=1e-5)

    def test_weather_to_id(self):
        assert ContextEncoder.weather_to_id("clear") == 0
        assert ContextEncoder.weather_to_id("heavy_rain") == 6
        assert ContextEncoder.weather_to_id("TOTALLY_UNKNOWN") == len(WEATHER_CONDITIONS) - 1

    def test_resolution_to_id(self):
        assert ContextEncoder.resolution_to_id(1920, 1080) == 0
        assert ContextEncoder.resolution_to_id(640, 360) == 1
        assert ContextEncoder.resolution_to_id(320, 240) == 2

    def test_different_cameras_different_embeddings(self):
        encoder = ContextEncoder(num_cameras=90, context_dim=64)
        ctx = encoder(
            torch.tensor([0, 0]),
            torch.tensor([28.0, 28.0]),
            torch.tensor([15.0, 15.0]),
            torch.tensor([12.0, 12.0]),
            torch.tensor([0, 50]),
            torch.tensor([0, 0]),
        )
        diff = (ctx[0] - ctx[1]).abs().sum()
        assert diff > 0.01


class TestCATIDetector:
    def test_initialization(self):
        config = CATIConfig(num_cameras=10, context_dim=32)
        cati = CATIDetector(config)
        assert cati is not None

    def test_parameter_count(self):
        config = CATIConfig(num_cameras=90, context_dim=64)
        cati = CATIDetector(config)
        params = cati.count_parameters()
        total = params["total_cati_overhead"]
        assert total < 150_000, f"CATI overhead {total:,} exceeds 150K params"

    def test_forward_pass(self):
        config = CATIConfig(num_cameras=10, context_dim=32, backbone_channels=[64, 128, 256])
        cati = CATIDetector(config)
        batch_size = 2
        features = [
            torch.randn(batch_size, 64, 80, 80),
            torch.randn(batch_size, 128, 40, 40),
            torch.randn(batch_size, 256, 20, 20),
        ]
        output = cati(
            features,
            torch.randint(0, 11, (batch_size,)),
            torch.randn(batch_size) * 5 + 28,
            torch.randn(batch_size).abs() * 20,
            torch.rand(batch_size) * 24,
            torch.randint(0, 10, (batch_size,)),
            torch.randint(0, 3, (batch_size,)),
        )
        assert len(output) == 3
        assert output[0].shape == (batch_size, 64, 80, 80)
        assert output[1].shape == (batch_size, 128, 40, 40)
        assert output[2].shape == (batch_size, 256, 20, 20)

    def test_gradient_flow_end_to_end(self):
        config = CATIConfig(num_cameras=10, context_dim=32, backbone_channels=[64])
        cati = CATIDetector(config)
        features = [torch.randn(2, 64, 8, 8, requires_grad=True)]
        output = cati(
            features,
            torch.randint(0, 11, (2,)),
            torch.tensor([28.0, 30.0]),
            torch.tensor([15.0, 20.0]),
            torch.tensor([12.0, 18.0]),
            torch.randint(0, 10, (2,)),
            torch.randint(0, 3, (2,)),
        )
        loss = output[0].sum()
        loss.backward()
        assert features[0].grad is not None
        assert features[0].grad.abs().sum() > 0

    def test_default_config(self):
        config = CATIConfig()
        assert config.num_cameras == 90
        assert config.num_classes == 6
        assert config.backbone_channels == [128, 256, 512]
