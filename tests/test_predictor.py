"""
Tests for the prediction models and feature engineering.

Covers:
- LSTM model forward pass shape validation
- Spatial-Temporal Graph model forward pass
- Feature engineering (cyclical encoding, normalization)
- Adjacency matrix construction
- Dataset shape validation
"""

import math

import numpy as np
import pytest
import torch

from src.analytics.predictor import (
    TrafficLSTM,
    SpatialTemporalPredictor,
    TrafficTimeSeriesDataset,
    SpatialTemporalDataset,
    PredictionTrainer,
    prepare_features,
    build_adjacency_matrix,
)


# =============================================================================
# LSTM Tests
# =============================================================================

class TestTrafficLSTM:
    def test_output_shape(self):
        model = TrafficLSTM(input_dim=6, hidden_dim=32)
        x = torch.randn(4, 30, 6)  # batch=4, window=30, features=6
        out = model(x)
        assert out.shape == (4,)    # One prediction per sample

    def test_single_sample(self):
        model = TrafficLSTM(input_dim=6, hidden_dim=32)
        x = torch.randn(1, 30, 6)
        out = model(x)
        assert out.shape == (1,)

    def test_different_window_sizes(self):
        model = TrafficLSTM(input_dim=6, hidden_dim=32)
        for window in [10, 30, 60]:
            x = torch.randn(2, window, 6)
            out = model(x)
            assert out.shape == (2,)

    def test_gradient_flow(self):
        model = TrafficLSTM(input_dim=6, hidden_dim=32)
        x = torch.randn(2, 30, 6)
        out = model(x)
        loss = out.sum()
        loss.backward()

        # Check gradients exist
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"


# =============================================================================
# Spatial-Temporal Model Tests
# =============================================================================

class TestSpatialTemporalPredictor:
    def test_output_shape(self):
        model = SpatialTemporalPredictor(
            num_features=6, hidden_dim=32,
            num_gat_layers=1, num_transformer_layers=1,
        )
        x = torch.randn(2, 10, 5, 6)     # batch=2, window=10, nodes=5, features=6
        adj = torch.ones(5, 5)             # Fully connected
        out = model(x, adj)
        assert out.shape == (2, 5)         # Prediction per node

    def test_sparse_adjacency(self):
        """Model should work with sparse graph (not all connected)."""
        model = SpatialTemporalPredictor(
            num_features=6, hidden_dim=32,
            num_gat_layers=1, num_transformer_layers=1,
        )
        x = torch.randn(2, 10, 5, 6)
        # Only self-loops + sequential connections
        adj = torch.eye(5)
        for i in range(4):
            adj[i, i+1] = 1
            adj[i+1, i] = 1

        out = model(x, adj)
        assert out.shape == (2, 5)

    def test_single_node_graph(self):
        """Edge case: single camera."""
        model = SpatialTemporalPredictor(
            num_features=6, hidden_dim=32,
            num_gat_layers=1, num_transformer_layers=1,
        )
        x = torch.randn(1, 10, 1, 6)
        adj = torch.ones(1, 1)
        out = model(x, adj)
        assert out.shape == (1, 1)

    def test_gradient_flow(self):
        model = SpatialTemporalPredictor(
            num_features=6, hidden_dim=32,
            num_gat_layers=1, num_transformer_layers=1,
        )
        x = torch.randn(2, 5, 3, 6)
        adj = torch.ones(3, 3)
        out = model(x, adj)
        loss = out.sum()
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"


# =============================================================================
# Dataset Tests
# =============================================================================

class TestTrafficTimeSeriesDataset:
    def test_length(self):
        data = np.random.randn(100, 6).astype(np.float32)
        dataset = TrafficTimeSeriesDataset(data, window_size=30, horizon=15)
        assert len(dataset) == 100 - 30 - 15 + 1  # 56

    def test_shapes(self):
        data = np.random.randn(100, 6).astype(np.float32)
        dataset = TrafficTimeSeriesDataset(data, window_size=30, horizon=15)
        x, y = dataset[0]
        assert x.shape == (30, 6)
        assert y.shape == ()  # Scalar

    def test_target_is_future_vehicle_count(self):
        """Target should be vehicle count (column 0) at future timestep."""
        data = np.arange(600).reshape(100, 6).astype(np.float32)
        dataset = TrafficTimeSeriesDataset(data, window_size=10, horizon=5)
        x, y = dataset[0]
        # y should be data[10 + 5 - 1, 0] = data[14, 0] = 14*6 = 84
        assert y.item() == 84.0


class TestSpatialTemporalDataset:
    def test_length(self):
        features = np.random.randn(100, 5, 6).astype(np.float32)
        adj = np.eye(5, dtype=np.float32)
        dataset = SpatialTemporalDataset(features, adj, window_size=20, horizon=10)
        assert len(dataset) == 100 - 20 - 10 + 1

    def test_shapes(self):
        features = np.random.randn(100, 5, 6).astype(np.float32)
        adj = np.eye(5, dtype=np.float32)
        dataset = SpatialTemporalDataset(features, adj, window_size=20, horizon=10)
        x, y, a = dataset[0]
        assert x.shape == (20, 5, 6)
        assert y.shape == (5,)
        assert a.shape == (5, 5)


# =============================================================================
# Feature Engineering Tests
# =============================================================================

class TestPrepareFeatures:
    def test_basic_output(self):
        records = [
            {"num_vehicles": 5, "temperature_celsius": 30.0,
             "pm25_reading": 15, "taxi_count_nearby_5km": 20,
             "timestamp": "2026-03-08T14:00:00"},
            {"num_vehicles": 8, "temperature_celsius": 31.0,
             "pm25_reading": 18, "taxi_count_nearby_5km": 25,
             "timestamp": "2026-03-08T15:00:00"},
        ]
        features = prepare_features(records)
        assert features.shape == (2, 6)  # 4 base + 2 cyclical

    def test_cyclical_encoding(self):
        """Hour 0 and hour 24 should have same encoding."""
        records_0 = [{"num_vehicles": 5, "temperature_celsius": 30,
                       "pm25_reading": 10, "taxi_count_nearby_5km": 10,
                       "timestamp": "2026-03-08T00:00:00"}]
        records_24 = [{"num_vehicles": 5, "temperature_celsius": 30,
                        "pm25_reading": 10, "taxi_count_nearby_5km": 10,
                        "timestamp": "2026-03-09T00:00:00"}]  # Midnight next day

        # Both should produce same sin/cos values (hour 0)
        f0 = prepare_features(records_0)
        f24 = prepare_features(records_24)
        np.testing.assert_array_almost_equal(f0[:, 4:], f24[:, 4:])

    def test_missing_values_default_to_zero(self):
        records = [{"timestamp": "2026-03-08T12:00:00"}]  # Missing everything
        features = prepare_features(records)
        assert features.shape == (1, 6)

    def test_normalization(self):
        """Output should be z-score normalized."""
        records = [
            {"num_vehicles": v, "temperature_celsius": 30,
             "pm25_reading": 10, "taxi_count_nearby_5km": 10,
             "timestamp": "2026-03-08T12:00:00"}
            for v in range(10)
        ]
        features = prepare_features(records)
        # Mean should be ~0, std ~1 per column
        assert abs(features[:, 0].mean()) < 0.1
        assert abs(features[:, 0].std() - 1.0) < 0.1


class TestAdjacencyMatrix:
    def test_self_loops(self):
        cameras = {"A": (1.3, 103.8), "B": (1.35, 103.85)}
        adj = build_adjacency_matrix(cameras, distance_threshold_km=100)
        assert adj[0, 0] == 1.0
        assert adj[1, 1] == 1.0

    def test_nearby_cameras_connected(self):
        cameras = {
            "A": (1.30, 103.80),
            "B": (1.301, 103.801),  # Very close
        }
        adj = build_adjacency_matrix(cameras, distance_threshold_km=5.0)
        assert adj[0, 1] == 1.0
        assert adj[1, 0] == 1.0

    def test_far_cameras_disconnected(self):
        cameras = {
            "A": (1.20, 103.70),
            "B": (1.45, 103.95),  # Far apart
        }
        adj = build_adjacency_matrix(cameras, distance_threshold_km=1.0)
        assert adj[0, 1] == 0.0

    def test_symmetric(self):
        cameras = {f"cam{i}": (1.3 + i*0.01, 103.8 + i*0.01) for i in range(5)}
        adj = build_adjacency_matrix(cameras, distance_threshold_km=3.0)
        np.testing.assert_array_equal(adj, adj.T)
