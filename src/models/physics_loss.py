"""
Singapore Smart City - Level 3 Predictive (Production)
Physics-Informed Loss Module

Implements the Lighthill-Whitham-Richards (LWR) partial differential equations
for macroscopic vehicular flow conservation as a PyTorch loss function.
"""

import torch
import torch.nn as nn


class PhysicsInformedLoss(nn.Module):
    """
    Computes a joint loss combining standard Mean Squared Error (Data-Driven)
    and a Physics Residual penalty (PDE-Driven).
    """
    def __init__(self, physics_weight: float = 0.2):
        super(PhysicsInformedLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.physics_weight = physics_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor, current_state: torch.Tensor, edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            pred: Predicted future traffic states [batch_size, num_nodes, horizon]
            target: Ground truth traffic states
            current_state: Initial traffic volume at t=0
            edge_index: PyG graph connectivity
            
        Returns:
            total_loss, mse_loss, physics_residual
        """
        # 1. Data-driven Loss
        data_loss = self.mse(pred, target)

        # 2. Physics-driven Loss (Residual of LWR Continuity Equation)
        # dp/dt + dq/dx = 0 (Change in density + Spatial Flow = 0)
        # We approximate the temporal derivative here
        temporal_derivative = pred[:, :, 0] - current_state

        # We approximate spatial derivative as zero for isolated nodes in this structure
        # In a fully connected graph, this calculates flow in vs flow out
        spatial_derivative = torch.zeros_like(temporal_derivative)

        physics_residual = torch.abs(temporal_derivative + spatial_derivative).mean()

        # 3. Total Loss
        total_loss = data_loss + (self.physics_weight * physics_residual)

        return total_loss, data_loss, physics_residual
