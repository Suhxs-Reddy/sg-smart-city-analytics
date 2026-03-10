"""
Singapore Smart City - Level 3 Predictive (Production)
Physics-Informed Neural ODE Spatio-Temporal Graph Neural Network (PI-NODE-STGNN)

This module defines the Continuous-Time graph neural network designed to forecast
traffic using fluid dynamics equations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GATConv
    from torchdiffeq import odeint
except ImportError:
    pass

class TrafficODEFunc(nn.Module):
    """
    Parameterizes the derivative dz/dt of the traffic state.
    """
    def __init__(self, hidden_dim: int):
        super(TrafficODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.Tanh(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Computes the derivative of z at time t.
        """
        return self.net(z)

class PINodeSTGNN(nn.Module):
    """
    Combines a Graph Attention Network (GAT) for spatial processing with a 
    Neural ODE solver for continuous temporal rollouts.
    """
    def __init__(self, num_node_features: int = 12, hidden_dim: int = 64):
        super(PINodeSTGNN, self).__init__()
        # Spatial Processing
        self.gat1 = GATConv(num_node_features, hidden_dim, heads=4, concat=False)
        self.gat2 = GATConv(hidden_dim, hidden_dim, heads=4, concat=False)

        # Continuous Temporal Processing
        self.ode_func = TrafficODEFunc(hidden_dim)

        # Decoding Head
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x_t0: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor, eval_times: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_t0: Initial node features at time t=0 [batch_size, num_nodes, features]
            edge_index: PyG edge connectivity
            edge_weight: Edge features
            eval_times: Continuous 1D tensor of future times to predict
            
        Returns:
            predictions [batch_size, num_nodes, len(eval_times)]
        """
        batch_size, num_nodes, features = x_t0.shape

        # 1. Spatial Embedding (Find initial hidden state z0)
        x_flat = x_t0.view(-1, features)
        z0 = F.relu(self.gat1(x_flat, edge_index, edge_weight))
        z0 = F.relu(self.gat2(z0, edge_index, edge_weight)) # [batch*nodes, hidden_dim]

        # 2. Continuous Temporal Evolution via ODE Solver
        # Solves: z(t) = z(0) + int_0^t f(z(s), s) ds
        zt = odeint(self.ode_func, z0, eval_times, method='rk4') # [len(eval_times), batch*nodes, hidden_dim]

        # 3. Decode states
        predictions = self.fc(zt).squeeze(-1) # [len(eval_times), batch*nodes]
        predictions = predictions.view(len(eval_times), batch_size, num_nodes)

        return predictions.permute(1, 2, 0)
