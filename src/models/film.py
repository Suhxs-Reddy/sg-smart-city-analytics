"""
FiLM — Feature-wise Linear Modulation

Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer" (AAAI 2018)

Modulates intermediate feature maps of a convolutional backbone by applying
a learned affine transformation conditioned on an external signal:

    h_out = γ ⊙ h_in + β

where γ (scale) and β (shift) are predicted by a conditioning network.

Novel application: We condition on environmental metadata (weather, time-of-day,
camera identity) to adapt detection features to deployment conditions. No prior
work applies FiLM to traffic detection with real-time environmental conditioning.
"""

import torch
import torch.nn as nn


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation for convolutional feature maps.

    Applies channel-wise affine transformation:
        output[b, c, h, w] = gamma[b, c] * input[b, c, h, w] + beta[b, c]

    Args:
        num_channels: Number of channels in the feature map to modulate.
    """

    def __init__(self, num_channels: int):
        super().__init__()
        self.num_channels = num_channels

        # Initialize gamma=1, beta=0 (identity transform) so the model
        # starts equivalent to an unconditioned backbone
        self.default_gamma = nn.Parameter(torch.ones(1, num_channels), requires_grad=False)
        self.default_beta = nn.Parameter(torch.zeros(1, num_channels), requires_grad=False)

    def forward(
        self,
        features: torch.Tensor,
        gamma: torch.Tensor | None = None,
        beta: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply FiLM conditioning to a feature map.

        Args:
            features: (B, C, H, W) convolutional feature map.
            gamma: (B, C) scale factors from the context encoder.
            beta: (B, C) shift factors from the context encoder.

        Returns:
            Modulated features: (B, C, H, W).
        """
        if gamma is None:
            gamma = self.default_gamma.expand(features.size(0), -1)
        if beta is None:
            beta = self.default_beta.expand(features.size(0), -1)

        # Reshape for broadcasting: (B, C) -> (B, C, 1, 1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)

        return gamma * features + beta


class FiLMGenerator(nn.Module):
    """Generates (gamma, beta) pairs for multiple FiLM layers from a context vector.

    Takes a context embedding and produces per-layer affine parameters.
    Each layer gets its own (gamma, beta) pair via dedicated linear projections.

    Args:
        context_dim: Dimension of the input context embedding.
        channel_dims: List of channel counts for each FiLM layer.
                     e.g., [128, 256, 512] for 3 backbone stages.
    """

    def __init__(self, context_dim: int, channel_dims: list[int]):
        super().__init__()

        self.num_layers = len(channel_dims)

        # Separate projection heads for each backbone stage
        self.gamma_projectors = nn.ModuleList([nn.Linear(context_dim, dim) for dim in channel_dims])
        self.beta_projectors = nn.ModuleList([nn.Linear(context_dim, dim) for dim in channel_dims])

        # Initialize close to identity: gamma ~ 1, beta ~ 0
        for proj in self.gamma_projectors:
            nn.init.zeros_(proj.weight)
            nn.init.ones_(proj.bias)
        for proj in self.beta_projectors:
            nn.init.zeros_(proj.weight)
            nn.init.zeros_(proj.bias)

    def forward(self, context: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Generate FiLM parameters for all backbone stages.

        Args:
            context: (B, context_dim) environmental context embedding.

        Returns:
            List of (gamma, beta) tuples, one per backbone stage.
            Each gamma/beta has shape (B, C_i) where C_i is the channel dim.
        """
        film_params = []
        for gamma_proj, beta_proj in zip(self.gamma_projectors, self.beta_projectors, strict=True):
            gamma = gamma_proj(context)  # (B, C_i)
            beta = beta_proj(context)  # (B, C_i)
            film_params.append((gamma, beta))
        return film_params
