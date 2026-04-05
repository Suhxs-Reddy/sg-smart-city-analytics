"""
FiLM — Feature-wise Linear Modulation (Upgraded)

Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer" (AAAI 2018)

Extended for CATI (Context-Aware Traffic Intelligence) with:
    1. AdaptiveFiLMLayer — residual gating that learns when to apply conditioning
    2. AttentionFiLMLayer — CBAM post-attention for selective amplification
    3. SpectralNorm projections — stabilize gamma/beta during training
    4. Multi-scale aware generation — separate projection heads per feature scale

Novel contribution: No prior traffic detection system combines FiLM conditioning
with adaptive gating and attention. This lets the model learn that, e.g., edge
detection channels don't need weather adaptation but texture channels do.

In Singapore context:
    - Clear day, 1080p camera: gate α ≈ 0 → mostly vanilla YOLO
    - Heavy rain, night, 240p camera: gate α ≈ 1 → full FiLM conditioning
    - Per-channel attention: road surface channels get more rain adaptation
      than sky/building channels
"""

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

from src.models.attention import CBAM, AdaptiveGate


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation for convolutional feature maps.

    Applies channel-wise affine transformation:
        output[b, c, h, w] = gamma[b, c] * input[b, c, h, w] + beta[b, c]

    Identity-initialized: γ=1, β=0 at start, so the model begins
    equivalent to an unconditioned backbone.

    Args:
        num_channels: Number of channels in the feature map to modulate.
    """

    def __init__(self, num_channels: int):
        super().__init__()
        self.num_channels = num_channels

        # Buffers (non-trainable defaults) for when no context is provided
        self.register_buffer("default_gamma", torch.ones(1, num_channels))
        self.register_buffer("default_beta", torch.zeros(1, num_channels))

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

        # Reshape for broadcasting: (B, C) → (B, C, 1, 1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)

        return gamma * features + beta


class AdaptiveFiLMLayer(nn.Module):
    """FiLM with learnable residual gating and optional CBAM post-attention.

    Architecture:
        FiLM(x, γ, β) → CBAM → αGate(x_original, x_conditioned) → output

    The gate α ∈ [0,1] is context-dependent: it uses the same context
    embedding to decide how strongly to apply conditioning. This means
    the model can learn that conditioning helps more in adverse conditions.

    Args:
        num_channels: Number of feature channels.
        context_dim: Context embedding dimension (for context-dependent gating).
        use_attention: Whether to apply CBAM after FiLM.
        se_reduction: Squeeze-excitation reduction ratio.
    """

    def __init__(
        self,
        num_channels: int,
        context_dim: int = 64,
        use_attention: bool = True,
        se_reduction: int = 16,
    ):
        super().__init__()

        self.film = FiLMLayer(num_channels)
        self.gate = AdaptiveGate(num_channels, context_dim=context_dim)
        self.attention = CBAM(num_channels, reduction=se_reduction) if use_attention else None

    def forward(
        self,
        features: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """Apply gated FiLM conditioning with optional attention.

        Args:
            features: (B, C, H, W) original backbone features.
            gamma: (B, C) scale factors.
            beta: (B, C) shift factors.
            context: (B, context_dim) context embedding for gating.

        Returns:
            Conditionally modulated features (B, C, H, W).
        """
        # Apply FiLM: γ ⊙ x + β
        conditioned = self.film(features, gamma, beta)

        # Apply channel + spatial attention if enabled
        if self.attention is not None:
            conditioned = self.attention(conditioned)

        # Adaptive gating: α * conditioned + (1-α) * original
        return self.gate(features, conditioned, context)


class FiLMGenerator(nn.Module):
    """Generates (gamma, beta) pairs for multiple FiLM layers from a context vector.

    Uses spectrally-normalized projections for training stability —
    prevents gamma/beta from exploding during early training when the
    context encoder outputs are still noisy.

    Each backbone stage gets its own projection head, allowing the model
    to learn different conditioning strategies at different scales:
        - P3 (high-res): fine-grained weather effects on textures
        - P4 (mid-res): vehicle shape adaptation to conditions
        - P5 (low-res): scene-level contextual priors

    Args:
        context_dim: Dimension of the input context embedding.
        channel_dims: List of channel counts for each FiLM layer.
        use_spectral_norm: Apply spectral normalization to projections.
    """

    def __init__(
        self,
        context_dim: int,
        channel_dims: list[int],
        use_spectral_norm: bool = True,
    ):
        super().__init__()

        self.num_layers = len(channel_dims)

        # Shared context transform (adds nonlinearity before per-stage projection)
        self.context_transform = nn.Sequential(
            nn.Linear(context_dim, context_dim * 2),
            nn.GELU(),
            nn.LayerNorm(context_dim * 2),
            nn.Linear(context_dim * 2, context_dim),
            nn.GELU(),
        )

        # Per-stage projection heads
        def make_proj(out_dim: int) -> nn.Linear:
            layer = nn.Linear(context_dim, out_dim)
            if use_spectral_norm:
                layer = spectral_norm(layer)
            return layer

        self.gamma_projectors = nn.ModuleList([make_proj(dim) for dim in channel_dims])
        self.beta_projectors = nn.ModuleList([make_proj(dim) for dim in channel_dims])

        # Identity initialization: γ → 1, β → 0
        self._init_identity()

    def _init_identity(self):
        """Initialize so that output is (γ≈1, β≈0) for any context input.

        Spectral norm computes weight = weight_orig / σ where σ = spectral_norm(weight_orig).
        Setting weight_orig to exactly zeros makes σ = 0 → NaN (0/0).
        We use a very small weight instead so σ is tiny but non-zero, making
        the effective weight W/σ ≈ 0 in practice while remaining numerically stable.
        The bias term then dominates and gives γ ≈ 1, β ≈ 0 as desired.
        """
        for proj in self.gamma_projectors:
            if hasattr(proj, "weight_orig"):
                nn.init.normal_(proj.weight_orig, std=1e-4)
            else:
                nn.init.normal_(proj.weight, std=1e-4)
            nn.init.ones_(proj.bias)

        for proj in self.beta_projectors:
            if hasattr(proj, "weight_orig"):
                nn.init.normal_(proj.weight_orig, std=1e-4)
            else:
                nn.init.normal_(proj.weight, std=1e-4)
            nn.init.zeros_(proj.bias)

    def forward(self, context: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Generate FiLM parameters for all backbone stages.

        Args:
            context: (B, context_dim) environmental context embedding.

        Returns:
            List of (gamma, beta) tuples, one per backbone stage.
            Each gamma/beta has shape (B, C_i) where C_i is the channel dim.
        """
        # Shared nonlinear transform
        h = self.context_transform(context)

        film_params = []
        for gamma_proj, beta_proj in zip(self.gamma_projectors, self.beta_projectors, strict=True):
            gamma = gamma_proj(h)  # (B, C_i)
            beta = beta_proj(h)  # (B, C_i)
            film_params.append((gamma, beta))
        return film_params
