"""
Attention Modules for CATI — Channel, Spatial, and Adaptive Gating

Implements attention mechanisms used by the FiLM conditioning pipeline:
    1. Squeeze-Excitation (SE) — channel attention (Hu et al., CVPR 2018)
    2. Spatial Attention — learned spatial weighting
    3. CBAM — Combined channel + spatial (Woo et al., ECCV 2018)
    4. AdaptiveGate — learnable residual gate α ∈ [0,1]

These modules enable the CATI detector to learn *how much* environmental
conditioning to apply per-channel and per-spatial-location, rather than
blindly scaling/shifting all features equally.

Design rationale for Singapore traffic:
    - Channel attention identifies which feature channels benefit from
      weather/time conditioning (e.g., edge detectors may not need
      weather adaptation, but texture channels do)
    - Spatial attention focuses conditioning on road regions rather
      than sky/buildings
    - Adaptive gating lets the model gracefully fall back to vanilla
      YOLO when environmental signals are unreliable
"""

import torch
import torch.nn as nn


class SqueezeExciteBlock(nn.Module):
    """Squeeze-and-Excitation channel attention.

    Learns per-channel importance weights via global average pooling
    followed by a bottleneck MLP.

    Architecture:
        GAP(C,H,W) → FC(C→C/r) → ReLU → FC(C/r→C) → Sigmoid → Scale

    Args:
        channels: Number of input channels.
        reduction: Bottleneck reduction ratio (default: 16).
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 8)  # Floor at 8 to avoid degenerate bottleneck

        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel attention.

        Args:
            x: (B, C, H, W) feature map.

        Returns:
            Channel-reweighted features (B, C, H, W).
        """
        b, c, _, _ = x.shape
        # Squeeze: (B, C, H, W) → (B, C)
        w = self.squeeze(x).view(b, c)
        # Excitation: (B, C) → (B, C)
        w = self.excitation(w).view(b, c, 1, 1)
        return x * w


class SpatialAttention(nn.Module):
    """Spatial attention via channel-wise statistics.

    Computes max and mean across channels, then learns a 2D attention map.

    Architecture:
        [MaxPool(C), AvgPool(C)] → Conv(2→1, k=7) → Sigmoid → Scale

    Args:
        kernel_size: Convolution kernel size (must be odd).
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spatial attention.

        Args:
            x: (B, C, H, W) feature map.

        Returns:
            Spatially-reweighted features (B, C, H, W).
        """
        # Channel statistics: (B, 1, H, W) each
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        # Concatenate and convolve: (B, 2, H, W) → (B, 1, H, W)
        spatial_map = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * spatial_map


class CBAM(nn.Module):
    """Convolutional Block Attention Module — sequential channel + spatial attention.

    Combines SE-style channel attention with spatial attention for
    comprehensive feature refinement.

    In CATI context: applied after FiLM modulation to selectively amplify
    the most useful conditioning effects.

    Args:
        channels: Number of input channels.
        reduction: SE bottleneck reduction ratio.
        spatial_kernel: Spatial attention kernel size.
    """

    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        self.channel_attention = SqueezeExciteBlock(channels, reduction)
        self.spatial_attention = SpatialAttention(spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply sequential channel → spatial attention.

        Args:
            x: (B, C, H, W) feature map.

        Returns:
            Attended features (B, C, H, W).
        """
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class AdaptiveGate(nn.Module):
    """Learnable residual gate for conditional feature modulation.

    Produces a per-channel scalar α ∈ [0,1] that blends between
    the original features and the conditioned features:

        output = α * conditioned + (1 - α) * original

    This allows the network to gracefully fall back to vanilla features
    when environmental conditioning is unhelpful or unreliable.

    For Singapore traffic: during clear daytime (when conditioning adds
    little value), α should learn to be near 0. During heavy rain at
    night (when conditioning is critical), α should be near 1.

    Args:
        channels: Number of feature channels.
        context_dim: Dimension of the context embedding (if context-dependent).
    """

    def __init__(self, channels: int, context_dim: int | None = None):
        super().__init__()

        if context_dim is not None:
            # Context-dependent gate: α depends on what context we're conditioning on
            self.gate = nn.Sequential(
                nn.Linear(context_dim, channels),
                nn.Sigmoid(),
            )
            self.mode = "context"
        else:
            # Learned static gate: one α per channel.
            # Initialize to -2.0 → sigmoid(-2) ≈ 0.12 → start mostly vanilla (favor original).
            self.gate = nn.Parameter(torch.full((1, channels, 1, 1), -2.0))
            self.mode = "static"

        # Initialize near 0 (favor original features initially)
        self._init_weights()

    def _init_weights(self):
        if self.mode == "context":
            nn.init.zeros_(self.gate[0].weight)
            # Bias = -2 → sigmoid(-2) ≈ 0.12 → start mostly vanilla
            nn.init.constant_(self.gate[0].bias, -2.0)

    def forward(
        self,
        original: torch.Tensor,
        conditioned: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Blend original and conditioned features.

        Args:
            original: (B, C, H, W) features before FiLM.
            conditioned: (B, C, H, W) features after FiLM.
            context: (B, context_dim) context vector (if context-dependent mode).

        Returns:
            Blended features (B, C, H, W).
        """
        if self.mode == "context" and context is not None:
            alpha = self.gate(context).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        else:
            alpha = torch.sigmoid(self.gate)  # (1, C, 1, 1) broadcast

        return alpha * conditioned + (1 - alpha) * original
