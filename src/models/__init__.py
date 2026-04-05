"""
Singapore Smart City — Novel Model Architectures

CATI (Context-Aware Traffic Intelligence):
    A FiLM-conditioned detection pipeline that adapts YOLOv11's feature
    extraction to environmental conditions using real-time metadata from
    Singapore's national APIs.

Modules:
    film:             Feature-wise Linear Modulation layers
    context_encoder:  Environmental metadata encoder
    cati_detector:    Full CATI detector + inference pipeline
"""

from src.models.context_encoder import ContextEncoder
from src.models.film import FiLMGenerator, FiLMLayer

__all__ = ["ContextEncoder", "FiLMGenerator", "FiLMLayer"]
