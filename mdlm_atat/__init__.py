"""
MDLM-ATAT: Adaptive Time-Aware Token Masking for Masked Diffusion Language Models

This package extends the original MDLM framework with a novel adaptive masking strategy
that learns token importance and adjusts masking probabilities dynamically during the
diffusion process.

Key Components:
- atat: Core ATAT modules (adaptive masking, curriculum learning, uncertainty sampling)
- models: ATAT-enhanced model architectures
- configs: Configuration files for experiments
- utils: Utility functions and helpers
"""

__version__ = "0.1.0"
__author__ = "MDLM-ATAT Research Team"

from mdlm_atat.atat import (
    AdaptiveMaskingScheduler,
    ImportanceEstimator,
    CurriculumScheduler,
    UncertaintyGuidedSampler,
)

from mdlm_atat.models import ATATDiT

__all__ = [
    "AdaptiveMaskingScheduler",
    "ImportanceEstimator",
    "CurriculumScheduler",
    "UncertaintyGuidedSampler",
    "ATATDiT",
]
