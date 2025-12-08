"""
ATAT (Adaptive Time-Aware Token Masking) Module

This module contains the core components of the ATAT extension to MDLM:

1. ImportanceEstimator: Learns token-level importance/difficulty scores
2. AdaptiveMaskingScheduler: Adjusts masking probabilities based on importance
3. CurriculumScheduler: Implements curriculum learning from easy to hard tokens
4. UncertaintyGuidedSampler: Uses model confidence to guide sampling

These components work together to enable adaptive, importance-aware masking
that improves upon the uniform random masking in baseline MDLM.
"""

from .importance_estimator import (
    ImportanceEstimator,
    AdaptiveImportanceEstimator,
)

from .adaptive_masking import (
    AdaptiveMaskingScheduler,
    PositionAwareMaskingScheduler,
)

from .curriculum import (
    CurriculumScheduler,
    DynamicCurriculumScheduler,
)

from .uncertainty_sampler import (
    UncertaintyGuidedSampler,
    ConfidenceBasedSampler,
)

__all__ = [
    # Importance estimation
    'ImportanceEstimator',
    'AdaptiveImportanceEstimator',
    
    # Adaptive masking
    'AdaptiveMaskingScheduler',
    'PositionAwareMaskingScheduler',
    
    # Curriculum learning
    'CurriculumScheduler',
    'DynamicCurriculumScheduler',
    
    # Uncertainty-guided sampling
    'UncertaintyGuidedSampler',
    'ConfidenceBasedSampler',
]
