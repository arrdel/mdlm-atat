"""
ATAT Utils Package

Utility modules for dataset configuration, training, and evaluation.
"""

# Import dataset configuration for global access
try:
    from mdlm_atat.utils.dataset_config import (
        DatasetConfigManager,
        DatasetConfig,
        get_dataset_manager,
        get_dataset,
        get_default_dataset,
    )
    __all__ = [
        "DatasetConfigManager",
        "DatasetConfig",
        "get_dataset_manager",
        "get_dataset",
        "get_default_dataset",
    ]
except ImportError:
    # Fallback for direct imports
    __all__ = []
