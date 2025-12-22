"""
Unit tests for ATAT components

Run with: pytest mdlm_atat/tests/
"""

import torch
import pytest
import sys
sys.path.insert(0, '/home/adelechinda/home/projects/mdlm')

from mdlm_atat.atat import (
    ImportanceEstimator,
    AdaptiveImportanceEstimator,
    AdaptiveMaskingScheduler,
    CurriculumScheduler,
    UncertaintyGuidedSampler,
)


class TestImportanceEstimator:
    """Test ImportanceEstimator classes."""
    
    def test_basic_importance_estimator(self):
        """Test basic importance estimation."""
        vocab_size = 1000
        hidden_dim = 128
        seq_length = 64
        batch_size = 4
        
        estimator = ImportanceEstimator(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=2,
            max_seq_length=seq_length,
        )
        
        # Create input
        x = torch.randint(0, vocab_size, (batch_size, seq_length))
        
        # Forward pass
        importance = estimator(x)
        
        # Check shape
        assert importance.shape == (batch_size, seq_length)
        
        # Check range [0, 1]
        assert (importance >= 0).all() and (importance <= 1).all()
        
    def test_adaptive_importance_estimator(self):
        """Test time-conditioned importance estimator."""
        vocab_size = 1000
        hidden_dim = 128
        seq_length = 64
        batch_size = 4
        
        estimator = AdaptiveImportanceEstimator(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=2,
            max_seq_length=seq_length,
            time_conditioning=True,
        )
        
        # Create input
        x = torch.randint(0, vocab_size, (batch_size, seq_length))
        t = torch.rand(batch_size, 1)  # Timesteps
        
        # Forward pass with time
        importance = estimator(x, t=t)
        
        # Check shape
        assert importance.shape == (batch_size, seq_length)
        
        # Check range
        assert (importance >= 0).all() and (importance <= 1).all()


class TestAdaptiveMaskingScheduler:
    """Test AdaptiveMaskingScheduler."""
    
    def test_importance_weighted_masking(self):
        """Test importance-weighted mask sampling."""
        batch_size = 4
        seq_length = 64
        vocab_size = 1000
        mask_index = vocab_size
        
        scheduler = AdaptiveMaskingScheduler(
            masking_strategy='importance',
            temperature=1.0,
        )
        
        # Create input
        x_0 = torch.randint(0, vocab_size, (batch_size, seq_length))
        importance = torch.rand(batch_size, seq_length)
        t = torch.tensor([0.5] * batch_size).unsqueeze(1)
        
        # Sample masks
        x_t = scheduler.sample_masks(x_0, importance, t, mask_index)
        
        # Check shape
        assert x_t.shape == x_0.shape
        
        # Check that some tokens are masked
        num_masked = (x_t == mask_index).sum()
        assert num_masked > 0
        
    def test_uniform_masking(self):
        """Test uniform masking (baseline)."""
        batch_size = 4
        seq_length = 64
        vocab_size = 1000
        mask_index = vocab_size
        
        scheduler = AdaptiveMaskingScheduler(
            masking_strategy='uniform',
        )
        
        # Create input
        x_0 = torch.randint(0, vocab_size, (batch_size, seq_length))
        importance = torch.ones(batch_size, seq_length)  # Uniform
        t = torch.tensor([0.5] * batch_size).unsqueeze(1)
        
        # Sample masks
        x_t = scheduler.sample_masks(x_0, importance, t, mask_index)
        
        # Check shape
        assert x_t.shape == x_0.shape


class TestCurriculumScheduler:
    """Test CurriculumScheduler."""
    
    def test_curriculum_progression(self):
        """Test curriculum stage progression."""
        total_steps = 10000
        scheduler = CurriculumScheduler(
            total_steps=total_steps,
            curriculum_type='linear',
            warmup_steps=1000,
        )
        
        # Check early stage (easy)
        stage_early = scheduler.get_current_stage(500)
        assert stage_early == 'easy'
        
        # Check middle stage
        stage_mid = scheduler.get_current_stage(5000)
        assert stage_mid in ['medium', 'mixed']
        
        # Check late stage (hard)
        stage_late = scheduler.get_current_stage(9500)
        assert stage_late == 'hard'
        
    def test_curriculum_weights(self):
        """Test curriculum weight computation."""
        total_steps = 10000
        batch_size = 4
        seq_length = 64
        
        scheduler = CurriculumScheduler(
            total_steps=total_steps,
            curriculum_type='linear',
        )
        
        # Create importance scores
        importance = torch.rand(batch_size, seq_length)
        
        # Get weights for early stage
        weights_early = scheduler.compute_curriculum_weights(importance, 'easy')
        
        # Check shape
        assert weights_early.shape == importance.shape
        
        # Check that weights favor easy tokens (low importance)
        # Should be higher for low importance tokens
        low_imp_mask = importance < 0.3
        high_imp_mask = importance > 0.7
        if low_imp_mask.any() and high_imp_mask.any():
            assert weights_early[low_imp_mask].mean() > weights_early[high_imp_mask].mean()


class TestUncertaintyGuidedSampler:
    """Test UncertaintyGuidedSampler."""
    
    def test_uncertainty_computation(self):
        """Test uncertainty metric computation."""
        batch_size = 4
        seq_length = 64
        vocab_size = 1000
        
        sampler = UncertaintyGuidedSampler(
            uncertainty_metric='entropy',
        )
        
        # Create logits
        logits = torch.randn(batch_size, seq_length, vocab_size)
        
        # Compute uncertainty
        uncertainty = sampler.compute_uncertainty(logits)
        
        # Check shape
        assert uncertainty.shape == (batch_size, seq_length)
        
        # Check non-negative
        assert (uncertainty >= 0).all()
        
    def test_token_selection(self):
        """Test uncertainty-based token selection."""
        batch_size = 4
        seq_length = 64
        vocab_size = 1000
        mask_index = vocab_size
        
        sampler = UncertaintyGuidedSampler(
            uncertainty_metric='entropy',
        )
        
        # Create input
        x_t = torch.randint(0, vocab_size, (batch_size, seq_length))
        x_t[:, :10] = mask_index  # Mask first 10 tokens
        
        logits = torch.randn(batch_size, seq_length, vocab_size)
        
        # Select tokens to denoise
        selected_mask = sampler.select_tokens_to_denoise(
            x_t, logits, mask_index, denoise_fraction=0.5
        )
        
        # Check shape
        assert selected_mask.shape == (batch_size, seq_length)
        
        # Check that we selected some tokens
        assert selected_mask.any()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
