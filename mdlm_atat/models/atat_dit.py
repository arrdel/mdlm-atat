"""
ATAT-Enhanced DiT Model

This module implements the DiT (Diffusion Transformer) architecture enhanced with
the ATAT (Adaptive Time-Aware Token Masking) components.

Key Enhancements:
1. Integrated ImportanceEstimator for token-level importance scores
2. Adaptive masking during training based on importance
3. Uncertainty-guided sampling during inference
4. Curriculum learning support
"""

import sys
sys.path.insert(0, '/home/adelechinda/home/projects/mdlm')

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict

# Import baseline DiT from MDLM
from mdlm.models.dit import DIT, DDiTBlock, EmbeddingLayer, DDitFinalLayer, TimestepEmbedder

# Import ATAT components
from mdlm_atat.atat import (
    ImportanceEstimator,
    AdaptiveImportanceEstimator,
    AdaptiveMaskingScheduler,
    CurriculumScheduler,
    UncertaintyGuidedSampler,
)
 

class ATATDiT(DIT):
    """
    ATAT-enhanced Diffusion Transformer.
    
    Extends the baseline DiT with:
    - Importance estimation network
    - Adaptive masking scheduler
    - Curriculum learning
    - Uncertainty-guided sampling
    
    Args:
        config: Model configuration
        vocab_size: Vocabulary size
        use_importance: Whether to use importance estimation
        use_adaptive_masking: Whether to use adaptive masking
        use_curriculum: Whether to use curriculum learning
        importance_loss_weight: Weight for importance estimation loss
    """
    
    def __init__(
        self,
        config,
        vocab_size: int,
        use_importance: bool = True,
        use_adaptive_masking: bool = True,
        use_curriculum: bool = True,
        importance_loss_weight: float = 0.1,
    ):
        super().__init__(config, vocab_size)
        
        self.use_importance = use_importance
        self.use_adaptive_masking = use_adaptive_masking
        self.use_curriculum = use_curriculum
        self.importance_loss_weight = importance_loss_weight
        
        # Initialize ATAT components
        if self.use_importance:
            self.importance_estimator = AdaptiveImportanceEstimator(
                vocab_size=vocab_size,
                hidden_dim=config.model.get('importance_hidden_dim', 256),
                num_layers=config.model.get('importance_num_layers', 2),
                max_seq_length=config.model.length,
                time_conditioning=True,
            )
            
        if self.use_adaptive_masking:
            self.masking_scheduler = AdaptiveMaskingScheduler(
                masking_strategy=config.model.get('masking_strategy', 'importance'),
                temperature=config.model.get('masking_temperature', 1.0),
                position_bias=config.model.get('position_bias', False),
            )
            
        if self.use_curriculum:
            # Will be initialized during training with total_steps
            self.curriculum_scheduler = None
            
        # Uncertainty-guided sampler for inference
        self.uncertainty_sampler = UncertaintyGuidedSampler(
            uncertainty_metric='entropy',
            temperature=1.0,
            adaptive_steps=True,
        )
        
    def initialize_curriculum(self, total_steps: int):
        """Initialize curriculum scheduler with total training steps."""
        if self.use_curriculum:
            self.curriculum_scheduler = CurriculumScheduler(
                total_steps=total_steps,
                curriculum_type='linear',
                warmup_steps=min(1000, total_steps // 10),
            )
            
    def forward(
        self,
        indices: torch.Tensor,
        sigma: torch.Tensor,
        return_importance: bool = False,
        return_uncertainty: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass with optional importance estimation.
        
        Args:
            indices: Input token indices of shape (batch_size, seq_length)
            sigma: Noise level
            return_importance: Whether to return importance scores
            return_uncertainty: Whether to return uncertainty estimates
            
        Returns:
            logits: Model logits
            importance (optional): Importance scores if return_importance=True
            uncertainty (optional): Uncertainty if return_uncertainty=True
        """
        # Standard DiT forward pass
        logits = super().forward(indices, sigma)
        
        # Compute importance if requested
        if return_importance and self.use_importance:
            with torch.no_grad() if not self.training else torch.enable_grad():
                importance = self.importance_estimator(
                    indices,
                    t=sigma if sigma is not None else None,
                )
        else:
            importance = None
            
        # Compute uncertainty if requested
        if return_uncertainty:
            uncertainty = self.uncertainty_sampler.compute_uncertainty(logits)
        else:
            uncertainty = None
            
        # Return appropriate outputs
        if return_importance and return_uncertainty:
            return logits, importance, uncertainty
        elif return_importance:
            return logits, importance
        elif return_uncertainty:
            return logits, uncertainty
        else:
            return logits
            
    def compute_importance_loss(
        self,
        importance: torch.Tensor,
        x_0: torch.Tensor,
        x_t: torch.Tensor,
        mask_index: int,
    ) -> torch.Tensor:
        """
        Compute auxiliary loss for importance estimation.
        
        The importance estimator should learn to predict which tokens are:
        1. Harder to denoise (high reconstruction error)
        2. More informative (high mutual information with context)
        
        Args:
            importance: Predicted importance scores
            x_0: Original clean tokens
            x_t: Noisy tokens
            mask_index: Index for masked tokens
            
        Returns:
            Importance estimation loss
        """
        # Tokens that were masked (and need to be reconstructed)
        was_masked = (x_t == mask_index).float()
        
        # Encourage importance estimator to identify masked tokens
        # This creates a weak supervision signal
        target_importance = was_masked
        
        # MSE loss between predicted and target importance
        importance_loss = F.mse_loss(importance, target_importance)
        
        return importance_loss
        
    def adaptive_forward_diffusion(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        mask_index: int,
        training_step: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward diffusion process with adaptive masking.
        
        Args:
            x_0: Clean tokens
            t: Diffusion timestep
            mask_index: Index for masked tokens
            training_step: Current training step (for curriculum)
            
        Returns:
            x_t: Noisy tokens
            importance: Importance scores
            curriculum_weights: Optional curriculum weights
        """
        # Compute importance scores
        if self.use_importance:
            importance = self.importance_estimator(x_0, t=t)
        else:
            # Uniform importance if not using importance estimator
            importance = torch.ones_like(x_0, dtype=torch.float)
            
        # Apply curriculum learning if enabled
        curriculum_weights = None
        if self.use_curriculum and self.curriculum_scheduler is not None and training_step is not None:
            current_stage = self.curriculum_scheduler.get_current_stage(training_step)
            curriculum_weights = self.curriculum_scheduler.compute_curriculum_weights(
                importance, current_stage
            )
            # Modulate importance by curriculum weights
            importance = importance * curriculum_weights
            
        # Apply adaptive masking
        if self.use_adaptive_masking:
            x_t = self.masking_scheduler.sample_masks(
                x_0, importance, t, mask_index
            )
        else:
            # Standard uniform masking (baseline)
            move_chance = t.unsqueeze(1)  # (batch_size, 1)
            mask_decisions = torch.rand_like(x_0, dtype=torch.float) < move_chance
            x_t = torch.where(mask_decisions, mask_index, x_0)
            
        return x_t, importance, curriculum_weights
        
    def uncertainty_guided_sample_step(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        dt: float,
        mask_index: int,
        importance: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        One step of uncertainty-guided sampling.
        
        Args:
            x_t: Current noisy tokens
            t: Current timestep
            dt: Time step size
            mask_index: Masked token index
            importance: Optional importance scores
            
        Returns:
            x_next: Updated tokens
            uncertainty: Uncertainty scores
        """
        # Get model predictions
        sigma = t if t.dim() > 0 else t.unsqueeze(0)
        logits = self.forward(x_t, sigma)
        
        # Use uncertainty-guided denoising
        x_next, uncertainty = self.uncertainty_sampler.uncertainty_guided_update(
            x_t=x_t,
            logits=logits,
            importance=importance,
            mask_index=mask_index,
            denoise_fraction=dt,
        )
        
        return x_next, uncertainty
        
    def get_atat_components(self) -> Dict[str, nn.Module]:
        """Get dictionary of ATAT-specific components."""
        components = {}
        
        if self.use_importance:
            components['importance_estimator'] = self.importance_estimator
        if self.use_adaptive_masking:
            components['masking_scheduler'] = self.masking_scheduler
        if self.use_curriculum and self.curriculum_scheduler is not None:
            components['curriculum_scheduler'] = self.curriculum_scheduler
            
        components['uncertainty_sampler'] = self.uncertainty_sampler
        
        return components
        
    def log_atat_statistics(
        self,
        importance: Optional[torch.Tensor] = None,
        uncertainty: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Log statistics about ATAT components for monitoring.
        
        Returns:
            Dictionary with statistics
        """
        stats = {}
        
        if importance is not None and self.use_importance:
            imp_stats = self.importance_estimator.get_importance_statistics(importance)
            stats.update({f'importance/{k}': v for k, v in imp_stats.items()})
            
        if uncertainty is not None:
            unc_stats = self.uncertainty_sampler.get_uncertainty_statistics(uncertainty)
            stats.update({f'uncertainty/{k}': v for k, v in unc_stats.items()})
            
        return stats


class ATATDiTWrapper(nn.Module):
    """
    Wrapper for ATAT-DiT that handles training and sampling logic.
    
    This makes it easier to integrate with the existing MDLM training code.
    """
    
    def __init__(self, config, tokenizer):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.mask_index = tokenizer.mask_token_id if hasattr(tokenizer, 'mask_token_id') else config.vocab_size
        
        # Initialize ATAT-DiT
        self.model = ATATDiT(
            config=config,
            vocab_size=tokenizer.vocab_size,
            use_importance=config.get('use_importance', True),
            use_adaptive_masking=config.get('use_adaptive_masking', True),
            use_curriculum=config.get('use_curriculum', True),
        )
        
    def training_step(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        step: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Single training step with ATAT components.
        
        Args:
            x_0: Clean tokens
            t: Diffusion timestep
            step: Training step number
            
        Returns:
            loss: Total loss
            metrics: Dictionary of metrics for logging
        """
        # Forward diffusion with adaptive masking
        x_t, importance, curriculum_weights = self.model.adaptive_forward_diffusion(
            x_0, t, self.mask_index, training_step=step
        )
        
        # Get model predictions
        logits, pred_importance = self.model(
            x_t, t, return_importance=True
        )
        
        # Compute main reconstruction loss
        main_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            x_0.reshape(-1),
            reduction='none',
        ).reshape(x_0.shape)
        
        # Apply curriculum weights if available
        if curriculum_weights is not None:
            main_loss = main_loss * curriculum_weights
            
        main_loss = main_loss.mean()
        
        # Compute importance estimation loss
        if self.model.use_importance and pred_importance is not None:
            importance_loss = self.model.compute_importance_loss(
                pred_importance, x_0, x_t, self.mask_index
            )
            total_loss = main_loss + self.model.importance_loss_weight * importance_loss
        else:
            importance_loss = torch.tensor(0.0)
            total_loss = main_loss
            
        # Collect metrics
        metrics = {
            'loss/total': total_loss.item(),
            'loss/main': main_loss.item(),
            'loss/importance': importance_loss.item(),
        }
        
        # Add ATAT statistics
        atat_stats = self.model.log_atat_statistics(
            importance=importance,
        )
        metrics.update(atat_stats)
        
        return total_loss, metrics
