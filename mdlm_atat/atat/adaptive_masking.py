"""
Adaptive Masking Scheduler

This module implements the core ATAT innovation: dynamically adjusting masking 
probabilities based on token importance and diffusion timestep.

Instead of uniform random masking, we use importance-weighted masking where:
- Important tokens have higher probability of being masked early
- Less important tokens can be denoised more quickly
- Masking strategy evolves throughout the diffusion process



ARM vs Diffusion Masking Example:

# Autoregressive: sees only left context
"The cat sat on the [MASK]"  # Hard to predict "mat" without right context

# Diffusion: sees bidirectional context from step 1
"The [MASK] sat [MASK] the [MASK]"  # Can use "the...the" pattern


Step 1: "The [MASK] [MASK] [MASK] the [MASK]"  → "animal"
Step 2: "The animal [MASK] [MASK] the [MASK]"   → "sat"  
Step 3: "The animal sat [MASK] the [MASK]"      → "on"
Step 4: "The animal sat on the [MASK]"          → "floor" (oops)
Step 5: "The cat sat on the [MASK]"             → "mat" (corrected!)




Adaptive Masking

Early Training: "The [MASK] sat on the [MASK]"
                Focus on common nouns, basic syntax
                
Mid Training:   "The [MASK] demonstrated quantum [MASK]"  
                Mix of medium and hard concepts
                
Late Training:  "[MASK] physics equations [MASK] entanglement [MASK]"
                Focus on domain-specific, complex relationships
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AdaptiveMaskingScheduler(nn.Module):
    """
    Schedules masking probabilities based on token importance and diffusion time.
    
    Core Innovation:
    Instead of p_mask(token) = f(t) for all tokens,
    we have p_mask(token) = f(t, importance(token), position(token))
    
    Args:
        masking_strategy: Strategy for adaptive masking ('importance', 'inverse', 'balanced')
        temperature: Temperature for importance-based masking
        position_bias: Whether to add positional bias to masking
    """
    
    def __init__(
        self,
        masking_strategy: str = 'importance',
        temperature: float = 1.0,
        position_bias: bool = False,
        curriculum_schedule: str = 'linear',
    ):
        super().__init__()
        self.masking_strategy = masking_strategy
        self.temperature = temperature
        self.position_bias = position_bias
        self.curriculum_schedule = curriculum_schedule
        
        # Learnable parameters for adaptive masking
        self.importance_weight = nn.Parameter(torch.tensor(0.7))
        self.time_weight = nn.Parameter(torch.tensor(0.3))
        
        

    def get_base_masking_rate(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute base masking rate as a function of timestep t.
        
        Args:
            t: Diffusion timestep in [0, 1]
            
        Returns:
            Base masking rate in [0, 1]
        """
        if self.curriculum_schedule == 'linear':
            # Linear schedule: mask more at early timesteps
            return t
        elif self.curriculum_schedule == 'cosine':
            # Cosine schedule: smoother transition 
            return 0.5 * (1 + torch.cos(np.pi * (1 - t)))
        elif self.curriculum_schedule == 'log':
            # Logarithmic: mask heavily early, taper off
            return torch.log1p(t * 9) / torch.log1p(torch.tensor(9.0))
        else:
            return t
            
    def compute_importance_weighted_mask_prob(
        self,
        importance: torch.Tensor,
        t: torch.Tensor,
        positions: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute per-token masking probabilities using importance scores.
        
        Args:
            importance: Importance scores of shape (batch_size, seq_length)
            t: Diffusion timestep (scalar or batch)
            positions: Optional position indices
            
        Returns:
            Masking probabilities of shape (batch_size, seq_length)
        """
        batch_size, seq_length = importance.shape
        
        # Get base masking rate from timestep
        base_rate = self.get_base_masking_rate(t)
        if base_rate.dim() == 0:
            base_rate = base_rate.unsqueeze(0)
        base_rate = base_rate.unsqueeze(1)  # (B, 1)
        
        if self.masking_strategy == 'importance':
            # Higher importance → higher masking probability
            # Intuition: Hard tokens need more denoising steps
            importance_factor = torch.pow(importance, 1.0 / self.temperature)
            mask_prob = base_rate * importance_factor
            
        elif self.masking_strategy == 'inverse':
            # Lower importance → higher masking probability
            # Intuition: Easy tokens can be masked/denoised quickly
            importance_factor = torch.pow(1.0 - importance, 1.0 / self.temperature)
            mask_prob = base_rate * importance_factor
            
        elif self.masking_strategy == 'balanced':
            # Balanced: importance affects distribution but maintains rate
            importance_normalized = importance / (importance.sum(dim=-1, keepdim=True) + 1e-8)
            mask_prob = base_rate * importance_normalized * seq_length
            
        else:
            # Uniform (baseline): no importance weighting
            mask_prob = base_rate.expand(batch_size, seq_length)
            
        # Optional: Add position bias
        if self.position_bias and positions is not None:
            pos_bias = self._compute_position_bias(positions)
            mask_prob = mask_prob * pos_bias
            
        # Clamp to valid probability range
        mask_prob = torch.clamp(mask_prob, 0.0, 1.0)
        
        return mask_prob
        
    def _compute_position_bias(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Compute position-dependent bias for masking.
        
        Tokens at beginning/end of sequences often provide strong context.
        We may want to preserve them longer or mask them earlier.
        """
        seq_length = positions.max() + 1
        middle = seq_length // 2
        
        # U-shaped bias: preserve edges, mask middle more
        distance_from_edge = torch.minimum(
            positions,
            seq_length - 1 - positions
        )
        bias = 1.0 + 0.2 * (distance_from_edge.float() / middle - 0.5)
        
        return bias
        
    def sample_masks(
        self,
        x: torch.Tensor,
        importance: torch.Tensor,
        t: torch.Tensor,
        mask_index: int,
    ) -> torch.Tensor:
        """
        Sample adaptive masks for the forward diffusion process.
        
        Args:
            x: Input token IDs of shape (batch_size, seq_length)
            importance: Importance scores of shape (batch_size, seq_length)
            t: Diffusion timestep
            mask_index: Index used for masked tokens
            
        Returns:
            Masked version of x
        """
        # Compute per-token masking probabilities
        mask_probs = self.compute_importance_weighted_mask_prob(
            importance, t
        )
        
        # Sample binary masks
        mask_decisions = torch.bernoulli(mask_probs).bool()
        
        # Apply masks
        x_masked = x.clone()
        x_masked[mask_decisions] = mask_index
        
        return x_masked
        
    def get_curriculum_stage(self, step: int, total_steps: int) -> str:
        """
        Determine curriculum learning stage based on training progress.
        
        Args:
            step: Current training step
            total_steps: Total training steps
            
        Returns:
            Stage name ('easy', 'medium', 'hard')
        """
        progress = step / total_steps
        
        if progress < 0.3:
            return 'easy'
        elif progress < 0.7:
            return 'medium'
        else:
            return 'hard'
            
    def adjust_for_curriculum(
        self,
        mask_prob: torch.Tensor,
        importance: torch.Tensor,
        stage: str,
    ) -> torch.Tensor:
        """
        Adjust masking probabilities based on curriculum stage.
        
        Early training (easy): Mask low-importance tokens
        Mid training (medium): Balanced masking
        Late training (hard): Focus on high-importance tokens
        """
        if stage == 'easy':
            # Focus on easy tokens (low importance)
            adjusted_prob = mask_prob * (1.0 - importance)
        elif stage == 'hard':
            # Focus on hard tokens (high importance)
            adjusted_prob = mask_prob * importance
        else:
            # Balanced
            adjusted_prob = mask_prob
            
        return adjusted_prob


class PositionAwareMaskingScheduler(AdaptiveMaskingScheduler):
    """
    Extended adaptive masking scheduler with explicit position modeling.
    
    Different positions in a sequence may have different importance:
    - Beginning: Sets context, important for coherence
    - Middle: Main content, varies in importance
    - End: Conclusions, often important
    """
    
    def __init__(self, *args, position_encoding: str = 'learned', **kwargs):
        super().__init__(*args, **kwargs)
        self.position_encoding = position_encoding
        
        if position_encoding == 'learned':
            # Learn position-specific masking biases
            self.position_bias_net = nn.Sequential(
                nn.Embedding(1024, 64),
                nn.Linear(64, 1),
                nn.Sigmoid(),
            )
            
    def compute_importance_weighted_mask_prob(
        self,
        importance: torch.Tensor,
        t: torch.Tensor,
        positions: torch.Tensor = None,
    ) -> torch.Tensor:
        """Enhanced masking with learned position bias."""
        mask_prob = super().compute_importance_weighted_mask_prob(
            importance, t, positions
        )
        
        if self.position_encoding == 'learned' and positions is None:
            batch_size, seq_length = importance.shape
            positions = torch.arange(seq_length, device=importance.device)
            positions = positions.unsqueeze(0).expand(batch_size, -1)
            
        if self.position_encoding == 'learned':
            pos_bias = self.position_bias_net[0](positions)  # Embed
            pos_bias = self.position_bias_net[1](pos_bias).squeeze(-1)  # Project
            pos_bias = self.position_bias_net[2](pos_bias)  # Sigmoid
            
            mask_prob = mask_prob * pos_bias
            
        return mask_prob
