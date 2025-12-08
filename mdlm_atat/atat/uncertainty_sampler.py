"""
Uncertainty-Guided Sampler

Implements sampling strategies that use model uncertainty to guide the denoising process.
Tokens with high uncertainty are denoised earlier or with more attention.

This complements the importance-based masking by using runtime model confidence
to dynamically adjust the sampling strategy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List


class UncertaintyGuidedSampler:
    """
    Guides the sampling process using model uncertainty estimates.
    
    Key Idea:
    - During sampling, prioritize denoising tokens where the model is most uncertain
    - Use predictive entropy or variance as uncertainty measures
    - Combine with importance scores for optimal denoising order
    
    Args:
        uncertainty_metric: Method to compute uncertainty ('entropy', 'variance', 'both')
        temperature: Temperature for uncertainty-based sampling
        top_k: Number of most uncertain tokens to denoise per step
    """
    
    def __init__(
        self,
        uncertainty_metric: str = 'entropy',
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        adaptive_steps: bool = True,
    ):
        self.uncertainty_metric = uncertainty_metric
        self.temperature = temperature
        self.top_k = top_k
        self.adaptive_steps = adaptive_steps
        
    def compute_uncertainty(
        self,
        logits: torch.Tensor,
        method: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Compute uncertainty from model logits.
        
        Args:
            logits: Model logits of shape (batch_size, seq_length, vocab_size)
            method: Uncertainty method (defaults to self.uncertainty_metric)
            
        Returns:
            Uncertainty scores of shape (batch_size, seq_length)
        """
        method = method or self.uncertainty_metric
        
        # Convert logits to probabilities
        probs = F.softmax(logits, dim=-1)
        
        if method == 'entropy':
            # Predictive entropy: H[p(x)] = -sum(p log p)
            uncertainty = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
            
        elif method == 'variance':
            # Variance: Var[p(x)]
            mean_prob = probs.mean(dim=-1, keepdim=True)
            uncertainty = ((probs - mean_prob) ** 2).sum(dim=-1)
            
        elif method == 'max_prob':
            # Inverse of max probability (low confidence)
            max_prob = probs.max(dim=-1)[0]
            uncertainty = 1.0 - max_prob
            
        elif method == 'both':
            # Combine entropy and variance
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
            mean_prob = probs.mean(dim=-1, keepdim=True)
            variance = ((probs - mean_prob) ** 2).sum(dim=-1)
            
            # Normalize and combine
            entropy_norm = entropy / entropy.max()
            variance_norm = variance / (variance.max() + 1e-10)
            uncertainty = 0.5 * (entropy_norm + variance_norm)
            
        else:
            raise ValueError(f"Unknown uncertainty method: {method}")
            
        return uncertainty
        
    def select_tokens_to_denoise(
        self,
        uncertainty: torch.Tensor,
        importance: Optional[torch.Tensor] = None,
        mask_indices: Optional[torch.Tensor] = None,
        num_tokens: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Select which tokens to denoise based on uncertainty and importance.
        
        Args:
            uncertainty: Uncertainty scores of shape (batch_size, seq_length)
            importance: Optional importance scores
            mask_indices: Boolean tensor indicating which tokens are currently masked
            num_tokens: Number of tokens to denoise (defaults to self.top_k)
            
        Returns:
            Boolean tensor indicating which tokens to denoise
        """
        batch_size, seq_length = uncertainty.shape
        num_tokens = num_tokens or self.top_k or seq_length
        
        # Combine uncertainty with importance if provided
        if importance is not None:
            # High uncertainty AND high importance = highest priority
            combined_score = uncertainty * importance
        else:
            combined_score = uncertainty
            
        # Only consider currently masked tokens
        if mask_indices is not None:
            combined_score = combined_score * mask_indices.float()
            combined_score = combined_score + (1 - mask_indices.float()) * (-1e10)
            
        # Select top-k most uncertain/important tokens
        if num_tokens < seq_length:
            _, top_indices = torch.topk(combined_score, k=num_tokens, dim=-1)
            denoise_mask = torch.zeros_like(combined_score, dtype=torch.bool)
            denoise_mask.scatter_(1, top_indices, True)
        else:
            denoise_mask = torch.ones_like(combined_score, dtype=torch.bool)
            
        return denoise_mask
        
    def adaptive_sampling_schedule(
        self,
        total_steps: int,
        uncertainty_history: List[float],
    ) -> int:
        """
        Adaptively determine number of denoising steps based on uncertainty.
        
        If model is very uncertain, use more steps.
        If model is confident, can use fewer steps.
        
        Args:
            total_steps: Default number of sampling steps
            uncertainty_history: History of average uncertainty scores
            
        Returns:
            Adjusted number of steps
        """
        if not self.adaptive_steps or len(uncertainty_history) == 0:
            return total_steps
            
        # Recent average uncertainty
        recent_uncertainty = np.mean(uncertainty_history[-10:])
        
        # Adjust steps based on uncertainty
        # High uncertainty → more steps
        # Low uncertainty → fewer steps
        if recent_uncertainty > 0.7:
            adjusted_steps = int(total_steps * 1.2)
        elif recent_uncertainty < 0.3:
            adjusted_steps = int(total_steps * 0.8)
        else:
            adjusted_steps = total_steps
            
        return adjusted_steps
        
    def uncertainty_guided_update(
        self,
        x_t: torch.Tensor,
        logits: torch.Tensor,
        importance: Optional[torch.Tensor] = None,
        mask_index: int = None,
        denoise_fraction: float = 0.1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform one step of uncertainty-guided denoising.
        
        Args:
            x_t: Current masked tokens of shape (batch_size, seq_length)
            logits: Model logits
            importance: Optional importance scores
            mask_index: Index representing masked tokens
            denoise_fraction: Fraction of masked tokens to denoise this step
            
        Returns:
            Updated tokens and uncertainty scores
        """
        batch_size, seq_length = x_t.shape
        
        # Compute uncertainty
        uncertainty = self.compute_uncertainty(logits)
        
        # Find currently masked tokens
        if mask_index is not None:
            mask_indices = (x_t == mask_index)
        else:
            mask_indices = torch.ones_like(x_t, dtype=torch.bool)
            
        # Determine how many tokens to denoise
        num_masked = mask_indices.sum(dim=-1).float().mean().item()
        num_to_denoise = max(1, int(num_masked * denoise_fraction))
        
        # Select tokens to denoise based on uncertainty
        denoise_mask = self.select_tokens_to_denoise(
            uncertainty,
            importance,
            mask_indices,
            num_to_denoise,
        )
        
        # Denoise selected tokens (sample from logits)
        probs = F.softmax(logits / self.temperature, dim=-1)
        sampled_tokens = torch.multinomial(
            probs.view(-1, probs.size(-1)),
            num_samples=1,
        ).view(batch_size, seq_length)
        
        # Update only selected tokens
        x_t_new = x_t.clone()
        x_t_new[denoise_mask] = sampled_tokens[denoise_mask]
        
        return x_t_new, uncertainty
        
    def get_uncertainty_statistics(
        self,
        uncertainty: torch.Tensor,
        mask_indices: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Compute statistics about uncertainty for analysis.
        
        Args:
            uncertainty: Uncertainty scores
            mask_indices: Optional mask indicating which tokens to analyze
            
        Returns:
            Dictionary with statistics
        """
        if mask_indices is not None:
            uncertainty = uncertainty[mask_indices]
            
        return {
            'mean': uncertainty.mean().item(),
            'std': uncertainty.std().item(),
            'min': uncertainty.min().item(),
            'max': uncertainty.max().item(),
            'median': uncertainty.median().item(),
        }


class ConfidenceBasedSampler(UncertaintyGuidedSampler):
    """
    Extended sampler that tracks and uses model confidence over time.
    
    Maintains a history of confidence levels and adapts sampling strategy
    based on how confident the model has been historically.
    """
    
    def __init__(self, *args, confidence_threshold: float = 0.8, **kwargs):
        super().__init__(*args, **kwargs)
        self.confidence_threshold = confidence_threshold
        self.confidence_history = []
        
    def update_confidence_history(self, uncertainty: torch.Tensor):
        """Track confidence (inverse of uncertainty) over time."""
        confidence = 1.0 - uncertainty.mean().item()
        self.confidence_history.append(confidence)
        
        # Keep only recent history
        if len(self.confidence_history) > 100:
            self.confidence_history = self.confidence_history[-100:]
            
    def should_continue_sampling(self) -> bool:
        """
        Determine if sampling should continue based on confidence.
        
        Returns:
            True if model is still uncertain, False if confident enough
        """
        if len(self.confidence_history) < 5:
            return True
            
        recent_confidence = np.mean(self.confidence_history[-5:])
        return recent_confidence < self.confidence_threshold
        
    def get_sampling_temperature(self) -> float:
        """
        Adjust sampling temperature based on confidence level.
        
        Higher confidence → lower temperature (more deterministic)
        Lower confidence → higher temperature (more exploration)
        """
        if len(self.confidence_history) == 0:
            return self.temperature
            
        recent_confidence = np.mean(self.confidence_history[-10:])
        
        # Inverse relationship: low confidence → high temperature
        adjusted_temp = self.temperature * (2.0 - recent_confidence)
        return np.clip(adjusted_temp, 0.5, 2.0)
