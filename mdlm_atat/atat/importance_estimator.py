"""
Importance Estimator Module

This module implements a lightweight neural network that predicts the "difficulty" 
or "importance" of each token in a sequence. The importance scores are used to 
guide adaptive masking during the diffusion process.

Key Ideas:
- Tokens with high importance (e.g., content words, rare tokens) need more denoising steps
- Tokens with low importance (e.g., common function words) can be denoised earlier
- Importance is learned jointly with the diffusion model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ImportanceEstimator(nn.Module):
    """
    Estimates the importance/difficulty of each token in a sequence.
    
    The importance score influences:
    1. Masking probability during forward diffusion
    2. Denoising priority during sampling
    3. Curriculum learning schedules
    
    Args:
        vocab_size: Size of the vocabulary
        hidden_dim: Hidden dimension for importance estimation
        num_layers: Number of transformer layers for context-aware importance
        max_seq_length: Maximum sequence length
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        max_seq_length: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        
        # Token embedding for importance estimation
        self.token_embed = nn.Embedding(vocab_size, hidden_dim)
        
        # Positional encoding
        self.pos_embed = nn.Embedding(max_seq_length, hidden_dim)
        
        # Context-aware importance estimation using small transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        
        # Output projection to importance scores
        self.importance_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),  # Importance scores in [0, 1]
        )
        
        # Optional: Token frequency prior (can be updated during training)
        self.register_buffer(
            'token_frequency_prior',
            torch.ones(vocab_size) / vocab_size
        )
        
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor = None,
        return_context: bool = False,
    ) -> torch.Tensor:
        """
        Compute importance scores for each token.
        
        Args:
            x: Token IDs of shape (batch_size, seq_length)
            attention_mask: Attention mask of shape (batch_size, seq_length)
            return_context: Whether to return contextual embeddings
            
        Returns:
            importance: Importance scores of shape (batch_size, seq_length)
            context (optional): Contextual embeddings if return_context=True
        """
        batch_size, seq_length = x.shape
        device = x.device
        
        # Get token embeddings
        token_emb = self.token_embed(x)  # (B, L, D)
        
        # Add positional embeddings
        positions = torch.arange(seq_length, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_embed(positions)
        
        # Combine embeddings
        h = token_emb + pos_emb
        
        # Apply transformer for context-aware representations
        if attention_mask is not None:
            # Convert attention mask to transformer format (inverted)
            attn_mask = ~attention_mask.bool()
        else:
            attn_mask = None
            
        context = self.transformer(h, src_key_padding_mask=attn_mask)
        
        # Compute importance scores
        importance = self.importance_head(context).squeeze(-1)  # (B, L)
        
        # Optional: Incorporate token frequency prior
        # Rare tokens are generally more important
        freq_prior = 1.0 - torch.log(self.token_frequency_prior[x] + 1e-10)
        freq_prior = freq_prior / freq_prior.max()
        
        # Combine learned importance with frequency prior
        importance = 0.7 * importance + 0.3 * freq_prior
        
        # Mask out padding tokens
        if attention_mask is not None:
            importance = importance * attention_mask
            
        if return_context:
            return importance, context
        return importance
    
    def update_frequency_prior(self, token_counts: torch.Tensor):
        """
        Update token frequency prior based on training data statistics.
        
        Args:
            token_counts: Token occurrence counts of shape (vocab_size,)
        """
        total_counts = token_counts.sum()
        self.token_frequency_prior.copy_(token_counts / total_counts)
        
    def get_importance_statistics(self, importance_scores: torch.Tensor) -> dict:
        """
        Compute statistics about importance scores for analysis.
        
        Args:
            importance_scores: Importance scores of shape (batch_size, seq_length)
            
        Returns:
            Dictionary with statistics (mean, std, min, max, etc.)
        """
        return {
            'mean': importance_scores.mean().item(),
            'std': importance_scores.std().item(),
            'min': importance_scores.min().item(),
            'max': importance_scores.max().item(),
            'median': importance_scores.median().item(),
        }


class AdaptiveImportanceEstimator(ImportanceEstimator):
    """
    Enhanced importance estimator that adapts based on the diffusion timestep.
    
    Early timesteps (t → 1): More uniform importance (explore broadly)
    Late timesteps (t → 0): Sharp importance distinctions (focus on hard tokens)
    """
    
    def __init__(self, *args, time_conditioning: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.time_conditioning = time_conditioning
        
        if time_conditioning:
            # Time embedding for adaptive importance
            self.time_embed = nn.Sequential(
                nn.Linear(1, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
            
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        return_context: bool = False,
    ) -> torch.Tensor:
        """
        Compute time-conditioned importance scores.
        
        Args:
            x: Token IDs
            t: Diffusion timestep in [0, 1]
            attention_mask: Attention mask
            return_context: Whether to return context
            
        Returns:
            Importance scores (potentially time-modulated)
        """
        # Get base importance
        importance = super().forward(x, attention_mask, return_context=False)
        
        if self.time_conditioning and t is not None:
            # Reshape t for broadcasting
            if t.dim() == 0:
                t = t.unsqueeze(0)
            if t.dim() == 1:
                t = t.unsqueeze(1)
                
            # Compute time modulation
            # Early in diffusion (t → 1): flatten importance (explore)
            # Late in diffusion (t → 0): sharpen importance (exploit)
            sharpness = 1.0 / (t + 0.1)  # Increases as t → 0
            importance = torch.pow(importance, sharpness)
            
        return importance
