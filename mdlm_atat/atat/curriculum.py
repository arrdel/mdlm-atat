"""
Curriculum Scheduler

Implements curriculum learning for the ATAT framework, progressively increasing
task difficulty during training.

Curriculum Strategy:
1. Early training: Focus on easy tokens (high frequency, low importance)
2. Mid training: Balanced mix of easy and hard tokens
3. Late training: Focus on hard tokens (rare, high importance)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple


class CurriculumScheduler:
    """
    Manages curriculum learning schedule for adaptive masking.
    
    The curriculum determines which tokens to focus on at each training stage,
    enabling the model to learn progressively from simple to complex patterns.
    
    Args:
        total_steps: Total number of training steps
        curriculum_type: Type of curriculum ('linear', 'step', 'exponential')
        warmup_steps: Number of warmup steps before curriculum begins
        difficulty_metric: How to measure token difficulty ('importance', 'frequency', 'both')
    """
    
    def __init__(
        self,
        total_steps: int,
        curriculum_type: str = 'linear',
        warmup_steps: int = 1000,
        difficulty_metric: str = 'both',
        num_stages: int = 3,
    ):
        self.total_steps = total_steps
        self.curriculum_type = curriculum_type
        self.warmup_steps = warmup_steps
        self.difficulty_metric = difficulty_metric
        self.num_stages = num_stages
        
        # Define curriculum stages
        self.stages = self._define_stages()
        
    def _define_stages(self) -> Dict[str, Dict]:
        """Define curriculum stages with their characteristics."""
        if self.num_stages == 3:
            return {
                'easy': {
                    'importance_range': (0.0, 0.3),
                    'focus_weight': 2.0,
                    'description': 'Focus on low-importance, high-frequency tokens',
                },
                'medium': {
                    'importance_range': (0.3, 0.7),
                    'focus_weight': 1.5,
                    'description': 'Balanced focus on all tokens',
                },
                'hard': {
                    'importance_range': (0.7, 1.0),
                    'focus_weight': 2.0,
                    'description': 'Focus on high-importance, rare tokens',
                },
            }
        else:
            # More granular stages
            stages = {}
            step_size = 1.0 / self.num_stages
            for i in range(self.num_stages):
                stage_name = f'stage_{i}'
                stages[stage_name] = {
                    'importance_range': (i * step_size, (i + 1) * step_size),
                    'focus_weight': 1.0 + 0.5 * i,
                    'description': f'Stage {i} focusing on importance {i*step_size:.2f}-{(i+1)*step_size:.2f}',
                }
            return stages
            
    def get_current_stage(self, step: int) -> str:
        """
        Get the current curriculum stage based on training progress.
        
        Args:
            step: Current training step
            
        Returns:
            Stage name
        """
        if step < self.warmup_steps:
            return 'easy'
            
        progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        progress = np.clip(progress, 0.0, 1.0)
        
        if self.curriculum_type == 'linear':
            # Linear progression through stages
            stage_idx = int(progress * self.num_stages)
            stage_idx = min(stage_idx, self.num_stages - 1)
            
        elif self.curriculum_type == 'step':
            # Discrete steps between stages
            stage_boundaries = np.linspace(0, 1, self.num_stages + 1)
            stage_idx = np.digitize(progress, stage_boundaries) - 1
            stage_idx = min(max(stage_idx, 0), self.num_stages - 1)
            
        elif self.curriculum_type == 'exponential':
            # Exponential progression (spend more time on early stages)
            exp_progress = 1 - np.exp(-3 * progress)
            stage_idx = int(exp_progress * self.num_stages)
            stage_idx = min(stage_idx, self.num_stages - 1)
            
        else:
            stage_idx = 0
            
        stage_names = list(self.stages.keys())
        return stage_names[stage_idx]
        
    def compute_curriculum_weights(
        self,
        importance: torch.Tensor,
        stage: str,
    ) -> torch.Tensor:
        """
        Compute per-token curriculum weights based on current stage.
        
        Args:
            importance: Importance scores of shape (batch_size, seq_length)
            stage: Current curriculum stage
            
        Returns:
            Curriculum weights of shape (batch_size, seq_length)
        """
        stage_config = self.stages[stage]
        imp_min, imp_max = stage_config['importance_range']
        focus_weight = stage_config['focus_weight']
        
        # Compute weights based on importance range
        in_range = ((importance >= imp_min) & (importance <= imp_max)).float()
        
        # Tokens in the target range get higher weight
        weights = torch.ones_like(importance)
        weights = torch.where(in_range.bool(), 
                             weights * focus_weight,
                             weights)
        
        # Normalize weights
        weights = weights / weights.sum(dim=-1, keepdim=True)
        
        return weights
        
    def sample_curriculum_batch(
        self,
        x: torch.Tensor,
        importance: torch.Tensor,
        stage: str,
        num_samples: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample a batch according to curriculum strategy.
        
        Args:
            x: Input tokens of shape (batch_size, seq_length)
            importance: Importance scores
            stage: Current stage
            num_samples: Number of samples to select
            
        Returns:
            Sampled tokens and their importance scores
        """
        batch_size, seq_length = x.shape
        
        # Get curriculum weights
        weights = self.compute_curriculum_weights(importance, stage)
        
        # Sample indices according to weights
        sampled_indices = []
        for b in range(batch_size):
            indices = torch.multinomial(
                weights[b],
                num_samples=min(num_samples, seq_length),
                replacement=False,
            )
            sampled_indices.append(indices)
            
        sampled_indices = torch.stack(sampled_indices)
        
        # Gather sampled tokens
        sampled_x = torch.gather(x, 1, sampled_indices)
        sampled_importance = torch.gather(importance, 1, sampled_indices)
        
        return sampled_x, sampled_importance
        
    def get_stage_info(self, stage: str) -> Dict:
        """Get information about a curriculum stage."""
        return self.stages[stage]
        
    def get_progress(self, step: int) -> float:
        """Get training progress as a value in [0, 1]."""
        if step < self.warmup_steps:
            return 0.0
        progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        return np.clip(progress, 0.0, 1.0)


class DynamicCurriculumScheduler(CurriculumScheduler):
    """
    Enhanced curriculum scheduler that adapts based on model performance.
    
    If the model is struggling with certain difficulty levels, the curriculum
    can slow down or repeat stages.
    """
    
    def __init__(
        self,
        *args,
        performance_threshold: float = 0.8,
        adaptive: bool = True,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.performance_threshold = performance_threshold
        self.adaptive = adaptive
        
        # Track performance per stage
        self.stage_performance = {
            stage: [] for stage in self.stages.keys()
        }
        
    def update_performance(self, stage: str, loss: float):
        """
        Update performance tracking for a stage.
        
        Args:
            stage: Current stage
            loss: Current loss value
        """
        if stage in self.stage_performance:
            self.stage_performance[stage].append(loss)
            
            # Keep only recent history
            if len(self.stage_performance[stage]) > 100:
                self.stage_performance[stage] = self.stage_performance[stage][-100:]
                
    def should_advance_stage(self, current_stage: str) -> bool:
        """
        Determine if we should advance to the next curriculum stage.
        
        Args:
            current_stage: Current curriculum stage
            
        Returns:
            True if performance is good enough to advance
        """
        if not self.adaptive:
            return True
            
        if current_stage not in self.stage_performance:
            return True
            
        perf_history = self.stage_performance[current_stage]
        
        if len(perf_history) < 10:
            # Not enough data yet
            return True
            
        # Check if recent performance is improving
        recent_perf = perf_history[-10:]
        avg_recent = np.mean(recent_perf)
        avg_earlier = np.mean(perf_history[-20:-10]) if len(perf_history) >= 20 else avg_recent
        
        # Advance if performance is stable or improving
        return avg_recent <= avg_earlier * 1.1
        
    def get_curriculum_state(self) -> Dict:
        """Get current state of curriculum learning."""
        return {
            'stages': list(self.stages.keys()),
            'performance': {
                stage: np.mean(perf) if perf else None
                for stage, perf in self.stage_performance.items()
            },
            'adaptive': self.adaptive,
        }
