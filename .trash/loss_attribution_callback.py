"""
Training Callback for Loss Attribution Tracking

Integrates with PyTorch Lightning training to track loss by token frequency
during training. This provides real-time monitoring of the claim:
"40% of training loss comes from high-frequency function words"

Usage:
    Add to your Lightning trainer:
    
    from mdlm_atat.utils.loss_attribution_callback import LossAttributionCallback
    
    callback = LossAttributionCallback(
        tokenizer_name='gpt2',
        log_every_n_steps=1000,
        num_batches_per_analysis=10
    )
    
    trainer = Trainer(
        callbacks=[callback, ...],
        ...
    )
"""

import torch
import torch.nn.functional as F
from collections import Counter, defaultdict
from pathlib import Path
import json
import numpy as np

try:
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import Callback
except ImportError:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import Callback

from transformers import GPT2Tokenizer


class LossAttributionCallback(Callback):
    """
    PyTorch Lightning callback to track loss attribution by token frequency.
    
    Logs metrics to WandB/TensorBoard showing what % of loss comes from
    high/medium/low frequency tokens during training.
    """
    
    def __init__(
        self,
        tokenizer_name: str = 'gpt2',
        log_every_n_steps: int = 1000,
        num_batches_per_analysis: int = 10,
        high_freq_pct: float = 0.1,
        medium_freq_pct: float = 0.4,
        freq_bins_path: str = None
    ):
        super().__init__()
        
        self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)
        self.log_every_n_steps = log_every_n_steps
        self.num_batches_per_analysis = num_batches_per_analysis
        self.high_freq_pct = high_freq_pct
        self.medium_freq_pct = medium_freq_pct
        
        self.freq_bins = None
        self.freq_bins_path = freq_bins_path
        self.token_counts = Counter()
        
        # Storage for analysis
        self.batch_buffer = []
        self.analysis_history = []
        
    def on_train_start(self, trainer, pl_module):
        """Initialize frequency bins from training data."""
        
        if self.freq_bins_path and Path(self.freq_bins_path).exists():
            # Load pre-computed bins
            print(f"Loading frequency bins from: {self.freq_bins_path}")
            with open(self.freq_bins_path, 'r') as f:
                data = json.load(f)
                self.freq_bins = {
                    k: set(v) for k, v in data['frequency_bins'].items()
                }
        else:
            # Compute from training data
            print("Computing token frequency bins from training data...")
            self._compute_frequency_bins(trainer)
            
            # Save for future use
            if self.freq_bins_path:
                self._save_frequency_bins(self.freq_bins_path)
    
    def _compute_frequency_bins(self, trainer, max_batches: int = 500):
        """Compute frequency bins from training dataloader."""
        
        dataloader = trainer.train_dataloader
        
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break
            
            # Extract tokens
            if isinstance(batch, dict):
                tokens = batch['input_ids']
            elif isinstance(batch, (tuple, list)):
                tokens = batch[0]
            else:
                tokens = batch
            
            # Count
            self.token_counts.update(tokens.flatten().cpu().tolist())
        
        # Categorize
        sorted_tokens = sorted(
            self.token_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        n = len(sorted_tokens)
        high_cutoff = int(self.high_freq_pct * n)
        med_cutoff = int((self.high_freq_pct + self.medium_freq_pct) * n)
        
        self.freq_bins = {'high': set(), 'medium': set(), 'low': set()}
        
        for i, (token_id, count) in enumerate(sorted_tokens):
            if i < high_cutoff:
                self.freq_bins['high'].add(token_id)
            elif i < med_cutoff:
                self.freq_bins['medium'].add(token_id)
            else:
                self.freq_bins['low'].add(token_id)
        
        print(f"Frequency bins computed:")
        print(f"  High: {len(self.freq_bins['high'])} tokens")
        print(f"  Medium: {len(self.freq_bins['medium'])} tokens")
        print(f"  Low: {len(self.freq_bins['low'])} tokens")
    
    def _save_frequency_bins(self, path: str):
        """Save frequency bins to JSON."""
        data = {
            'frequency_bins': {k: list(v) for k, v in self.freq_bins.items()},
            'top_1000_tokens': dict(self.token_counts.most_common(1000))
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved frequency bins to: {path}")
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Collect batches for periodic analysis."""
        
        if self.freq_bins is None:
            return
        
        # Store batch for analysis
        self.batch_buffer.append((batch, outputs))
        
        # Analyze periodically
        current_step = trainer.global_step
        
        if (current_step > 0 and 
            current_step % self.log_every_n_steps == 0 and
            len(self.batch_buffer) >= self.num_batches_per_analysis):
            
            self._analyze_and_log(trainer, pl_module, current_step)
            self.batch_buffer = []
    
    def _analyze_and_log(self, trainer, pl_module, step):
        """Analyze loss attribution and log metrics."""
        
        pl_module.eval()
        
        loss_by_bin = defaultdict(float)
        count_by_bin = defaultdict(int)
        
        device = next(pl_module.parameters()).device
        
        with torch.no_grad():
            for batch, _ in self.batch_buffer[:self.num_batches_per_analysis]:
                
                # Extract input
                if isinstance(batch, dict):
                    x_0 = batch['input_ids'].to(device)
                elif isinstance(batch, (tuple, list)):
                    x_0 = batch[0].to(device)
                else:
                    x_0 = batch.to(device)
                
                B, L = x_0.shape
                
                # Sample timestep and create masked version
                t = torch.rand(B, device=device)
                mask_prob = t.unsqueeze(1).expand(B, L)
                mask_decisions = torch.rand_like(mask_prob) < mask_prob
                
                x_t = x_0.clone()
                mask_index = 50256  # GPT-2 mask token
                x_t[mask_decisions] = mask_index
                
                # Model forward
                try:
                    logits = pl_module(x_t, t)
                except:
                    try:
                        logits = pl_module.model(x_t, t)
                    except:
                        continue
                
                # Compute per-token losses
                losses = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    x_0.reshape(-1),
                    reduction='none'
                )
                
                # Attribute to frequency bins
                for i, token_id in enumerate(x_0.reshape(-1)):
                    tid = token_id.item()
                    loss_val = losses[i].item()
                    
                    # Get bin
                    if tid in self.freq_bins['high']:
                        bin_name = 'high'
                    elif tid in self.freq_bins['medium']:
                        bin_name = 'medium'
                    else:
                        bin_name = 'low'
                    
                    loss_by_bin[bin_name] += loss_val
                    count_by_bin[bin_name] += 1
        
        pl_module.train()
        
        # Compute statistics
        total_loss = sum(loss_by_bin.values())
        total_tokens = sum(count_by_bin.values())
        
        if total_loss == 0:
            return
        
        # Compute metrics
        metrics = {}
        for bin_name in ['high', 'medium', 'low']:
            if count_by_bin[bin_name] > 0:
                loss_contrib = loss_by_bin[bin_name]
                loss_pct = (loss_contrib / total_loss) * 100
                token_pct = (count_by_bin[bin_name] / total_tokens) * 100
                avg_loss = loss_contrib / count_by_bin[bin_name]
                
                metrics[f'loss_attribution/{bin_name}_freq_loss_pct'] = loss_pct
                metrics[f'loss_attribution/{bin_name}_freq_token_pct'] = token_pct
                metrics[f'loss_attribution/{bin_name}_freq_avg_loss'] = avg_loss
        
        # Log to trainer
        pl_module.log_dict(metrics, on_step=True, on_epoch=False, sync_dist=True)
        
        # Store for later analysis
        self.analysis_history.append({
            'step': step,
            'metrics': metrics
        })
        
        # Print summary
        high_loss_pct = metrics.get('loss_attribution/high_freq_loss_pct', 0)
        print(f"\n[Step {step}] High-frequency tokens: {high_loss_pct:.1f}% of loss")
    
    def on_train_end(self, trainer, pl_module):
        """Save final analysis."""
        
        if self.analysis_history:
            output_path = Path(trainer.log_dir) / 'loss_attribution_history.json'
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(self.analysis_history, f, indent=2)
            
            print(f"\nSaved loss attribution history to: {output_path}")
            
            # Print final summary
            if len(self.analysis_history) > 0:
                final = self.analysis_history[-1]['metrics']
                print("\n" + "="*70)
                print("FINAL LOSS ATTRIBUTION")
                print("="*70)
                
                for bin_name in ['high', 'medium', 'low']:
                    loss_key = f'loss_attribution/{bin_name}_freq_loss_pct'
                    if loss_key in final:
                        print(f"{bin_name.upper():8s}: {final[loss_key]:6.2f}% of loss")
                
                print("="*70 + "\n")


class LossAttributionLogger:
    """
    Standalone logger for loss attribution (non-Lightning).
    
    Can be used in standard PyTorch training loops.
    """
    
    def __init__(self, tokenizer_name: str = 'gpt2'):
        self.callback = LossAttributionCallback(tokenizer_name=tokenizer_name)
        self.callback.freq_bins = None
    
    def compute_frequency_bins(self, dataloader, max_batches: int = 500):
        """Compute frequency bins from data."""
        print("Computing token frequencies...")
        
        token_counts = Counter()
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break
            
            if isinstance(batch, dict):
                tokens = batch['input_ids']
            elif isinstance(batch, (tuple, list)):
                tokens = batch[0]
            else:
                tokens = batch
            
            token_counts.update(tokens.flatten().tolist())
        
        # Create bins
        sorted_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
        n = len(sorted_tokens)
        
        high_cutoff = int(0.1 * n)
        med_cutoff = int(0.5 * n)
        
        freq_bins = {'high': set(), 'medium': set(), 'low': set()}
        
        for i, (token_id, count) in enumerate(sorted_tokens):
            if i < high_cutoff:
                freq_bins['high'].add(token_id)
            elif i < med_cutoff:
                freq_bins['medium'].add(token_id)
            else:
                freq_bins['low'].add(token_id)
        
        self.callback.freq_bins = freq_bins
        print(f"Bins: High={len(freq_bins['high'])}, "
              f"Med={len(freq_bins['medium'])}, "
              f"Low={len(freq_bins['low'])}")
        
        return freq_bins
    
    def analyze_batch(self, model, batch, device='cuda'):
        """Analyze a single batch."""
        
        if self.callback.freq_bins is None:
            raise ValueError("Must compute frequency bins first")
        
        model.eval()
        
        # Extract input
        if isinstance(batch, dict):
            x_0 = batch['input_ids'].to(device)
        elif isinstance(batch, (tuple, list)):
            x_0 = batch[0].to(device)
        else:
            x_0 = batch.to(device)
        
        B, L = x_0.shape
        
        # Create masked version
        t = torch.rand(B, device=device)
        mask_prob = t.unsqueeze(1).expand(B, L)
        mask_decisions = torch.rand_like(mask_prob) < mask_prob
        
        x_t = x_0.clone()
        x_t[mask_decisions] = 50256
        
        # Forward
        with torch.no_grad():
            logits = model(x_t, t)
        
        # Losses
        losses = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            x_0.reshape(-1),
            reduction='none'
        )
        
        # Attribute
        loss_by_bin = defaultdict(float)
        count_by_bin = defaultdict(int)
        
        for i, token_id in enumerate(x_0.reshape(-1)):
            tid = token_id.item()
            loss_val = losses[i].item()
            
            if tid in self.callback.freq_bins['high']:
                bin_name = 'high'
            elif tid in self.callback.freq_bins['medium']:
                bin_name = 'medium'
            else:
                bin_name = 'low'
            
            loss_by_bin[bin_name] += loss_val
            count_by_bin[bin_name] += 1
        
        # Compute percentages
        total_loss = sum(loss_by_bin.values())
        
        results = {}
        for bin_name in ['high', 'medium', 'low']:
            if count_by_bin[bin_name] > 0:
                results[bin_name] = {
                    'loss_pct': (loss_by_bin[bin_name] / total_loss) * 100,
                    'avg_loss': loss_by_bin[bin_name] / count_by_bin[bin_name]
                }
        
        model.train()
        return results
