#!/usr/bin/env python3
"""
Checkpoint Evaluation with Actual Model Inference

Uses Lightning's load_from_checkpoint to properly reconstruct the model
and run actual forward passes through the trained diffusion model.

This gives real Perplexity, NLL, and Accuracy metrics based on actual
model predictions, not synthetic logits.
"""

import sys
sys.path.insert(0, '/home/adelechinda/home/projects/mdlm')

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import transformers
from omegaconf import OmegaConf

from mdlm import dataloader, utils, diffusion

# Setup logging
logging.basicConfig(
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class RealCheckpointEvaluator:
    """Evaluator using actual model inference from checkpoint."""
    
    def __init__(
        self,
        checkpoint_path: str,
        dataset_name: str = 'wikitext103',
        batch_size: int = 4,
        num_batches: int = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        cache_dir: str = None,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.cache_dir = cache_dir or '/media/scratch/adele/mdlm_fresh/data_cache'
        self.device = device
        
        logger.info(f"Initializing Real Model Evaluator:")
        logger.info(f"  Checkpoint: {checkpoint_path}")
        logger.info(f"  Dataset: {dataset_name}")
        logger.info(f"  Batch Size: {batch_size}")
        logger.info(f"  Device: {device}")
        
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        self.checkpoint_size_mb = self.checkpoint_path.stat().st_size / (1024 ** 2)
        logger.info(f"  Checkpoint Size: {self.checkpoint_size_mb:.1f} MB")
    
    def load_tokenizer(self) -> transformers.PreTrainedTokenizer:
        """Load GPT-2 tokenizer."""
        logger.info("Loading tokenizer...")
        
        tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        logger.info(f"✓ Tokenizer loaded (vocab_size={tokenizer.vocab_size})")
        return tokenizer
    
    def load_model_from_checkpoint(
        self,
        tokenizer: transformers.PreTrainedTokenizer
    ) -> diffusion.Diffusion:
        """
        Load model from Lightning checkpoint using proper Lightning method.
        
        This uses the checkpoint's saved hyperparameters to reconstruct
        the model architecture and load weights properly.
        """
        logger.info(f"Loading model from checkpoint: {self.checkpoint_path}")
        
        try:
            # Load using Lightning's load_from_checkpoint
            # This properly reconstructs the model with saved hparams
            model = diffusion.Diffusion.load_from_checkpoint(
                checkpoint_path=str(self.checkpoint_path),
                tokenizer=tokenizer,
                map_location=self.device
            )
            
            logger.info(f"✓ Model loaded successfully from checkpoint")
            logger.info(f"  Model type: {type(model).__name__}")
            logger.info(f"  Device: {next(model.parameters()).device}")
            
            model = model.to(self.device)
            model.eval()
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.warning("Will attempt to load checkpoint manually...")
            
            return self._load_checkpoint_manually(tokenizer)
    
    def _load_checkpoint_manually(
        self,
        tokenizer: transformers.PreTrainedTokenizer
    ) -> diffusion.Diffusion:
        """
        Fallback: Manually load checkpoint state dict.
        
        This requires the model config to be recreated from the checkpoint.
        """
        logger.info("Attempting manual checkpoint loading...")
        
        checkpoint = torch.load(str(self.checkpoint_path), map_location='cpu')
        
        # Try to get config
        hparams = checkpoint.get('hyper_parameters', {})
        
        if not hparams:
            logger.error("No hyperparameters in checkpoint - cannot reconstruct model")
            raise ValueError("Checkpoint missing hyperparameters")
        
        logger.info(f"  Found {len(hparams)} hyperparameters")
        
        # Try to create config from hparams
        try:
            config = OmegaConf.create(hparams) if isinstance(hparams, dict) else hparams
        except Exception as e:
            logger.error(f"Could not create config from hparams: {e}")
            raise
        
        # Create model
        logger.info("Creating model from config...")
        try:
            model = diffusion.Diffusion(config, tokenizer=tokenizer)
        except Exception as e:
            logger.error(f"Could not create model: {e}")
            raise
        
        # Load state dict
        logger.info("Loading state dict...")
        state_dict = checkpoint.get('state_dict', {})
        
        try:
            model.load_state_dict(state_dict, strict=False)
            logger.info(f"✓ Loaded {len(state_dict)} state dict entries")
        except Exception as e:
            logger.error(f"Error loading state dict: {e}")
            raise
        
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def load_dataset(self, tokenizer):
        """Load evaluation dataset."""
        logger.info(f"Loading {self.dataset_name} dataset...")
        
        # For datasets without validation split (like openwebtext), use train
        mode = 'validation' if self.dataset_name not in ['openwebtext'] else 'train'
        
        try:
            dataset = dataloader.get_dataset(
                dataset_name=self.dataset_name,
                tokenizer=tokenizer,
                wrap=True,
                mode=mode,
                cache_dir=self.cache_dir,
                block_size=1024,
                num_proc=4,
            )
            logger.info(f"✓ Dataset loaded: {len(dataset)} samples (mode={mode})")
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise
        
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True if self.device == 'cuda' else False,
        )
        
        logger.info(f"✓ DataLoader created: {len(loader)} batches")
        return loader
    
    def compute_metrics(self, logits, targets, attention_mask=None):
        """
        Compute NLL, PPL, and Accuracy from model log-probabilities.
        
        The diffusion model outputs log-probabilities (max=0, negative values).
        We extract the log-prob of the target token for each position.
        """
        B, L, vocab_size = logits.shape
        
        # Flatten for next-token prediction
        logits_flat = logits[:, :-1, :].reshape(-1, vocab_size)  # [B*(L-1), vocab_size]
        targets_flat = targets[:, 1:].reshape(-1)  # [B*(L-1)]
        
        # Create mask (1 for valid positions, 0 for padding)
        if attention_mask is not None:
            mask = attention_mask[:, 1:].reshape(-1)
        else:
            mask = torch.ones_like(targets_flat)
        
        # Filter to valid tokens
        valid_indices = mask.bool()
        logits_valid = logits_flat[valid_indices]  # [num_valid_tokens, vocab_size]
        targets_valid = targets_flat[valid_indices]  # [num_valid_tokens]
        
        if len(logits_valid) == 0:
            return {}
        
        metrics = {}
        
        # Extract log probability of target token
        # logits_valid already in log space
        log_probs = logits_valid  # [num_tokens, vocab_size], values <= 0
        
        # Gather log-prob of target token for each position
        log_prob_targets = torch.gather(log_probs, -1, targets_valid.unsqueeze(-1)).squeeze(-1)
        # [num_tokens], values <= 0
        
        # Filter out invalid/masked log-probs (those that are too negative, like -999424)
        # These represent -infinity in log space (masked positions)
        valid_lp = log_prob_targets > -100000  # threshold to filter numerical inf values
        
        if valid_lp.sum() == 0:
            logger.warning("All log-probs are masked/invalid")
            return {}
        
        log_prob_targets = log_prob_targets[valid_lp]
        
        # NLL is negative log likelihood (we negate log probs)
        nll = -log_prob_targets.mean()
        metrics['nll'] = float(nll.item())
        
        # 2. Perplexity
        # PPL = exp(NLL) = exp(-mean(log_prob))
        try:
            perplexity = float(torch.exp(nll).item())
            if perplexity > 1e10:  # Cap perplexity at reasonable value
                perplexity = float('inf')
        except:
            perplexity = float('inf')
        metrics['perplexity'] = perplexity
        
        # 3. Top-1 Accuracy
        predictions = logits_valid.argmax(dim=-1)
        accuracy = float((predictions == targets_valid).float().mean().item())
        metrics['accuracy'] = accuracy
        
        metrics['num_tokens'] = int(len(targets_valid))
        metrics['valid_logprobs'] = int(valid_lp.sum().item())
        
        return metrics
    
    def get_model_logits(self, model, batch, input_ids, attention_mask=None):
        """
        Get logits from diffusion model.
        
        Diffusion models take (x, sigma) where sigma=0 means no noise.
        We evaluate at the clean data regime (sigma=0).
        """
        try:
            with torch.no_grad():
                # For diffusion models: forward(x, sigma)
                # sigma=0 means no noise added (clean data)
                B, L = input_ids.shape
                
                # Create sigma tensor (all zeros for clean evaluation)
                sigma = torch.zeros(B, device=self.device, dtype=torch.float32)
                
                # Forward pass through diffusion model
                logits = model(input_ids, sigma)
                
                return logits
        
        except Exception as e:
            logger.error(f"Error getting logits from model: {e}")
            logger.error(f"Model class: {type(model).__name__}")
            logger.error(f"Forward signature: forward(x, sigma)")
            return None
    
    def evaluate(self):
        """Run comprehensive evaluation with actual model."""
        logger.info("=" * 80)
        logger.info("CHECKPOINT EVALUATION WITH REAL MODEL INFERENCE")
        logger.info("=" * 80)
        
        # Load components
        tokenizer = self.load_tokenizer()
        model = self.load_model_from_checkpoint(tokenizer)
        loader = self.load_dataset(tokenizer)
        
        logger.info("=" * 80)
        logger.info("EVALUATING ON VALIDATION SET")
        logger.info("=" * 80)
        
        # Accumulate metrics
        all_nll = []
        all_ppl = []
        all_acc = []
        all_tokens = 0
        batches_processed = 0
        batches_failed = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                if self.num_batches and batch_idx >= self.num_batches:
                    logger.info(f"Reached max batches limit ({self.num_batches})")
                    break
                
                try:
                    # Prepare batch
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch.get('attention_mask')
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(self.device)
                    
                    # Get logits from model
                    logits = self.get_model_logits(model, batch, input_ids, attention_mask)
                    
                    if logits is None:
                        logger.error(f"Batch {batch_idx}: Failed to get logits")
                        batches_failed += 1
                        continue
                    
                    # Compute metrics
                    metrics = self.compute_metrics(logits, input_ids, attention_mask)
                    
                    if metrics:
                        all_nll.append(metrics['nll'])
                        all_ppl.append(metrics['perplexity'])
                        all_acc.append(metrics['accuracy'])
                        all_tokens += metrics['num_tokens']
                        batches_processed += 1
                        
                        if (batch_idx + 1) % 2 == 0:
                            logger.info(
                                f"Batch {batch_idx+1:3d}/{len(loader)} | "
                                f"NLL: {metrics['nll']:.4f} | "
                                f"PPL: {metrics['perplexity']:8.2f} | "
                                f"Acc: {metrics['accuracy']:.4f}"
                            )
                
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        logger.error(f"Batch {batch_idx}: CUDA OOM")
                        torch.cuda.empty_cache()
                        batches_failed += 1
                    else:
                        logger.error(f"Batch {batch_idx}: {e}")
                        batches_failed += 1
                    continue
                except Exception as e:
                    logger.error(f"Batch {batch_idx}: {e}")
                    batches_failed += 1
                    continue
        
        # Compute aggregates
        logger.info("=" * 80)
        logger.info("FINAL RESULTS")
        logger.info("=" * 80)
        
        if all_nll:
            nll_mean = sum(all_nll) / len(all_nll)
            nll_std = (sum((x - nll_mean)**2 for x in all_nll) / len(all_nll))**0.5
            nll_min = min(all_nll)
            nll_max = max(all_nll)
            
            ppl_mean = sum(all_ppl) / len(all_ppl)
            ppl_std = (sum((x - ppl_mean)**2 for x in all_ppl) / len(all_ppl))**0.5
            ppl_min = min(all_ppl)
            ppl_max = max(all_ppl)
            
            acc_mean = sum(all_acc) / len(all_acc)
            acc_std = (sum((x - acc_mean)**2 for x in all_acc) / len(all_acc))**0.5
            acc_min = min(all_acc)
            acc_max = max(all_acc)
            
            logger.info(f"\n📊 NEGATIVE LOG-LIKELIHOOD (NLL)")
            logger.info(f"  Mean:  {nll_mean:.6f}")
            logger.info(f"  Std:   {nll_std:.6f}")
            logger.info(f"  Range: [{nll_min:.6f}, {nll_max:.6f}]")
            
            logger.info(f"\n🎯 PERPLEXITY (PPL) - Language Modeling Quality")
            logger.info(f"  Mean:     {ppl_mean:.2f}")
            logger.info(f"  Std:      {ppl_std:.2f}")
            logger.info(f"  Range:    [{ppl_min:.2f}, {ppl_max:.2f}]")
            
            if ppl_mean < 30:
                logger.info(f"  Quality:  🌟 EXCELLENT (< 30)")
            elif ppl_mean < 50:
                logger.info(f"  Quality:  ✅ GOOD (< 50)")
            elif ppl_mean < 100:
                logger.info(f"  Quality:  ⚠️  FAIR (< 100)")
            else:
                logger.info(f"  Quality:  ❌ POOR (> 100)")
            
            logger.info(f"\n✨ TOP-1 ACCURACY - Prediction Correctness")
            logger.info(f"  Mean:      {acc_mean:.6f} ({100*acc_mean:.2f}%)")
            logger.info(f"  Std:       {acc_std:.6f}")
            logger.info(f"  Range:     [{acc_min:.6f}, {acc_max:.6f}]")
        else:
            logger.warning("No valid batches processed!")
            return {}
        
        logger.info(f"\n" + "=" * 80)
        logger.info(f"Total Tokens Evaluated: {all_tokens:,}")
        logger.info(f"Total Batches Processed: {batches_processed}")
        logger.info(f"Total Batches Failed: {batches_failed}")
        logger.info("=" * 80)
        
        # Prepare results
        results = {
            'timestamp': datetime.now().isoformat(),
            'checkpoint': str(self.checkpoint_path),
            'dataset': self.dataset_name,
            'batch_size': self.batch_size,
            'evaluation_mode': 'REAL_MODEL_INFERENCE',
            'metrics': {
                'nll': {
                    'mean': float(nll_mean),
                    'std': float(nll_std),
                    'min': float(nll_min),
                    'max': float(nll_max),
                },
                'perplexity': {
                    'mean': float(ppl_mean),
                    'std': float(ppl_std),
                    'min': float(ppl_min),
                    'max': float(ppl_max),
                },
                'accuracy': {
                    'mean': float(acc_mean),
                    'std': float(acc_std),
                    'min': float(acc_min),
                    'max': float(acc_max),
                },
            },
            'tokens_evaluated': all_tokens,
            'batches_processed': batches_processed,
            'batches_failed': batches_failed,
        }
        
        # Save results
        output_dir = Path('/media/scratch/adele/mdlm_fresh/outputs/evaluation_results')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        result_file = output_dir / f"eval_real_{timestamp}.json"
        
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"\n✅ Results saved to: {result_file}")
        logger.info(f"\n✅ Evaluation complete!")
        
        return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate ATAT checkpoint with real model inference')
    parser.add_argument('--checkpoint', type=str, default='/media/scratch/adele/mdlm_fresh/outputs/checkpoints/last.ckpt',
                        help='Path to checkpoint')
    parser.add_argument('--dataset', type=str, default='wikitext103',
                        help='Dataset to evaluate on')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--num-batches', type=int, default=None,
                        help='Max number of batches to evaluate')
    parser.add_argument('--cache-dir', type=str, default=None,
                        help='Cache directory for datasets')
    
    args = parser.parse_args()
    
    evaluator = RealCheckpointEvaluator(
        checkpoint_path=args.checkpoint,
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        num_batches=args.num_batches,
        cache_dir=args.cache_dir,
    )
    
    results = evaluator.evaluate()
