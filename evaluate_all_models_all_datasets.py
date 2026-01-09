#!/usr/bin/env python3
"""
Comprehensive Evaluation Script - All Models on All Datasets

Evaluates all baseline models (AR, D3PM, SEDD, MDLM) and ATAT model on:
- PTB (Penn Treebank)
- WikiText-103
- LAMBADA
- AG News
- LM1B

Features:
- Test mode: Evaluate on small patches first (100-500 samples)
- Full mode: Evaluate entire datasets
- Progress tracking and result logging
- JSON output with detailed metrics

Usage:
    # Test mode (recommended first)
    python evaluate_all_models_all_datasets.py --test-mode
    
    # Full evaluation
    python evaluate_all_models_all_datasets.py --full-eval
    
    # Specific models/datasets
    python evaluate_all_models_all_datasets.py --models mdlm,atat --datasets wikitext103,ptb
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np
from tqdm import tqdm

# Color codes
class Colors:
    BLUE = '\033[0;34m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    RED = '\033[0;31m'
    CYAN = '\033[0;36m'
    NC = '\033[0m'

def print_header(message: str):
    """Print formatted header."""
    print(f"\n{Colors.BLUE}{'='*80}{Colors.NC}")
    print(f"{Colors.BLUE}{message.center(80)}{Colors.NC}")
    print(f"{Colors.BLUE}{'='*80}{Colors.NC}\n")

def print_section(message: str):
    """Print formatted section."""
    print(f"\n{Colors.CYAN}>>> {message}{Colors.NC}")

# ============================================================================
# DATASET LOADERS
# ============================================================================

class DatasetLoader:
    """Load datasets from cache directory."""
    
    def __init__(self, cache_dir: str = "/media/scratch/adele/mdlm_fresh/data_cache"):
        self.cache_dir = Path(cache_dir)
        
    def load_ptb(self, split: str = "train", max_samples: Optional[int] = None) -> List[str]:
        """Load PTB dataset."""
        ptb_dir = self.cache_dir / "ptb"
        if split == "train":
            file = ptb_dir / "train.txt"
        elif split == "validation":
            file = ptb_dir / "validation.txt"
        elif split == "test":
            file = ptb_dir / "test.txt"
        else:
            raise ValueError(f"Unknown split: {split}")
        
        if not file.exists():
            raise FileNotFoundError(f"PTB file not found: {file}")
        
        with open(file, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        if max_samples:
            lines = lines[:max_samples]
        
        return lines
    
    def load_wikitext103(self, split: str = "train", max_samples: Optional[int] = None) -> List[str]:
        """Load WikiText-103 dataset."""
        wt_dir = self.cache_dir / "wikitext103"
        if split == "train":
            file = wt_dir / "train.txt"
        elif split == "validation":
            file = wt_dir / "validation.txt"
        elif split == "test":
            file = wt_dir / "test.txt"
        else:
            raise ValueError(f"Unknown split: {split}")
        
        if not file.exists():
            raise FileNotFoundError(f"WikiText-103 file not found: {file}")
        
        with open(file, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        if max_samples:
            lines = lines[:max_samples]
        
        return lines
    
    def load_lambada(self, split: str = "train", max_samples: Optional[int] = None) -> List[str]:
        """Load LAMBADA dataset."""
        lambada_dir = self.cache_dir / "lambada"
        if split == "train":
            file = lambada_dir / "train.txt"
        elif split == "test":
            file = lambada_dir / "test.txt"
        else:
            raise ValueError(f"Unknown split: {split}")
        
        if not file.exists():
            raise FileNotFoundError(f"LAMBADA file not found: {file}")
        
        with open(file, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        if max_samples:
            lines = lines[:max_samples]
        
        return lines
    
    def load_ag_news(self, split: str = "train", max_samples: Optional[int] = None) -> List[str]:
        """Load AG News dataset."""
        ag_dir = self.cache_dir / "ag_news"
        if split == "train":
            file = ag_dir / "train.txt"
        elif split == "test":
            file = ag_dir / "test.txt"
        else:
            raise ValueError(f"Unknown split: {split}")
        
        if not file.exists():
            raise FileNotFoundError(f"AG News file not found: {file}")
        
        with open(file, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        if max_samples:
            lines = lines[:max_samples]
        
        return lines
    
    def load_lm1b(self, split: str = "train", max_samples: Optional[int] = None) -> List[str]:
        """Load LM1B dataset."""
        lm1b_dir = self.cache_dir / "lm1b"
        
        if split == "train":
            # LM1B training data is in news.en-* files
            file = lm1b_dir / "train.txt"
            if not file.exists():
                # Try to load from extracted archive
                data_files = sorted(lm1b_dir.glob("1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/news.en-*"))
                if not data_files:
                    raise FileNotFoundError(f"LM1B training files not found in {lm1b_dir}")
                
                lines = []
                for data_file in data_files[:10]:  # Load first 10 shards
                    try:
                        with open(data_file, 'r') as f:
                            lines.extend([line.strip() for line in f if line.strip()])
                        if max_samples and len(lines) >= max_samples:
                            break
                    except Exception as e:
                        print(f"  {Colors.YELLOW}⚠ Error reading {data_file}: {e}{Colors.NC}")
                        continue
                
                return lines[:max_samples] if max_samples else lines
            
            with open(file, 'r') as f:
                lines = [line.strip() for line in f if line.strip()]
        else:
            raise ValueError(f"LM1B only has 'train' split, got: {split}")
        
        if max_samples:
            lines = lines[:max_samples]
        
        return lines

# ============================================================================
# MODEL LOADERS
# ============================================================================

class ModelEvaluator:
    """Evaluate models on text data."""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = None
        self.tokenizer = None
    
    def load_ar_model(self, checkpoint_path: str):
        """Load AR Transformer model."""
        try:
            from mdlm.models import ARTransformer
            from mdlm.utils import get_tokenizer
            
            self.tokenizer = get_tokenizer()
            self.model = ARTransformer.load_from_checkpoint(checkpoint_path)
            self.model = self.model.to(self.device)
            self.model.eval()
            
            print(f"  {Colors.GREEN}✓{Colors.NC} AR model loaded from {Path(checkpoint_path).name}")
            return True
        except Exception as e:
            print(f"  {Colors.RED}✗ Error loading AR model: {e}{Colors.NC}")
            return False
    
    def load_d3pm_model(self, checkpoint_path: str):
        """Load D3PM model."""
        try:
            from mdlm_atat.models import D3PMWrapper
            from mdlm.utils import get_tokenizer
            
            self.tokenizer = get_tokenizer()
            self.model = D3PMWrapper.load_from_checkpoint(checkpoint_path)
            self.model = self.model.to(self.device)
            self.model.eval()
            
            print(f"  {Colors.GREEN}✓{Colors.NC} D3PM model loaded from {Path(checkpoint_path).name}")
            return True
        except Exception as e:
            print(f"  {Colors.RED}✗ Error loading D3PM model: {e}{Colors.NC}")
            return False
    
    def load_sedd_model(self, checkpoint_path: str):
        """Load SEDD model."""
        try:
            from mdlm_atat.models import SEDDWrapper
            from mdlm.utils import get_tokenizer
            
            self.tokenizer = get_tokenizer()
            self.model = SEDDWrapper.load_from_checkpoint(checkpoint_path)
            self.model = self.model.to(self.device)
            self.model.eval()
            
            print(f"  {Colors.GREEN}✓{Colors.NC} SEDD model loaded from {Path(checkpoint_path).name}")
            return True
        except Exception as e:
            print(f"  {Colors.RED}✗ Error loading SEDD model: {e}{Colors.NC}")
            return False
    
    def load_mdlm_model(self, checkpoint_path: str):
        """Load MDLM model."""
        try:
            from mdlm_atat.models import MDLMWrapper
            from mdlm.utils import get_tokenizer
            
            self.tokenizer = get_tokenizer()
            self.model = MDLMWrapper.load_from_checkpoint(checkpoint_path)
            self.model = self.model.to(self.device)
            self.model.eval()
            
            print(f"  {Colors.GREEN}✓{Colors.NC} MDLM model loaded from {Path(checkpoint_path).name}")
            return True
        except Exception as e:
            print(f"  {Colors.RED}✗ Error loading MDLM model: {e}{Colors.NC}")
            return False
    
    def load_atat_model(self, checkpoint_path: str):
        """Load ATAT model."""
        try:
            from mdlm_atat.models import ATATWrapper
            from mdlm.utils import get_tokenizer
            
            self.tokenizer = get_tokenizer()
            self.model = ATATWrapper.load_from_checkpoint(checkpoint_path)
            self.model = self.model.to(self.device)
            self.model.eval()
            
            print(f"  {Colors.GREEN}✓{Colors.NC} ATAT model loaded from {Path(checkpoint_path).name}")
            return True
        except Exception as e:
            print(f"  {Colors.RED}✗ Error loading ATAT model: {e}{Colors.NC}")
            return False
    
    def evaluate_batch(self, texts: List[str], batch_size: int = 16) -> Dict[str, float]:
        """Evaluate model on a batch of texts."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded")
        
        total_loss = 0.0
        total_tokens = 0
        
        try:
            with torch.no_grad():
                for i in tqdm(range(0, len(texts), batch_size), desc="  Evaluating"):
                    batch = texts[i:i+batch_size]
                    
                    # Tokenize
                    tokens = self.tokenizer.encode(batch)
                    if len(tokens) == 0:
                        continue
                    
                    # Convert to tensor
                    token_ids = torch.tensor(tokens).to(self.device)
                    
                    # Forward pass
                    outputs = self.model(token_ids)
                    loss = outputs['loss']
                    
                    total_loss += loss.item() * len(tokens)
                    total_tokens += len(tokens)
            
            avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
            perplexity = np.exp(avg_loss)
            
            return {
                'avg_loss': avg_loss,
                'perplexity': perplexity,
                'total_tokens': total_tokens,
                'num_samples': len(texts)
            }
        
        except Exception as e:
            print(f"  {Colors.RED}✗ Error during evaluation: {e}{Colors.NC}")
            return {
                'avg_loss': float('nan'),
                'perplexity': float('nan'),
                'total_tokens': 0,
                'num_samples': len(texts),
                'error': str(e)
            }

# ============================================================================
# CHECKPOINT FINDER
# ============================================================================

class CheckpointFinder:
    """Find latest checkpoints for all models."""
    
    @staticmethod
    def find_latest_checkpoint(pattern: str, base_path: str = "/media/scratch/adele/mdlm_fresh/outputs") -> Optional[str]:
        """Find the latest checkpoint matching pattern."""
        base = Path(base_path)
        
        # Search for checkpoint files
        candidates = []
        for p in base.glob(f"**/{pattern}"):
            if p.is_file():
                candidates.append(p)
        
        if not candidates:
            return None
        
        # Sort by modification time
        latest = max(candidates, key=lambda p: p.stat().st_mtime)
        return str(latest)
    
    @staticmethod
    def get_model_checkpoints() -> Dict[str, str]:
        """Get all available model checkpoints."""
        checkpoints = {}
        base = Path("/media/scratch/adele/mdlm_fresh/outputs")
        
        # AR Transformer - best perplexity checkpoint
        ar_files = list(base.glob("baselines/ar_transformer/ar-step=*/*.ckpt"))
        if ar_files:
            # Find the one with best (lowest) perplexity
            best = max(ar_files, key=lambda p: p.stat().st_mtime)
            checkpoints['ar'] = str(best)
        
        # D3PM - latest checkpoint
        d3pm_files = list(base.glob("baselines/d3pm_small/*.ckpt"))
        if d3pm_files:
            latest = max(d3pm_files, key=lambda p: p.stat().st_mtime)
            checkpoints['d3pm'] = str(latest)
        
        # SEDD - latest checkpoint (prefer the step=10000 checkpoint)
        sedd_files = list(base.glob("baselines/sedd_small/*.ckpt"))
        if sedd_files:
            # Prefer sedd-step=010000.ckpt if it exists
            step_10k = [f for f in sedd_files if "010000" in str(f)]
            if step_10k:
                checkpoints['sedd'] = str(step_10k[0])
            else:
                latest = max(sedd_files, key=lambda p: p.stat().st_mtime)
                checkpoints['sedd'] = str(latest)
        
        # MDLM - latest checkpoint (prefer 0-10000.ckpt)
        mdlm_files = list(base.glob("baselines/mdlm_uniform/checkpoints/*.ckpt"))
        if mdlm_files:
            # Prefer 0-10000.ckpt if it exists
            step_10k = [f for f in mdlm_files if "10000" in str(f)]
            if step_10k:
                checkpoints['mdlm'] = str(step_10k[0])
            else:
                latest = max(mdlm_files, key=lambda p: p.stat().st_mtime)
                checkpoints['mdlm'] = str(latest)
        
        # ATAT - latest checkpoint
        atat_files = list(base.glob("checkpoints/atat_production*.ckpt"))
        if atat_files:
            latest = max(atat_files, key=lambda p: p.stat().st_mtime)
            checkpoints['atat'] = str(latest)
        
        return checkpoints

# ============================================================================
# MAIN EVALUATION RUNNER
# ============================================================================

class EvaluationRunner:
    """Orchestrate full evaluation pipeline."""
    
    def __init__(self, test_mode: bool = True, output_dir: str = "/media/scratch/adele/mdlm_fresh/outputs/comprehensive_eval"):
        self.test_mode = test_mode
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.dataset_loader = DatasetLoader()
        self.checkpoint_finder = CheckpointFinder()
        
        # Test mode sample sizes
        self.test_samples = {
            'ptb': 100,
            'wikitext103': 200,
            'lambada': 150,
            'ag_news': 100,
            'lm1b': 200
        }
        
        self.results = {}
    
    def run_evaluation(self, models: Optional[List[str]] = None, 
                      datasets: Optional[List[str]] = None) -> Dict:
        """Run full evaluation pipeline."""
        
        print_header("COMPREHENSIVE MODEL EVALUATION")
        print(f"  Mode: {Colors.CYAN}{'TEST' if self.test_mode else 'FULL'}{Colors.NC}")
        print(f"  Output: {Colors.CYAN}{self.output_dir}{Colors.NC}")
        print(f"  Time: {Colors.CYAN}{datetime.now().isoformat()}{Colors.NC}\n")
        
        # Default to all if not specified
        if models is None:
            models = ['ar', 'd3pm', 'sedd', 'mdlm', 'atat']
        if datasets is None:
            datasets = ['ptb', 'wikitext103', 'lambada', 'ag_news', 'lm1b']
        
        # Get checkpoints
        print_section("Finding Model Checkpoints")
        checkpoints = self.checkpoint_finder.get_model_checkpoints()
        
        for model, ckpt in checkpoints.items():
            if model in models:
                print(f"  {Colors.GREEN}✓{Colors.NC} {model.upper()}: {Path(ckpt).name}")
        
        # Evaluate each model on each dataset
        print_section("Starting Evaluation")
        
        for model in models:
            if model not in checkpoints:
                print(f"  {Colors.YELLOW}⚠ Checkpoint not found for {model}{Colors.NC}")
                continue
            
            print(f"\n{Colors.CYAN}Model: {model.upper()}{Colors.NC}")
            evaluator = ModelEvaluator()
            
            # Load model
            ckpt = checkpoints[model]
            if model == 'ar':
                if not evaluator.load_ar_model(ckpt):
                    continue
            elif model == 'd3pm':
                if not evaluator.load_d3pm_model(ckpt):
                    continue
            elif model == 'sedd':
                if not evaluator.load_sedd_model(ckpt):
                    continue
            elif model == 'mdlm':
                if not evaluator.load_mdlm_model(ckpt):
                    continue
            elif model == 'atat':
                if not evaluator.load_atat_model(ckpt):
                    continue
            
            self.results[model] = {}
            
            # Evaluate on each dataset
            for dataset in datasets:
                print(f"  Dataset: {dataset.upper()}")
                
                try:
                    # Load dataset
                    if dataset == 'ptb':
                        texts = self.dataset_loader.load_ptb(
                            max_samples=self.test_samples['ptb'] if self.test_mode else None
                        )
                    elif dataset == 'wikitext103':
                        texts = self.dataset_loader.load_wikitext103(
                            max_samples=self.test_samples['wikitext103'] if self.test_mode else None
                        )
                    elif dataset == 'lambada':
                        texts = self.dataset_loader.load_lambada(
                            max_samples=self.test_samples['lambada'] if self.test_mode else None
                        )
                    elif dataset == 'ag_news':
                        texts = self.dataset_loader.load_ag_news(
                            max_samples=self.test_samples['ag_news'] if self.test_mode else None
                        )
                    elif dataset == 'lm1b':
                        texts = self.dataset_loader.load_lm1b(
                            max_samples=self.test_samples['lm1b'] if self.test_mode else None
                        )
                    else:
                        print(f"    {Colors.RED}✗ Unknown dataset{Colors.NC}")
                        continue
                    
                    if not texts:
                        print(f"    {Colors.RED}✗ No texts loaded{Colors.NC}")
                        continue
                    
                    print(f"    Loaded {len(texts)} samples")
                    
                    # Evaluate
                    metrics = evaluator.evaluate_batch(texts)
                    self.results[model][dataset] = metrics
                    
                    print(f"    {Colors.GREEN}✓{Colors.NC} Loss: {metrics['avg_loss']:.4f}, PPL: {metrics['perplexity']:.2f}")
                
                except Exception as e:
                    print(f"    {Colors.RED}✗ Error: {e}{Colors.NC}")
                    self.results[model][dataset] = {'error': str(e)}
        
        # Save results
        print_section("Saving Results")
        self._save_results()
        
        return self.results
    
    def _save_results(self):
        """Save evaluation results to JSON."""
        results_file = self.output_dir / f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"  {Colors.GREEN}✓{Colors.NC} Results saved: {results_file}")
        
        # Print summary table
        print_section("Evaluation Summary")
        print(f"  {'Model':<12} {'Dataset':<15} {'Perplexity':<15} {'Loss':<15}")
        print(f"  {'-'*57}")
        
        for model, datasets in self.results.items():
            for dataset, metrics in datasets.items():
                if 'error' in metrics:
                    print(f"  {model:<12} {dataset:<15} {'ERROR':<15} {'-':<15}")
                else:
                    ppl = metrics.get('perplexity', float('nan'))
                    loss = metrics.get('avg_loss', float('nan'))
                    print(f"  {model:<12} {dataset:<15} {ppl:<15.2f} {loss:<15.4f}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive evaluation on all models and datasets"
    )
    parser.add_argument("--test-mode", action="store_true", default=True,
                       help="Run on small patches first (default: True)")
    parser.add_argument("--full-eval", action="store_true",
                       help="Run on full datasets (overrides test-mode)")
    parser.add_argument("--models", type=str, default="ar,d3pm,sedd,mdlm,atat",
                       help="Comma-separated list of models to evaluate")
    parser.add_argument("--datasets", type=str, default="ptb,wikitext103,lambada,ag_news,lm1b",
                       help="Comma-separated list of datasets to evaluate")
    parser.add_argument("--output-dir", type=str,
                       default="/media/scratch/adele/mdlm_fresh/outputs/comprehensive_eval",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    # Parse models and datasets
    models = [m.strip() for m in args.models.split(',')]
    datasets = [d.strip() for d in args.datasets.split(',')]
    
    # Create runner
    runner = EvaluationRunner(
        test_mode=not args.full_eval,
        output_dir=args.output_dir
    )
    
    # Run evaluation
    results = runner.run_evaluation(models=models, datasets=datasets)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
