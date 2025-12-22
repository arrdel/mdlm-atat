#!/usr/bin/env python3
"""
MDLM ATAT Evaluation Script with WandB Integration

Evaluates a trained ATAT model on multiple datasets.
Computes perplexity and generates samples for analysis.

Usage:
    python scripts/eval_atat.py --checkpoint PATH [OPTIONS]

Required Arguments:
    --checkpoint PATH       Path to model checkpoint file

Evaluation Options:
    --mode MODE             Evaluation mode: perplexity, generate, or both [default: perplexity]
    --datasets NAMES        Comma-separated datasets [default: wikitext2,wikitext103,ptb_text_only]
    --num-samples N         Number of samples to generate [default: 10]
    --sample-length N       Length of generated samples [default: 256]

Data Options:
    --cache-dir PATH        Dataset cache directory [default: /media/scratch/adele/mdlm_fresh/data_cache]
    --log-dir PATH          Log directory [default: /media/scratch/adele/mdlm_fresh/logs]

Model Options:
    --config-name NAME      Model config [default: atat/tiny]
    --num-gpus N            Number of GPUs [default: 1]
    --batch-size N          Evaluation batch size [default: 85]

WandB Options:
    --wandb-project NAME    WandB project [default: mdlm-atat-eval]
    --wandb-run-name NAME   WandB run name [default: auto-generated]
    --wandb-offline         Run WandB offline

Examples:
    # Basic evaluation with perplexity
    python scripts/eval_atat.py --checkpoint /path/to/best.ckpt

    # Evaluate and generate samples
    python scripts/eval_atat.py --checkpoint /path/to/best.ckpt --mode both

    # Evaluate on specific dataset
    python scripts/eval_atat.py --checkpoint /path/to/best.ckpt --datasets wikitext2

    # Generate samples with custom settings
    python scripts/eval_atat.py --checkpoint /path/to/best.ckpt --mode generate \\
        --num-samples 20 --sample-length 512

    # Evaluate with importance analysis
    python scripts/eval_atat.py --checkpoint /path/to/best.ckpt --mode both \\
        --datasets wikitext2,ptb_text_only
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List

# Color codes
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    CYAN = '\033[0;36m'
    NC = '\033[0m'


def check_checkpoint(path: str) -> bool:
    """Check if checkpoint file exists."""
    return os.path.isfile(path)


def check_datasets(cache_dir: str) -> bool:
    """Check if datasets are available."""
    if not os.path.isdir(cache_dir):
        return False
    return len(os.listdir(cache_dir)) > 0


def print_header(message: str) -> None:
    """Print formatted header."""
    print(f"\n{Colors.BLUE}{'='*60}{Colors.NC}")
    print(f"{Colors.BLUE}{message.center(60)}{Colors.NC}")
    print(f"{Colors.BLUE}{'='*60}{Colors.NC}\n")


def evaluate_dataset(
    checkpoint_path: str,
    dataset_name: str,
    config_name: str,
    cache_dir: str,
    num_gpus: int,
    batch_size: int,
    mode: str,
    num_samples: int,
    sample_length: int,
    wandb_offline: bool,
    run_name: str,
    log_file: str
) -> int:
    """
    Run evaluation on a single dataset.
    
    Returns:
        Exit code from evaluation process
    """
    print(f"\n{Colors.CYAN}Evaluating on: {dataset_name}{Colors.NC}\n")
    
    # Change to project directory
    project_dir = Path(__file__).parent.parent
    os.chdir(project_dir)
    
    # Build command
    cmd = [
        "python", "../mdlm/main.py",
        "--config-path", "../mdlm_atat/configs",
        "--config-name", config_name,
        "mode=eval",
        f"eval.checkpoint_path={checkpoint_path}",
        f"data.valid={dataset_name}",
        f"data.cache_dir={cache_dir}",
        f"trainer.devices={num_gpus}",
        f"loader.eval_batch_size={batch_size}",
        f"wandb.offline={'true' if wandb_offline else 'false'}",
        f"wandb.name={run_name}_{dataset_name}",
    ]
    
    # Add mode-specific options
    if mode in ["perplexity", "both"]:
        cmd.extend([
            "eval.compute_generative_perplexity=true",
        ])
    
    if mode in ["generate", "both"]:
        cmd.extend([
            "eval.generate_samples=true",
            f"sampling.num_sample_batches={num_samples}",
            f"model.length={sample_length}",
        ])
    
    # Run evaluation
    with open(log_file, 'a') as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"Evaluating on: {dataset_name}\n")
        f.write(f"{'='*60}\n\n")
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        for line in process.stdout:
            print(line, end='')
            f.write(line)
            f.flush()
        
        exit_code = process.wait()
    
    return exit_code


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate MDLM ATAT model with WandB integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Required arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    
    # Evaluation options
    parser.add_argument(
        "--mode",
        type=str,
        default="perplexity",
        choices=["perplexity", "generate", "both"],
        help="Evaluation mode"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="wikitext2,wikitext103,ptb_text_only",
        help="Comma-separated dataset names"
    )
    parser.add_argument("--num-samples", type=int, default=10, help="Number of samples to generate")
    parser.add_argument("--sample-length", type=int, default=256, help="Sample length")
    
    # Data paths
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="/media/scratch/adele/mdlm_fresh/data_cache",
        help="Dataset cache directory"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="/media/scratch/adele/mdlm_fresh/logs",
        help="Log directory"
    )
    
    # Model options
    parser.add_argument("--config-name", type=str, default="atat/tiny", help="Model config")
    parser.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--batch-size", type=int, default=85, help="Batch size")
    
    # WandB options
    parser.add_argument("--wandb-project", type=str, default="mdlm-atat-eval", help="WandB project")
    parser.add_argument("--wandb-run-name", type=str, help="WandB run name")
    parser.add_argument("--wandb-offline", action="store_true", help="Run WandB offline")
    
    args = parser.parse_args()
    
    # Check checkpoint
    if not check_checkpoint(args.checkpoint):
        print(f"{Colors.RED}✗ Checkpoint not found: {args.checkpoint}{Colors.NC}")
        sys.exit(1)
    
    # Create log directory
    os.makedirs(args.log_dir, exist_ok=True)
    
    print_header("MDLM ATAT Evaluation with WandB")
    
    # Check datasets
    if not check_datasets(args.cache_dir):
        print(f"{Colors.RED}✗ Datasets not found in {args.cache_dir}{Colors.NC}")
        print(f"{Colors.YELLOW}Please run: python scripts/download_datasets.py{Colors.NC}")
        sys.exit(1)
    print(f"{Colors.GREEN}✓{Colors.NC} Datasets found")
    
    # Parse dataset list
    datasets = [d.strip() for d in args.datasets.split(",")]
    
    # Generate run name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.wandb_run_name or f"eval_{timestamp}"
    log_file = os.path.join(args.log_dir, f"eval_{timestamp}.log")
    
    # Print configuration
    print(f"\n{Colors.BLUE}Evaluation Configuration:{Colors.NC}")
    print(f"  Checkpoint: {Colors.GREEN}{args.checkpoint}{Colors.NC}")
    print(f"  Mode: {Colors.GREEN}{args.mode}{Colors.NC}")
    print(f"  Datasets: {Colors.GREEN}{', '.join(datasets)}{Colors.NC}")
    print(f"  Config: {Colors.GREEN}{args.config_name}{Colors.NC}")
    print(f"  GPUs: {Colors.GREEN}{args.num_gpus}{Colors.NC}")
    print(f"  Log file: {Colors.GREEN}{log_file}{Colors.NC}")
    
    if args.mode in ["generate", "both"]:
        print(f"  Num samples: {Colors.GREEN}{args.num_samples}{Colors.NC}")
        print(f"  Sample length: {Colors.GREEN}{args.sample_length}{Colors.NC}")
    
    print(f"\n{Colors.GREEN}Starting Evaluation...{Colors.NC}")
    
    # Evaluate each dataset
    failed_datasets = []
    for dataset in datasets:
        exit_code = evaluate_dataset(
            checkpoint_path=args.checkpoint,
            dataset_name=dataset,
            config_name=args.config_name,
            cache_dir=args.cache_dir,
            num_gpus=args.num_gpus,
            batch_size=args.batch_size,
            mode=args.mode,
            num_samples=args.num_samples,
            sample_length=args.sample_length,
            wandb_offline=args.wandb_offline,
            run_name=run_name,
            log_file=log_file
        )
        
        if exit_code != 0:
            failed_datasets.append(dataset)
            print(f"{Colors.RED}✗ Evaluation failed for {dataset}{Colors.NC}")
        else:
            print(f"{Colors.GREEN}✓ Evaluation completed for {dataset}{Colors.NC}")
    
    # Print summary
    print_header("Evaluation Complete")
    
    if failed_datasets:
        print(f"{Colors.YELLOW}Some evaluations failed:{Colors.NC}")
        for dataset in failed_datasets:
            print(f"  - {dataset}")
        print()
    else:
        print(f"{Colors.GREEN}All evaluations completed successfully!{Colors.NC}\n")
    
    print(f"Evaluation logs: {Colors.GREEN}{log_file}{Colors.NC}")
    print(f"WandB dashboard: {Colors.CYAN}https://wandb.ai{Colors.NC}\n")
    
    sys.exit(1 if failed_datasets else 0)


if __name__ == "__main__":
    main()
