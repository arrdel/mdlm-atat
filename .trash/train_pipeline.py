#!/usr/bin/env python3
"""
Run Complete ATAT Pipeline on Different Model Sizes

Python wrapper for running training, evaluation, and ablation studies.
Supports tiny, small, and medium models with various datasets.

Usage:
    # Small model on OpenWebText
    python scripts/train_pipeline.py --model small --dataset openwebtext
    
    # Medium model on WikiText-103
    python scripts/train_pipeline.py --model medium --dataset wikitext103 --gpus 2
    
    # Tiny model on synthetic data (for testing)
    python scripts/train_pipeline.py --model tiny --dataset synthetic_tiny --gpus 2

Model Sizes:
    tiny:   33.8M params  (256 hidden, 8 blocks)   - Fast testing
    small:  150M params   (512 hidden, 12 blocks)  - Standard experiments  
    medium: 355M params   (1024 hidden, 24 blocks) - Large-scale experiments

Datasets:
    synthetic_tiny: Small synthetic dataset for pipeline testing
    openwebtext:    Large web corpus (8M samples)
    wikitext103:    Wikipedia text (28K articles)
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


# Model configurations
MODEL_CONFIGS = {
    "tiny": {
        "params": "33.8M",
        "hidden_size": 256,
        "n_blocks": 8,
        "max_steps": 10000,
        "val_interval": 500,
        "config_name": "atat/tiny",
    },
    "small": {
        "params": "~150M",
        "hidden_size": 512,
        "n_blocks": 12,
        "max_steps": 100000,
        "val_interval": 5000,
        "config_name": "atat/small",
    },
    "medium": {
        "params": "~355M",
        "hidden_size": 1024,
        "n_blocks": 24,
        "max_steps": 200000,
        "val_interval": 10000,
        "config_name": "atat/medium",
    },
}

# Dataset configurations
DATASET_CONFIGS = {
    "synthetic_tiny": {
        "train": "synthetic_tiny",
        "valid": "synthetic_tiny",
        "override_steps": 200,
        "override_val_interval": 50,
    },
    "openwebtext": {
        "train": "openwebtext",
        "valid": "wikitext103",
    },
    "wikitext103": {
        "train": "wikitext103",
        "valid": "wikitext103",
    },
}


def print_header(message):
    """Print formatted header."""
    print(f"\n{'='*70}")
    print(f"{message.center(70)}")
    print(f"{'='*70}\n")


def run_training(args, model_cfg, dataset_cfg, log_file):
    """Run model training."""
    print_header(f"STEP 1/3: TRAINING {args.model.upper()} MODEL")
    
    # Override steps for synthetic data
    max_steps = dataset_cfg.get("override_steps", model_cfg["max_steps"])
    val_interval = dataset_cfg.get("override_val_interval", model_cfg["val_interval"])
    
    cmd = [
        "python", "../mdlm/main.py",
        "--config-path", "../mdlm_atat/configs",
        "--config-name", model_cfg["config_name"],
        f"data.train={dataset_cfg['train']}",
        f"data.valid={dataset_cfg['valid']}",
        f"trainer.devices={args.gpus}",
        f"trainer.max_steps={max_steps}",
        f"trainer.val_check_interval={val_interval}",
        f"wandb.name={args.run_name}",
        f"wandb.offline={'true' if args.wandb_offline else 'false'}",
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print(f"Max steps: {max_steps}")
    print(f"Val interval: {val_interval}\n")
    
    with open(log_file, 'a') as f:
        f.write(f"\n{'='*70}\n")
        f.write(f"TRAINING {args.model.upper()} MODEL\n")
        f.write(f"{'='*70}\n\n")
        
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
    
    return exit_code, max_steps


def find_checkpoint(output_dir, max_age_minutes=120):
    """Find the most recent checkpoint."""
    import time
    current_time = time.time()
    cutoff_time = current_time - (max_age_minutes * 60)
    
    checkpoints = []
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            if file == "best.ckpt":
                filepath = os.path.join(root, file)
                mtime = os.path.getmtime(filepath)
                if mtime >= cutoff_time:
                    checkpoints.append((filepath, mtime))
    
    if checkpoints:
        # Return most recent checkpoint
        checkpoints.sort(key=lambda x: x[1], reverse=True)
        return checkpoints[0][0]
    
    return None


def run_evaluation(args, model_cfg, dataset_cfg, checkpoint, log_file):
    """Run model evaluation."""
    print_header("STEP 2/3: EVALUATION")
    
    cmd = [
        "python", "scripts/eval_atat.py",
        "--checkpoint", checkpoint,
        "--mode", "both",
        "--datasets", dataset_cfg['valid'],
        "--num-samples", "20",
        "--config-name", model_cfg["config_name"],
        "--num-gpus", str(args.gpus),
        "--cache-dir", args.cache_dir,
        "--log-dir", args.log_dir,
    ]
    
    if args.wandb_offline:
        cmd.append("--wandb-offline")
    
    print(f"Command: {' '.join(cmd)}\n")
    
    with open(log_file, 'a') as f:
        f.write(f"\n{'='*70}\n")
        f.write(f"EVALUATION\n")
        f.write(f"{'='*70}\n\n")
        
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


def run_ablations(args, model_cfg, max_steps, log_file):
    """Run ablation studies."""
    print_header("STEP 3/3: ABLATION STUDIES")
    
    # Scale ablation steps (typically 5-10% of main training)
    ablation_steps = max(50, max_steps // 20)
    ablation_val_interval = max(25, ablation_steps // 4)
    
    cmd = [
        "python", "scripts/run_ablation.py",
        "--study", "all",
        "--max-steps", str(ablation_steps),
        "--val-interval", str(ablation_val_interval),
        "--num-gpus", str(args.gpus),
        "--output-dir", f"{args.output_dir}/ablations",
        "--log-dir", args.log_dir,
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print(f"Ablation steps: {ablation_steps}")
    print(f"Val interval: {ablation_val_interval}\n")
    
    with open(log_file, 'a') as f:
        f.write(f"\n{'='*70}\n")
        f.write(f"ABLATION STUDIES\n")
        f.write(f"{'='*70}\n\n")
        
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
        description="Run complete ATAT pipeline on different model sizes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Model and dataset
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(MODEL_CONFIGS.keys()),
        help="Model size to train"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="openwebtext",
        choices=list(DATASET_CONFIGS.keys()),
        help="Dataset to use for training"
    )
    
    # Hardware
    parser.add_argument(
        "--gpus",
        type=int,
        default=2,
        help="Number of GPUs to use"
    )
    
    # Directories
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="/media/scratch/adele/mdlm_fresh/data_cache",
        help="Dataset cache directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/media/scratch/adele/mdlm_fresh/outputs",
        help="Output directory for checkpoints"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="/media/scratch/adele/mdlm_fresh/logs",
        help="Log directory"
    )
    
    # WandB
    parser.add_argument(
        "--wandb-offline",
        action="store_true",
        help="Run WandB in offline mode"
    )
    parser.add_argument(
        "--run-name",
        type=str,
        help="Custom run name (auto-generated if not specified)"
    )
    
    # Pipeline control
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training step (use existing checkpoint)"
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Skip evaluation step"
    )
    parser.add_argument(
        "--skip-ablations",
        action="store_true",
        help="Skip ablation studies"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Use specific checkpoint (for skipping training)"
    )
    
    args = parser.parse_args()
    
    # Get configurations
    model_cfg = MODEL_CONFIGS[args.model]
    dataset_cfg = DATASET_CONFIGS[args.dataset]
    
    # Generate run name if not provided
    if not args.run_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_name = f"atat_{args.model}_{args.dataset}_{timestamp}"
    
    # Setup directories
    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    log_file = os.path.join(args.log_dir, f"pipeline_{args.run_name}.log")
    
    # Change to project directory
    project_dir = Path(__file__).parent.parent
    os.chdir(project_dir)
    
    # Print configuration
    print_header(f"ATAT FULL PIPELINE - {args.model.upper()} MODEL")
    
    print("Configuration:")
    print(f"  Model: {args.model} ({model_cfg['params']} parameters)")
    print(f"  Dataset: {args.dataset} (train={dataset_cfg['train']}, valid={dataset_cfg['valid']})")
    print(f"  GPUs: {args.gpus}")
    print(f"  Run name: {args.run_name}")
    print(f"  Log file: {log_file}")
    print()
    
    # Initialize results
    results = {
        "training": None,
        "evaluation": None,
        "ablations": None,
    }
    checkpoint = args.checkpoint
    max_steps = model_cfg["max_steps"]
    
    # Step 1: Training
    if not args.skip_training:
        train_exit_code, max_steps = run_training(args, model_cfg, dataset_cfg, log_file)
        results["training"] = train_exit_code == 0
        
        if train_exit_code != 0:
            print(f"\n✗ Training failed with exit code {train_exit_code}")
            sys.exit(train_exit_code)
        
        # Find checkpoint
        checkpoint = find_checkpoint(args.output_dir)
        if not checkpoint:
            print("\n✗ Could not find checkpoint")
            sys.exit(1)
        
        print(f"\n✓ Training completed")
        print(f"  Checkpoint: {checkpoint}\n")
    elif not checkpoint:
        print("\n✗ Must provide --checkpoint when using --skip-training")
        sys.exit(1)
    
    # Step 2: Evaluation
    if not args.skip_evaluation:
        eval_exit_code = run_evaluation(args, model_cfg, dataset_cfg, checkpoint, log_file)
        results["evaluation"] = eval_exit_code == 0
        
        if eval_exit_code != 0:
            print(f"\n⚠ Evaluation failed with exit code {eval_exit_code}")
            print("  Continuing to ablation studies...\n")
    
    # Step 3: Ablation Studies
    if not args.skip_ablations:
        ablation_exit_code = run_ablations(args, model_cfg, max_steps, log_file)
        results["ablations"] = ablation_exit_code == 0
        
        if ablation_exit_code != 0:
            print(f"\n⚠ Ablation studies failed with exit code {ablation_exit_code}\n")
    
    # Print summary
    print_header("PIPELINE COMPLETE")
    
    print("Results:")
    if results["training"] is not None:
        print(f"  Training: {'✓' if results['training'] else '✗'}")
    if results["evaluation"] is not None:
        print(f"  Evaluation: {'✓' if results['evaluation'] else '✗'}")
    if results["ablations"] is not None:
        print(f"  Ablations: {'✓' if results['ablations'] else '✗'}")
    
    print(f"\nFiles:")
    print(f"  Checkpoint: {checkpoint}")
    print(f"  Full log: {log_file}")
    print()
    
    # Exit with error if any step failed
    if any(v is False for v in results.values() if v is not None):
        sys.exit(1)


if __name__ == "__main__":
    main()
