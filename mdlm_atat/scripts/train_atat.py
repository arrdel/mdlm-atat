#!/usr/bin/env python3
"""
MDLM ATAT Training Script with WandB Integration

Trains the ATAT-enhanced MDLM model on text generation tasks.
Assumes datasets are already downloaded via download_datasets.py.

Usage:
    python scripts/train_atat.py [OPTIONS]

Required Configuration:
    --config-name NAME      Model config (atat/tiny, atat/small) [default: atat/tiny]

Training Options:
    --max-steps N           Maximum training steps [default: 100000]
    --val-interval N        Validation check interval [default: 5000]
    --log-interval N        Logging interval [default: 100]
    --num-gpus N            Number of GPUs to use [default: 2]
    --batch-size N          Per-GPU batch size [default: 85]
    --learning-rate LR      Learning rate [default: 0.0003]

Data Options:
    --cache-dir PATH        Dataset cache directory [default: /media/scratch/adele/mdlm_fresh/data_cache]
    --output-dir PATH       Output directory for checkpoints [default: /media/scratch/adele/mdlm_fresh/outputs]
    --log-dir PATH          Log directory [default: /media/scratch/adele/mdlm_fresh/logs]

WandB Options:
    --wandb-project NAME    WandB project name [default: mdlm-atat]
    --wandb-run-name NAME   WandB run name [default: auto-generated]
    --wandb-offline         Run WandB in offline mode

Other Options:
    --no-confirm            Skip confirmation prompt
    --resume PATH           Resume from checkpoint

Examples:
    # Basic training with tiny model
    python scripts/train_atat.py

    # Train small model with custom settings
    python scripts/train_atat.py --config-name atat/small --num-gpus 4 --batch-size 64

    # Resume training from checkpoint
    python scripts/train_atat.py --resume /path/to/checkpoint.ckpt

    # Train with custom learning rate and steps
    python scripts/train_atat.py --learning-rate 0.0001 --max-steps 200000

    # No confirmation prompt (for scripts)
    python scripts/train_atat.py --no-confirm
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Color codes
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    CYAN = '\033[0;36m'
    NC = '\033[0m'


def check_datasets(cache_dir: str) -> bool:
    """Check if datasets are downloaded."""
    if not os.path.isdir(cache_dir):
        return False
    return len(os.listdir(cache_dir)) > 0


def check_gpus() -> int:
    """Check available GPUs and return count."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--list-gpus"],
            capture_output=True,
            text=True,
            check=True
        )
        return len(result.stdout.strip().split('\n'))
    except Exception:
        return 0


def get_gpu_info() -> str:
    """Get formatted GPU information."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True
        )
        lines = result.stdout.strip().split('\n')
        formatted = []
        for line in lines:
            parts = line.split(', ')
            formatted.append(f"  GPU {parts[0]}: {parts[1]} ({parts[2]} total, {parts[3]} free)")
        return '\n'.join(formatted)
    except Exception:
        return "  Unable to query GPU info"


def check_wandb_login() -> bool:
    """Check if WandB is logged in."""
    try:
        result = subprocess.run(
            ["wandb", "status"],
            capture_output=True,
            text=True
        )
        return "Logged in" in result.stdout
    except Exception:
        return False


def print_header(message: str) -> None:
    """Print formatted header."""
    print(f"\n{Colors.BLUE}{'='*60}{Colors.NC}")
    print(f"{Colors.BLUE}{message.center(60)}{Colors.NC}")
    print(f"{Colors.BLUE}{'='*60}{Colors.NC}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Train MDLM ATAT model with WandB integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Model config
    parser.add_argument(
        "--config-name",
        type=str,
        default="atat/tiny",
        choices=[
            "atat/tiny", 
            "atat/small", 
            "atat/wikitext103_validation", 
            "atat/production_training",
            # Importance Ablation Study: Importance Estimator Variants
            "atat/importance_ablation_base",
            "atat/importance_ablation_full",
            "atat/importance_ablation_frequency_only",
            "atat/importance_ablation_learned_only",
            "atat/importance_ablation_uniform",
        ],
        help="Model configuration"
    )
    
    # Training parameters
    parser.add_argument("--max-steps", type=int, default=100000, help="Maximum training steps")
    parser.add_argument("--val-interval", type=int, default=5000, help="Validation interval")
    parser.add_argument("--log-interval", type=int, default=100, help="Logging interval")
    parser.add_argument("--num-gpus", type=int, default=2, help="Number of GPUs")
    parser.add_argument("--batch-size", type=int, default=4, help="Per-GPU batch size (4 for 6 GPUs)")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    
    # Data paths
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
        help="Output directory"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="/media/scratch/adele/mdlm_fresh/logs",
        help="Log directory"
    )
    
    # WandB options
    parser.add_argument("--wandb-project", type=str, default="mdlm-atat", help="WandB project")
    parser.add_argument("--wandb-run-name", type=str, help="WandB run name")
    parser.add_argument("--wandb-offline", action="store_true", help="Run WandB offline")
    
    # Other options
    parser.add_argument("--no-confirm", action="store_true", help="Skip confirmation")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    print_header("MDLM ATAT Training with WandB")
    
    # Check datasets
    if not check_datasets(args.cache_dir):
        print(f"{Colors.RED}✗ Datasets not found in {args.cache_dir}{Colors.NC}")
        print(f"{Colors.YELLOW}Please run: python scripts/download_datasets.py{Colors.NC}")
        sys.exit(1)
    print(f"{Colors.GREEN}✓{Colors.NC} Datasets found in: {args.cache_dir}")
    
    # Check WandB
    if not args.wandb_offline:
        print(f"\n{Colors.CYAN}Checking WandB authentication...{Colors.NC}")
        if not check_wandb_login():
            print(f"{Colors.YELLOW}WandB not logged in. Running in offline mode...{Colors.NC}")
            args.wandb_offline = True
        else:
            print(f"{Colors.GREEN}✓{Colors.NC} WandB authenticated")
    
    # Check GPUs
    print(f"\n{Colors.CYAN}Checking GPU availability...{Colors.NC}")
    gpu_count = check_gpus()
    if gpu_count == 0:
        print(f"{Colors.RED}✗ No GPUs found!{Colors.NC}")
        sys.exit(1)
    
    print(f"{Colors.GREEN}✓{Colors.NC} Found {gpu_count} GPUs")
    print(get_gpu_info())
    
    # Adjust GPU count if needed
    if args.num_gpus > gpu_count:
        print(f"{Colors.YELLOW}⚠ Requested {args.num_gpus} GPUs but only {gpu_count} available{Colors.NC}")
        print(f"{Colors.YELLOW}  Adjusting to use {gpu_count} GPUs{Colors.NC}")
        args.num_gpus = gpu_count
    
    # Generate run name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.wandb_run_name or f"atat_owt_{timestamp}"
    log_file = os.path.join(args.log_dir, f"training_{timestamp}.log")
    
    # Print configuration
    print(f"\n{Colors.BLUE}Training Configuration:{Colors.NC}")
    print(f"  Model: {Colors.GREEN}{args.config_name}{Colors.NC}")
    print(f"  Max steps: {Colors.GREEN}{args.max_steps}{Colors.NC}")
    print(f"  Validation interval: {Colors.GREEN}{args.val_interval}{Colors.NC}")
    print(f"  Batch size: {Colors.GREEN}{args.batch_size}{Colors.NC}")
    print(f"  Learning rate: {Colors.GREEN}{args.learning_rate}{Colors.NC}")
    print(f"  GPUs: {Colors.GREEN}{args.num_gpus}{Colors.NC}")
    print(f"  Log file: {Colors.GREEN}{log_file}{Colors.NC}")
    print(f"  Run name: {Colors.GREEN}{run_name}{Colors.NC}")
    
    if args.resume:
        print(f"  Resume from: {Colors.GREEN}{args.resume}{Colors.NC}")
    
    # Confirm start
    if not args.no_confirm:
        response = input(f"\n{Colors.YELLOW}Start training? [Y/n]: {Colors.NC}")
        if response.lower() not in ['', 'y', 'yes']:
            print(f"{Colors.RED}Training cancelled{Colors.NC}")
            sys.exit(0)
    
    print_header("Starting Training...")
    print(f"Monitor training:")
    print(f"  {Colors.CYAN}Logs:{Colors.NC}   tail -f {log_file}")
    print(f"  {Colors.CYAN}GPUs:{Colors.NC}   watch -n 1 nvidia-smi")
    print(f"  {Colors.CYAN}WandB:{Colors.NC}  https://wandb.ai\n")
    
    # Change to project directory
    project_dir = Path(__file__).parent.parent
    os.chdir(project_dir)
    
    # Build training command (use sys.executable to ensure same Python environment)
    # NOTE: get_dataloaders() uses torch.cuda.device_count() (ALL GPUs) not trainer.devices
    # So global_batch_size must account for ALL available GPUs, not just trainer.devices
    num_total_gpus = check_gpus()
    global_batch_size = args.batch_size * 1 * num_total_gpus * 1  # num_nodes=1, accumulation=1, using ALL GPUs
    
    cmd = [
        sys.executable, "../mdlm/main.py",
        "--config-path", "../mdlm_atat/configs",
        "--config-name", args.config_name,
        f"trainer.max_steps={args.max_steps}",
        f"trainer.val_check_interval={args.val_interval}",
        f"trainer.log_every_n_steps={args.log_interval}",
        "trainer.accelerator=cuda",
        f"trainer.devices={args.num_gpus}",
        f"loader.batch_size={args.batch_size}",
        f"loader.global_batch_size={global_batch_size}",  # Use ALL GPUs in calculation
        f"optim.lr={args.learning_rate}",
        f"data.cache_dir={args.cache_dir}",
        f"hydra.run.dir={args.output_dir}/${{data.train}}/${{now:%Y.%m.%d}}/${{now:%H%M%S}}",
    ]
    
    if args.resume:
        cmd.append(f"eval.checkpoint_path={args.resume}")
    
    # Run training with tee to log file
    with open(log_file, 'w') as f:
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
    
    # Print completion message
    print_header("Training Complete" if exit_code == 0 else "Training Failed")
    
    if exit_code == 0:
        print(f"{Colors.GREEN}Training completed successfully!{Colors.NC}\n")
    else:
        print(f"{Colors.RED}Training failed with exit code: {exit_code}{Colors.NC}\n")
    
    print(f"Training logs: {Colors.GREEN}{log_file}{Colors.NC}")
    print(f"Output directory: {Colors.GREEN}{args.output_dir}{Colors.NC}")
    print(f"WandB dashboard: {Colors.CYAN}https://wandb.ai{Colors.NC}\n")
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
# TODO: Find small dataset instead of using synthetic
# TODO: Find other methods for importance estimation
# TODO: Find how shifting from using diffusion models for image generation to text generation has an impact on the project
# TODO: Find a specific problem you are solving and define it mathematically