#!/usr/bin/env python3
"""
MDLM ATAT Quick Start - Interactive Setup Wizard

Interactive script to guide through the complete MDLM ATAT workflow.

Usage:
    python quickstart.py [OPTIONS]

Options:
    --skip-datasets         Skip dataset download (assumes already downloaded)
    --skip-training         Skip training step
    --auto-yes              Automatically answer yes to all prompts
    --config-name NAME      Model config to use [default: atat/tiny]

Examples:
    # Interactive full workflow
    python quickstart.py

    # Skip dataset download (already downloaded)
    python quickstart.py --skip-datasets

    # Non-interactive mode (auto-approve all steps)
    python quickstart.py --auto-yes

    # Use small model config
    python quickstart.py --config-name atat/small
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

# Color codes
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    CYAN = '\033[0;36m'
    MAGENTA = '\033[0;35m'
    NC = '\033[0m'


def clear_screen():
    """Clear terminal screen."""
    os.system('clear' if os.name != 'nt' else 'cls')


def print_banner():
    """Print welcome banner."""
    banner = """
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║          MDLM ATAT - Quick Start Guide                   ║
║                                                           ║
║  Adaptive Token-Aware Training for Masked Diffusion      ║
║                  Language Models                          ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
"""
    print(f"{Colors.MAGENTA}{banner}{Colors.NC}")


def print_section_header(title: str):
    """Print section header."""
    print(f"\n{Colors.BLUE}{'='*60}{Colors.NC}")
    print(f"{Colors.YELLOW}{title}{Colors.NC}")
    print(f"{Colors.BLUE}{'='*60}{Colors.NC}")


def check_prerequisites() -> bool:
    """Check if prerequisites are met."""
    print(f"{Colors.CYAN}Checking prerequisites...{Colors.NC}")
    
    # Check conda environment
    try:
        result = subprocess.run(
            ["conda", "info", "--envs"],
            capture_output=True,
            text=True
        )
        if "mdlm-atat" not in result.stdout:
            print(f"{Colors.RED}✗ Conda environment 'mdlm-atat' not found!{Colors.NC}")
            print(f"{Colors.YELLOW}Please create it first:{Colors.NC}")
            print("  cd /home/adelechinda/home/projects/mdlm/mdlm")
            print("  conda env create -f requirements.yaml")
            return False
        print(f"{Colors.GREEN}✓{Colors.NC} Conda environment found")
    except Exception:
        print(f"{Colors.RED}✗ Conda not found!{Colors.NC}")
        return False
    
    # Check GPUs
    try:
        result = subprocess.run(
            ["nvidia-smi", "--list-gpus"],
            capture_output=True,
            text=True,
            check=True
        )
        gpu_count = len(result.stdout.strip().split('\n'))
        print(f"{Colors.GREEN}✓{Colors.NC} Found {gpu_count} GPUs")
    except Exception:
        print(f"{Colors.YELLOW}⚠ No GPUs detected!{Colors.NC}")
        if not confirm("Continue anyway?", default=False):
            return False
    
    return True


def confirm(message: str, default: bool = True, auto_yes: bool = False) -> bool:
    """Ask for user confirmation."""
    if auto_yes:
        return True
    
    prompt = f"{Colors.YELLOW}{message} [{'Y/n' if default else 'y/N'}]: {Colors.NC}"
    response = input(prompt).strip().lower()
    
    if not response:
        return default
    return response in ['y', 'yes']


def check_datasets(cache_dir: str) -> bool:
    """Check if datasets are downloaded."""
    if not os.path.isdir(cache_dir):
        return False
    return len(os.listdir(cache_dir)) > 0


def get_cache_size(cache_dir: str) -> str:
    """Get cache directory size."""
    try:
        result = subprocess.run(
            ["du", "-sh", cache_dir],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.split()[0]
    except Exception:
        return "Unknown"


def download_datasets(cache_dir: str, auto_yes: bool = False) -> bool:
    """Download datasets using the Python script."""
    project_dir = Path(__file__).parent
    download_script = project_dir / "scripts" / "download_datasets.py"
    
    if not download_script.exists():
        print(f"{Colors.RED}✗ Download script not found!{Colors.NC}")
        return False
    
    cmd = ["python", str(download_script), "--cache-dir", cache_dir]
    
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError:
        print(f"{Colors.RED}✗ Dataset download failed!{Colors.NC}")
        return False


def start_training(config_name: str, auto_yes: bool = False) -> bool:
    """Start training using the Python script."""
    project_dir = Path(__file__).parent
    train_script = project_dir / "scripts" / "train_atat.py"
    
    if not train_script.exists():
        print(f"{Colors.RED}✗ Training script not found!{Colors.NC}")
        return False
    
    cmd = [
        "python", str(train_script),
        "--config-name", config_name,
    ]
    
    if auto_yes:
        cmd.append("--no-confirm")
    
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError:
        print(f"{Colors.RED}✗ Training failed or was cancelled!{Colors.NC}")
        return False


def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    """Find the latest checkpoint."""
    try:
        result = subprocess.run(
            ["find", output_dir, "-name", "best.ckpt", "-type", "f"],
            capture_output=True,
            text=True
        )
        
        checkpoints = [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
        if checkpoints:
            return checkpoints[0]  # Return first found
        return None
    except Exception:
        return None


def evaluate_checkpoint(checkpoint_path: str) -> bool:
    """Evaluate checkpoint using the Python script."""
    project_dir = Path(__file__).parent
    eval_script = project_dir / "scripts" / "eval_atat.py"
    
    if not eval_script.exists():
        print(f"{Colors.RED}✗ Evaluation script not found!{Colors.NC}")
        return False
    
    cmd = [
        "python", str(eval_script),
        "--checkpoint", checkpoint_path,
        "--mode", "both"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError:
        print(f"{Colors.RED}✗ Evaluation failed!{Colors.NC}")
        return False


def print_next_steps(scratch_dir: str):
    """Print helpful next steps."""
    print(f"\n{Colors.GREEN}{'='*60}{Colors.NC}")
    print(f"{Colors.GREEN}{'Quick Start Complete!'.center(60)}{Colors.NC}")
    print(f"{Colors.GREEN}{'='*60}{Colors.NC}\n")
    
    print(f"{Colors.BLUE}Next Steps:{Colors.NC}\n")
    
    print(f"{Colors.CYAN}Monitor Training:{Colors.NC}")
    print(f"  {Colors.YELLOW}tail -f {scratch_dir}/logs/training_*.log{Colors.NC}")
    print(f"  {Colors.YELLOW}watch -n 1 nvidia-smi{Colors.NC}")
    print(f"  {Colors.YELLOW}https://wandb.ai{Colors.NC}\n")
    
    print(f"{Colors.CYAN}Evaluate Model:{Colors.NC}")
    print(f"  {Colors.YELLOW}python scripts/eval_atat.py --checkpoint /path/to/best.ckpt{Colors.NC}\n")
    
    print(f"{Colors.CYAN}Documentation:{Colors.NC}")
    print(f"  {Colors.YELLOW}cat WORKFLOW_README.md{Colors.NC}      # Full workflow guide")
    print(f"  {Colors.YELLOW}cat RESTRUCTURING_SUMMARY.md{Colors.NC} # Project structure\n")
    
    print(f"{Colors.MAGENTA}Happy training! 🚀{Colors.NC}\n")


def main():
    parser = argparse.ArgumentParser(
        description="MDLM ATAT Quick Start - Interactive Setup Wizard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument("--skip-datasets", action="store_true", help="Skip dataset download")
    parser.add_argument("--skip-training", action="store_true", help="Skip training step")
    parser.add_argument("--auto-yes", action="store_true", help="Auto-approve all steps")
    parser.add_argument("--config-name", type=str, default="atat/tiny", help="Model config")
    parser.add_argument(
        "--scratch-dir",
        type=str,
        default="/media/scratch/adele/mdlm_fresh",
        help="Scratch directory"
    )
    
    args = parser.parse_args()
    
    cache_dir = os.path.join(args.scratch_dir, "data_cache")
    output_dir = os.path.join(args.scratch_dir, "outputs")
    
    # Clear screen and show banner
    clear_screen()
    print_banner()
    
    print(f"{Colors.BLUE}This script will guide you through:{Colors.NC}")
    print("  1. Downloading datasets (one-time)")
    print("  2. Training the model")
    print("  3. Evaluating results\n")
    
    # Check prerequisites
    if not check_prerequisites():
        sys.exit(1)
    
    # Step 1: Datasets
    if not args.skip_datasets:
        print_section_header("Step 1: Dataset Management")
        
        if check_datasets(cache_dir):
            cache_size = get_cache_size(cache_dir)
            print(f"{Colors.GREEN}✓{Colors.NC} Datasets found in cache ({cache_size})")
            
            if confirm("Re-download datasets?", default=False, auto_yes=args.auto_yes):
                if not download_datasets(cache_dir, args.auto_yes):
                    sys.exit(1)
            else:
                print(f"{Colors.GREEN}Using cached datasets{Colors.NC}")
        else:
            print(f"{Colors.YELLOW}No cached datasets found. Downloading now...{Colors.NC}")
            print(f"{Colors.CYAN}This will download ~40GB and may take 30-60 minutes.{Colors.NC}\n")
            
            if confirm("Continue?", auto_yes=args.auto_yes):
                if not download_datasets(cache_dir, args.auto_yes):
                    sys.exit(1)
            else:
                print(f"{Colors.RED}Datasets required for training. Exiting.{Colors.NC}")
                sys.exit(1)
    
    # Step 2: Training
    training_done = False
    if not args.skip_training:
        print_section_header("Step 2: Model Training")
        
        print(f"{Colors.CYAN}Training will:{Colors.NC}")
        print("  • Use 2 GPUs by default")
        print("  • Train for 100,000 steps (~2-3 days)")
        print("  • Save checkpoints every 5,000 steps")
        print("  • Log to WandB (https://wandb.ai)")
        print(f"  • Save outputs to {output_dir}/\n")
        
        if confirm("Start training now?", auto_yes=args.auto_yes):
            print(f"{Colors.GREEN}Starting training...{Colors.NC}\n")
            training_done = start_training(args.config_name, args.auto_yes)
        else:
            print(f"{Colors.CYAN}To start training later, run:{Colors.NC}")
            print(f"  {Colors.YELLOW}python scripts/train_atat.py{Colors.NC}")
    
    # Step 3: Evaluation
    if training_done:
        print_section_header("Step 3: Model Evaluation")
        
        print(f"{Colors.CYAN}Finding latest checkpoint...{Colors.NC}")
        latest_ckpt = find_latest_checkpoint(output_dir)
        
        if latest_ckpt:
            print(f"{Colors.GREEN}✓{Colors.NC} Found: {latest_ckpt}\n")
            
            if confirm("Evaluate this checkpoint?", auto_yes=args.auto_yes):
                evaluate_checkpoint(latest_ckpt)
        else:
            print(f"{Colors.YELLOW}No checkpoints found yet.{Colors.NC}")
            print(f"{Colors.CYAN}To evaluate later, run:{Colors.NC}")
            print(f"  {Colors.YELLOW}python scripts/eval_atat.py --checkpoint /path/to/checkpoint.ckpt{Colors.NC}")
    
    # Print next steps
    print_next_steps(args.scratch_dir)


if __name__ == "__main__":
    main()
