#!/usr/bin/env python3
"""
MDLM ATAT Ablation Studies

Runs ablation studies to analyze individual component contributions.
Supports multiple ablation configurations.

Usage:
    python scripts/run_ablation.py --study STUDY_NAME [OPTIONS]

Ablation Studies:
    importance_only         Test importance estimation alone (no adaptive masking, no curriculum)
    no_curriculum          Test ATAT without curriculum learning
    no_adaptive_masking    Test ATAT without adaptive masking
    baseline               Standard MDLM without any ATAT enhancements

Training Options:
    --max-steps N          Maximum training steps [default: 100000]
    --val-interval N       Validation interval [default: 5000]
    --num-gpus N           Number of GPUs [default: 2]
    --batch-size N         Batch size [default: 85]

Data Options:
    --cache-dir PATH       Dataset cache directory [default: /media/scratch/adele/mdlm_fresh/data_cache]
    --output-dir PATH      Output directory [default: /media/scratch/adele/mdlm_fresh/outputs]
    --log-dir PATH         Log directory [default: /media/scratch/adele/mdlm_fresh/logs]

WandB Options:
    --wandb-project NAME   WandB project [default: mdlm-atat-ablation]
    --wandb-run-name NAME  WandB run name [default: auto-generated]

Examples:
    # Run importance-only ablation
    python scripts/run_ablation.py --study importance_only

    # Run no-curriculum ablation with custom steps
    python scripts/run_ablation.py --study no_curriculum --max-steps 50000

    # Run baseline comparison
    python scripts/run_ablation.py --study baseline --num-gpus 4

    # Run all ablations sequentially
    python scripts/run_ablation.py --study all
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# Color codes
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    CYAN = '\033[0;36m'
    NC = '\033[0m'


# Ablation study configurations
ABLATION_CONFIGS = {
    "importance_only": {
        "description": "Importance estimation only (no adaptive masking, no curriculum)",
        "config_overrides": {
            "use_importance": "true",
            "use_adaptive_masking": "false",
            "use_curriculum": "false",
        }
    },
    "no_curriculum": {
        "description": "ATAT without curriculum learning",
        "config_overrides": {
            "use_importance": "true",
            "use_adaptive_masking": "true",
            "use_curriculum": "false",
        }
    },
    "no_adaptive_masking": {
        "description": "ATAT without adaptive masking",
        "config_overrides": {
            "use_importance": "true",
            "use_adaptive_masking": "false",
            "use_curriculum": "true",
        }
    },
    "baseline": {
        "description": "Standard MDLM without ATAT enhancements",
        "config_name": "atat/tiny",  # Use atat config but disable features
        "config_overrides": {
            "use_importance": "false",
            "use_adaptive_masking": "false",
            "use_curriculum": "false",
        }
    },
}


def print_header(message: str) -> None:
    """Print formatted header."""
    print(f"\n{Colors.BLUE}{'='*70}{Colors.NC}")
    print(f"{Colors.BLUE}{message.center(70)}{Colors.NC}")
    print(f"{Colors.BLUE}{'='*70}{Colors.NC}\n")


def run_ablation_study(
    study_name: str,
    config: Dict,
    max_steps: int,
    val_interval: int,
    num_gpus: int,
    batch_size: int,
    cache_dir: str,
    output_dir: str,
    log_dir: str,
    wandb_project: str,
    wandb_run_name: str = None
) -> int:
    """
    Run a single ablation study.
    
    Returns:
        Exit code from training process
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = wandb_run_name or f"ablation_{study_name}_{timestamp}"
    log_file = os.path.join(log_dir, f"ablation_{study_name}_{timestamp}.log")
    
    print_header(f"Ablation Study: {study_name}")
    print(f"{Colors.CYAN}Description:{Colors.NC} {config['description']}")
    print(f"{Colors.CYAN}Run name:{Colors.NC} {run_name}")
    print(f"{Colors.CYAN}Log file:{Colors.NC} {log_file}\n")
    
    # Change to project directory
    project_dir = Path(__file__).parent.parent
    os.chdir(project_dir)
    
    # Build command
    config_name = config.get("config_name", "atat/tiny")
    
    cmd = [
        "python", "../mdlm/main.py",
        "--config-path", "../mdlm_atat/configs",
        "--config-name", config_name,
        f"trainer.max_steps={max_steps}",
        f"trainer.val_check_interval={val_interval}",
        "trainer.log_every_n_steps=100",
        f"trainer.devices={num_gpus}",
        f"wandb.project={wandb_project}",
        f"wandb.name={run_name}",
        "wandb.offline=true",
        f"hydra.run.dir={output_dir}/ablation_{study_name}/${{now:%Y.%m.%d}}/${{now:%H%M%S}}",
    ]
    
    # Add config overrides
    for key, value in config.get("config_overrides", {}).items():
        cmd.append(f"{key}={value}")
    
    # Run training
    print(f"{Colors.GREEN}Starting training...{Colors.NC}\n")
    
    with open(log_file, 'w') as f:
        f.write(f"Ablation Study: {study_name}\n")
        f.write(f"Description: {config['description']}\n")
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
        description="Run MDLM ATAT ablation studies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Required arguments
    parser.add_argument(
        "--study",
        type=str,
        required=True,
        choices=list(ABLATION_CONFIGS.keys()) + ["all"],
        help="Ablation study to run"
    )
    
    # Training options
    parser.add_argument("--max-steps", type=int, default=100000, help="Max training steps")
    parser.add_argument("--val-interval", type=int, default=5000, help="Validation interval")
    parser.add_argument("--num-gpus", type=int, default=2, help="Number of GPUs")
    parser.add_argument("--batch-size", type=int, default=85, help="Batch size")
    
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
    parser.add_argument("--wandb-project", type=str, default="mdlm-atat-ablation", help="WandB project")
    parser.add_argument("--wandb-run-name", type=str, help="WandB run name")
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    print_header("MDLM ATAT Ablation Studies")
    
    # Determine which studies to run
    if args.study == "all":
        studies_to_run = list(ABLATION_CONFIGS.keys())
        print(f"{Colors.CYAN}Running all {len(studies_to_run)} ablation studies{Colors.NC}\n")
    else:
        studies_to_run = [args.study]
    
    # Run studies
    results = {}
    for study_name in studies_to_run:
        config = ABLATION_CONFIGS[study_name]
        
        exit_code = run_ablation_study(
            study_name=study_name,
            config=config,
            max_steps=args.max_steps,
            val_interval=args.val_interval,
            num_gpus=args.num_gpus,
            batch_size=args.batch_size,
            cache_dir=args.cache_dir,
            output_dir=args.output_dir,
            log_dir=args.log_dir,
            wandb_project=args.wandb_project,
            wandb_run_name=args.wandb_run_name
        )
        
        results[study_name] = exit_code
        
        if exit_code == 0:
            print(f"\n{Colors.GREEN}✓ {study_name} completed successfully{Colors.NC}\n")
        else:
            print(f"\n{Colors.RED}✗ {study_name} failed with exit code {exit_code}{Colors.NC}\n")
    
    # Print summary
    print_header("Ablation Studies Summary")
    
    success_count = sum(1 for code in results.values() if code == 0)
    total_count = len(results)
    
    print(f"Completed: {success_count}/{total_count} studies\n")
    
    for study_name, exit_code in results.items():
        status = f"{Colors.GREEN}✓ Success{Colors.NC}" if exit_code == 0 else f"{Colors.RED}✗ Failed{Colors.NC}"
        print(f"  {study_name}: {status}")
    
    print(f"\n{Colors.CYAN}View results at: https://wandb.ai{Colors.NC}\n")
    
    sys.exit(0 if success_count == total_count else 1)


if __name__ == "__main__":
    main()
