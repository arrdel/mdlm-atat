#!/usr/bin/env python3
"""
Phase 1B: Masking Strategy Ablation Training

Orchestrates training of all 4 masking strategy variants:
1. Balanced (ours): Early phase preserves, late phase focuses on important tokens
2. Importance-Proportional: Always mask important tokens more
3. Importance-Inverse: Always preserve important tokens
4. Time-Only (control): Uniform masking (no importance signal)

Each variant uses full ATAT importance estimator (0.7 learned + 0.3 frequency).

The key difference is the masking function:
- Balanced:              g_bal(i,t) = (1-t) * g_inv + t * g_prop
- Proportional:          g_prop(i,t) = 0.7*i + 0.3*(1-t)
- Inverse:              g_inv(i,t) = 0.7*(1-i) + 0.3*t
- Time-Only:            g_time(i,t) = f(t)  (importance ignored)

where:
  i = importance score [0,1]
  t = curriculum time [0,1]
  g(*) = masking probability [0,1]

Usage:
    # Train single strategy on debug dataset
    python train_masking_strategies.py \
      --strategy balanced \
      --dataset-preset debug \
      --max-steps 1000

    # Train all 4 strategies on validation dataset
    python train_masking_strategies.py \
      --all-strategies \
      --dataset-preset validation \
      --max-steps 100000

    # Production run with detailed configuration
    python train_masking_strategies.py \
      --all-strategies \
      --dataset-preset production \
      --max-steps 500000 \
      --num-gpus 8 \
      --batch-size 256 \
      --wandb-project atat-phase1b
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import datetime
import os

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

# Import dataset manager
try:
    from mdlm_atat.utils.dataset_config import get_dataset_manager
except ImportError:
    sys.path.insert(0, str(PROJECT_ROOT))
    from mdlm_atat.utils.dataset_config import get_dataset_manager


# Define all 4 masking strategy variants
STRATEGIES = {
    "balanced": {
        "config_file": "masking_balanced.yaml",
        "description": "Balanced strategy: (1-t)*g_inv + t*g_prop",
        "formula": "Early: preserve important, Late: focus on important",
        "expected_ppl": 39.03,
        "notes": "Our proposed strategy - balances curriculum with importance",
    },
    "proportional": {
        "config_file": "masking_proportional.yaml",
        "description": "Importance-Proportional: Always mask important tokens more",
        "formula": "g_prop(i,t) = 0.7*i + 0.3*(1-t)",
        "expected_ppl": 39.87,
        "notes": "May lead to overtrain on easy tokens, undertrain on hard",
    },
    "inverse": {
        "config_file": "masking_inverse.yaml",
        "description": "Importance-Inverse: Always preserve important tokens",
        "formula": "g_inv(i,t) = 0.7*(1-i) + 0.3*t",
        "expected_ppl": 40.21,
        "notes": "May lead to undertrain on important tokens early",
    },
    "time_only": {
        "config_file": "masking_time_only.yaml",
        "description": "Time-Only (control): Uniform masking, no importance signal",
        "formula": "g_time(i,t) = f(t)  (importance ignored)",
        "expected_ppl": 42.31,
        "notes": "Baseline without importance - validates curriculum matters",
    },
}


class MaskingStrategyTrainer:
    """Orchestrate masking strategy ablation experiments (Phase 1B)"""

    def __init__(
        self,
        dataset_preset: str = "debug",
        output_base: Optional[Path] = None,
        num_gpus: int = 2,
        batch_size: int = 64,
        max_steps: int = 500000,
        checkpoint_resume: Optional[Path] = None,
        wandb_project: Optional[str] = None,
        wandb_offline: bool = False,
    ):
        """
        Initialize masking strategy trainer

        Args:
            dataset_preset: debug, validation, or production
            output_base: Base output directory for checkpoints
            num_gpus: Number of GPUs per variant
            batch_size: Training batch size
            max_steps: Maximum training steps per variant
            checkpoint_resume: Path to checkpoint to resume from
            wandb_project: WandB project name
            wandb_offline: Run WandB offline
        """
        self.dataset_preset = dataset_preset
        self.num_gpus = num_gpus
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.checkpoint_resume = checkpoint_resume
        self.wandb_project = wandb_project
        self.wandb_offline = wandb_offline

        # Set output directory
        if output_base is None:
            output_base = Path("/media/scratch/adele/mdlm_fresh/outputs/phase1b_masking")
        self.output_base = Path(output_base)
        self.variants_dir = self.output_base / "variants"
        self.variants_dir.mkdir(parents=True, exist_ok=True)

        # Get dataset configuration
        self.dataset_manager = get_dataset_manager(preset=dataset_preset)

        # Phase 1B output directory
        self.phase_output_dir = self.output_base / "variants"
        self.phase_output_dir.mkdir(parents=True, exist_ok=True)

    def train_strategy(
        self,
        strategy: str,
        dry_run: bool = False,
    ) -> Dict:
        """
        Train a single masking strategy variant

        Args:
            strategy: Strategy name (balanced, proportional, inverse, time_only)
            dry_run: If True, print command without executing

        Returns:
            Dictionary with training results
        """
        if strategy not in STRATEGIES:
            print(f"❌ Unknown strategy: {strategy}")
            return {"strategy": strategy, "status": "error", "error": "Unknown strategy"}

        strategy_config = STRATEGIES[strategy]

        # Create variant output directory
        variant_output_dir = self.variants_dir / strategy
        variant_output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*80}")
        print(f"TRAINING STRATEGY: {strategy.upper()}")
        print(f"{'='*80}")
        print(f"Description:        {strategy_config['description']}")
        print(f"Formula:            {strategy_config['formula']}")
        print(f"Expected PPL:       {strategy_config['expected_ppl']}")
        print(f"Notes:              {strategy_config['notes']}")
        print(f"Output dir:         {variant_output_dir}")
        print(f"Dataset preset:     {self.dataset_preset}")
        print(f"Max steps:          {self.max_steps}")
        print(f"Num GPUs:           {self.num_gpus}")
        print(f"Batch size:         {self.batch_size}")

        # Build training command
        cmd = self._build_training_command(
            strategy=strategy,
            config_file=strategy_config["config_file"],
            output_dir=variant_output_dir,
        )

        if dry_run:
            print(f"\n[DRY RUN] Command:")
            print(" ".join(cmd))
            return {
                "strategy": strategy,
                "status": "dry_run",
                "command": " ".join(cmd),
            }

        # Execute training
        print(f"\nStarting training...")
        print(f"Command: {' '.join(cmd)}\n")

        try:
            # Run in subprocess - conda environment already active
            result = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                capture_output=False,
                text=True,
            )

            if result.returncode == 0:
                print(f"\n✓ Training completed successfully for {strategy}")
                return {
                    "strategy": strategy,
                    "status": "completed",
                    "output_dir": str(variant_output_dir),
                    "timestamp": datetime.datetime.now().isoformat(),
                }
            else:
                print(f"\n❌ Training failed for {strategy} (return code: {result.returncode})")
                return {
                    "strategy": strategy,
                    "status": "error",
                    "error": f"Training failed with return code {result.returncode}",
                }

        except Exception as e:
            print(f"\n❌ Error training {strategy}: {e}")
            return {
                "strategy": strategy,
                "status": "error",
                "error": str(e),
            }

    def _build_training_command(
        self,
        strategy: str,
        config_file: str,
        output_dir: Path,
    ) -> List[str]:
        """Build training command for a strategy variant"""
        # Extract config name without .yaml extension
        config_name = f"atat/{config_file.replace('.yaml', '')}"

        cmd = [
            "python",
            str(PROJECT_ROOT / "mdlm_atat" / "scripts" / "training" / "train_atat.py"),
            f"--config-name={config_name}",
            f"--output-dir={output_dir}",
            f"--max-steps={self.max_steps}",
            f"--batch-size={self.batch_size}",
            f"--num-gpus={self.num_gpus}",
            "--no-confirm",  # Skip confirmation prompt for automated runs
        ]

        # Add optional arguments
        if self.checkpoint_resume:
            cmd.append(f"--resume={self.checkpoint_resume}")

        if self.wandb_project:
            cmd.append(f"--wandb-project={self.wandb_project}")

        if self.wandb_offline:
            cmd.append("--wandb-offline")

        return cmd

    def train_all_strategies(
        self,
        dry_run: bool = False,
        sequential: bool = True,
    ) -> Dict[str, Dict]:
        """
        Train all 4 masking strategy variants

        Args:
            dry_run: If True, print commands without executing
            sequential: If True, train sequentially; if False, try parallel (not recommended)

        Returns:
            Dictionary with results for all strategies
        """
        print(f"\n{'#'*80}")
        print(f"# PHASE 1B: MASKING STRATEGY ABLATION")
        print(f"# Training {len(STRATEGIES)} strategies")
        print(f"# Preset: {self.dataset_preset}")
        print(f"# Max steps per variant: {self.max_steps}")
        print(f"{'#'*80}\n")

        all_results = {}

        if sequential:
            # Train strategies sequentially
            for strategy in STRATEGIES.keys():
                results = self.train_strategy(strategy, dry_run=dry_run)
                all_results[strategy] = results

                if results["status"] == "error":
                    print(f"⚠️  Strategy {strategy} failed, continuing with next...")

        else:
            print("⚠️  Parallel training not recommended (would exceed GPU allocation)")
            print("    Training sequentially instead...")
            for strategy in STRATEGIES.keys():
                results = self.train_strategy(strategy, dry_run=dry_run)
                all_results[strategy] = results

        return all_results

    def save_results(self, all_results: Dict[str, Dict], filename: str = "strategy_results.json"):
        """Save training results to JSON"""
        results_file = self.output_base / filename
        with open(results_file, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n✓ Results saved to: {results_file}")
        return results_file

    def generate_summary(self, all_results: Dict[str, Dict]) -> str:
        """Generate human-readable summary"""
        summary = "\n" + "=" * 100 + "\n"
        summary += "PHASE 1B: MASKING STRATEGY ABLATION - SUMMARY\n"
        summary += "=" * 100 + "\n"
        summary += f"Dataset Preset:     {self.dataset_preset}\n"
        summary += f"Max Steps:          {self.max_steps}\n"
        summary += f"Timestamp:          {datetime.datetime.now().isoformat()}\n\n"

        summary += f"{'Strategy':<20} | {'Expected PPL':<12} | {'Status':<15} | {'Notes':<40}\n"
        summary += "-" * 100 + "\n"

        for strategy in STRATEGIES.keys():
            result = all_results.get(strategy, {})
            expected_ppl = STRATEGIES[strategy]["expected_ppl"]
            status = result.get("status", "unknown")
            notes = STRATEGIES[strategy]["notes"][:37] + "..." if len(STRATEGIES[strategy]["notes"]) > 40 else STRATEGIES[strategy]["notes"]

            summary += f"{strategy:<20} | {expected_ppl:<12} | {status:<15} | {notes:<40}\n"

        summary += "=" * 100 + "\n"
        return summary

    def print_strategy_details(self):
        """Print detailed information about each strategy"""
        print(f"\n{'#'*80}")
        print(f"# MASKING STRATEGY DETAILS")
        print(f"{'#'*80}\n")

        for strategy, config in STRATEGIES.items():
            print(f"Strategy: {strategy.upper()}")
            print(f"  Description:   {config['description']}")
            print(f"  Formula:       {config['formula']}")
            print(f"  Expected PPL:  {config['expected_ppl']}")
            print(f"  Config file:   {config['config_file']}")
            print(f"  Notes:         {config['notes']}")
            print()


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1B: Train masking strategy ablation variants",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train single strategy on debug dataset
  python train_masking_strategies.py --strategy balanced --dataset-preset debug --max-steps 1000

  # Train all 4 strategies on validation dataset
  python train_masking_strategies.py --all-strategies --dataset-preset validation --max-steps 100000

  # Production run with WandB logging
  python train_masking_strategies.py --all-strategies --dataset-preset production \
    --max-steps 500000 --num-gpus 8 --wandb-project atat-phase1b

  # Dry run to see commands without executing
  python train_masking_strategies.py --all-strategies --dry-run
        """,
    )

    # Strategy selection
    strategy_group = parser.add_mutually_exclusive_group(required=True)
    strategy_group.add_argument(
        "--strategy",
        choices=list(STRATEGIES.keys()),
        help="Train single masking strategy variant",
    )
    strategy_group.add_argument(
        "--all-strategies",
        action="store_true",
        help="Train all 4 masking strategy variants sequentially",
    )
    strategy_group.add_argument(
        "--info",
        action="store_true",
        help="Print strategy details and exit",
    )

    # Dataset and training options
    parser.add_argument(
        "--dataset-preset",
        choices=["debug", "validation", "production"],
        default="debug",
        help="Dataset preset: debug (100K), validation (10M), production (262B) [default: debug]",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Maximum training steps [default: 1000]",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=2,
        help="Number of GPUs to use [default: 2]",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Training batch size [default: 64]",
    )

    # Output and checkpointing
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for checkpoints [default: /media/scratch/adele/mdlm_fresh/outputs/phase1b_masking]",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume from checkpoint",
    )

    # WandB logging
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="WandB project name for logging",
    )
    parser.add_argument(
        "--wandb-offline",
        action="store_true",
        help="Run WandB in offline mode",
    )

    # Execution options
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )

    args = parser.parse_args()

    # Print strategy info if requested
    if args.info:
        trainer = MaskingStrategyTrainer(dataset_preset=args.dataset_preset)
        trainer.print_strategy_details()
        return

    # Create trainer
    trainer = MaskingStrategyTrainer(
        dataset_preset=args.dataset_preset,
        output_base=args.output_dir,
        num_gpus=args.num_gpus,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        checkpoint_resume=args.resume,
        wandb_project=args.wandb_project,
        wandb_offline=args.wandb_offline,
    )

    # Train strategy(ies)
    if args.strategy:
        result = trainer.train_strategy(args.strategy, dry_run=args.dry_run)
        print(f"\nResult: {json.dumps(result, indent=2)}")

    elif args.all_strategies:
        trainer.print_strategy_details()
        all_results = trainer.train_all_strategies(dry_run=args.dry_run)
        summary = trainer.generate_summary(all_results)
        print(summary)
        trainer.save_results(all_results)


if __name__ == "__main__":
    main()
