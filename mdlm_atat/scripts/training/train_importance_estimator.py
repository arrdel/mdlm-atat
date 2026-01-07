#!/usr/bin/env python3
"""
Importance Estimator Ablation Study Training Script

Trains all 4 importance variants for Phase 1A ablation:
1. Full ATAT (0.7 learned + 0.3 frequency)
2. Frequency-only (no learned component)
3. Learned-only (no frequency prior)
4. Uniform (no importance weighting - baseline)

Usage:
    # Train a single variant on small dataset (debug mode)
    python train_importance_variants.py \
      --variant full \
      --dataset-preset debug \
      --max-steps 1000

    # Train all variants sequentially with small dataset
    python train_importance_variants.py \
      --all \
      --dataset-preset debug \
      --max-steps 1000

    # Train single variant on medium dataset (validation)
    python train_importance_variants.py \
      --variant full \
      --dataset-preset validation \
      --max-steps 100000

    # Train variant on full dataset (production)
    python train_importance_variants.py \
      --variant full \
      --dataset-preset production \
      --max-steps 500000
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional
import json
import datetime

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "mdlm_atat" / "scripts" / "training"
CONFIGS_DIR = PROJECT_ROOT / "mdlm_atat" / "configs" / "atat"
OUTPUTS_DIR = Path("/media/scratch/adele/mdlm_fresh/outputs/phase1a_ablation")

# Define the 4 variants for importance estimator ablation
VARIANTS = {
    "full": {
        "config": "importance_estimator_full.yaml",
        "description": "Full ATAT (0.7 learned + 0.3 frequency)",
        "expected_ppl": 39.03,
    },
    "frequency_only": {
        "config": "importance_estimator_frequency_only.yaml",
        "description": "No learned component, frequency-only importance",
        "expected_ppl": 41.87,
    },
    "learned_only": {
        "config": "importance_estimator_learned_only.yaml",
        "description": "No frequency prior, learned-only importance",
        "expected_ppl": 40.12,
    },
    "uniform": {
        "config": "importance_estimator_uniform.yaml",
        "description": "No importance weighting (baseline)",
        "expected_ppl": 42.31,
    },
}


class ImportanceAblationTrainer:
    """Manages training of importance estimator ablation variants"""

    def __init__(
        self,
        dataset_preset: str = "debug",
        max_steps: int = 1000,
        num_gpus: int = 2,
        batch_size: int = 32,
        wandb_project: str = "mdlm-atat-phase1a",
        offline_mode: bool = False,
        resume_from: Optional[str] = None,
    ):
        """
        Initialize trainer

        Args:
            dataset_preset: debug, validation, or production
            max_steps: Maximum training steps
            num_gpus: Number of GPUs to use
            batch_size: Per-GPU batch size
            wandb_project: WandB project name
            offline_mode: Run WandB offline
            resume_from: Resume from checkpoint
        """
        self.dataset_preset = dataset_preset
        self.max_steps = max_steps
        self.num_gpus = num_gpus
        self.batch_size = batch_size
        self.wandb_project = wandb_project
        self.offline_mode = offline_mode
        self.resume_from = resume_from

        # Create output directory
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

        # Log configuration
        self.log_dir = OUTPUTS_DIR / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_dir = OUTPUTS_DIR / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train_variant(self, variant: str, verbose: bool = True) -> bool:
        """
        Train a single importance variant

        Args:
            variant: One of: full, frequency_only, learned_only, uniform
            verbose: Print output

        Returns:
            True if successful, False otherwise
        """
        if variant not in VARIANTS:
            print(f"❌ Unknown variant: {variant}")
            print(f"   Available: {', '.join(VARIANTS.keys())}")
            return False

        config = VARIANTS[variant]
        config_file = CONFIGS_DIR / config["config"]

        if not config_file.exists():
            print(f"❌ Config file not found: {config_file}")
            return False

        print(f"\n{'='*70}")
        print(f"TRAINING IMPORTANCE VARIANT: {variant.upper()}")
        print(f"{'='*70}")
        print(f"Description:     {config['description']}")
        print(f"Config:          {config_file.name}")
        print(f"Dataset preset:  {self.dataset_preset}")
        print(f"Max steps:       {self.max_steps}")
        print(f"GPUs:            {self.num_gpus}")
        print(f"Batch size:      {self.batch_size}")
        print(f"Expected PPL:    {config['expected_ppl']}")
        print(f"Timestamp:       {datetime.datetime.now().isoformat()}")
        print(f"{'='*70}\n")

        # Build command
        cmd = [
            "conda",
            "run",
            "-n",
            "mdlm-atat",
            "python",
            str(PROJECT_ROOT / "mdlm_atat" / "scripts" / "training" / "train_atat.py"),
            "--config-name",
            f"atat/{config['config'].replace('.yaml', '')}",
            "--max-steps",
            str(self.max_steps),
            "--num-gpus",
            str(self.num_gpus),
            "--batch-size",
            str(self.batch_size),
            "--output-dir",
            str(self.checkpoint_dir / variant),
            "--log-dir",
            str(self.log_dir / variant),
            "--wandb-project",
            self.wandb_project,
            "--wandb-run-name",
            f"phase1a-{variant}-{self.dataset_preset}",
            "--no-confirm",
        ]

        if self.offline_mode:
            cmd.append("--wandb-offline")

        if self.resume_from:
            cmd.extend(["--resume", self.resume_from])

        # Run training
        try:
            result = subprocess.run(
                cmd,
                cwd=PROJECT_ROOT,
                check=False,
            )
            return result.returncode == 0
        except Exception as e:
            print(f"❌ Error training variant {variant}: {e}")
            return False

    def train_all_variants(
        self, sequential: bool = True, skip_failed: bool = False
    ) -> dict:
        """
        Train all importance variants

        Args:
            sequential: If True, train one after another; if False, attempt parallel
            skip_failed: If True, continue even if a variant fails

        Returns:
            Dictionary with results for each variant
        """
        results = {}
        variant_list = list(VARIANTS.keys())

        print(f"\n{'#'*70}")
        print(f"# IMPORTANCE ESTIMATOR ABLATION STUDY")
        print(f"# Phase 1A: Train All 4 Variants")
        print(f"# Training {len(variant_list)} variants sequentially")
        print(f"# Preset: {self.dataset_preset} | Max steps: {self.max_steps}")
        print(f"{'#'*70}\n")

        for variant in variant_list:
            print(f"\n[{variant_list.index(variant) + 1}/{len(variant_list)}]")
            success = self.train_variant(variant)
            results[variant] = {
                "success": success,
                "config": VARIANTS[variant]["config"],
                "expected_ppl": VARIANTS[variant]["expected_ppl"],
                "timestamp": datetime.datetime.now().isoformat(),
            }

            if not success and not skip_failed:
                print(f"❌ Training failed for {variant}, stopping")
                break

        return results

    def save_results(self, results: dict, filename: str = "ablation_results.json"):
        """Save training results to JSON"""
        output_file = OUTPUTS_DIR / filename
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Train importance estimator ablation variants for Phase 1A",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train single variant on small dataset (quick test)
  python train_importance_variants.py --variant full --dataset-preset debug --max-steps 1000

  # Train all variants on small dataset (full ablation test)
  python train_importance_variants.py --all --dataset-preset debug --max-steps 1000

  # Train single variant on medium dataset (validation)
  python train_importance_variants.py --variant full --dataset-preset validation

  # Train variant on full dataset (production)
  python train_importance_variants.py --variant full --dataset-preset production --max-steps 500000
        """,
    )

    # Variant selection
    variant_group = parser.add_mutually_exclusive_group(required=True)
    variant_group.add_argument(
        "--variant",
        choices=list(VARIANTS.keys()),
        help="Train single variant",
    )
    variant_group.add_argument(
        "--all",
        action="store_true",
        help="Train all variants sequentially",
    )

    # Dataset and training options
    parser.add_argument(
        "--dataset-preset",
        choices=["debug", "validation", "production"],
        default="debug",
        help="Dataset size preset [default: debug - small for quick testing]",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Maximum training steps [default: 1000 for debug, 100000 for validation, 500000 for production]",
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
        default=32,
        help="Per-GPU batch size [default: 32]",
    )

    # WandB options
    parser.add_argument(
        "--wandb-project",
        default="mdlm-atat-phase1a",
        help="WandB project name",
    )
    parser.add_argument(
        "--wandb-offline",
        action="store_true",
        help="Run WandB in offline mode",
    )

    # Training options
    parser.add_argument(
        "--resume",
        metavar="PATH",
        help="Resume from checkpoint",
    )
    parser.add_argument(
        "--skip-failed",
        action="store_true",
        help="Continue even if a variant fails",
    )

    args = parser.parse_args()

    # Create trainer
    trainer = ImportanceAblationTrainer(
        dataset_preset=args.dataset_preset,
        max_steps=args.max_steps,
        num_gpus=args.num_gpus,
        batch_size=args.batch_size,
        wandb_project=args.wandb_project,
        offline_mode=args.wandb_offline,
        resume_from=args.resume,
    )

    # Train variant(s)
    if args.variant:
        success = trainer.train_variant(args.variant)
        sys.exit(0 if success else 1)
    elif args.all:
        results = trainer.train_all_variants(skip_failed=args.skip_failed)
        trainer.save_results(results)
        
        # Summary
        print(f"\n{'='*70}")
        print("ABLATION STUDY SUMMARY")
        print(f"{'='*70}")
        for variant, result in results.items():
            status = "✓" if result["success"] else "✗"
            print(f"{status} {variant:20s} | Expected PPL: {result['expected_ppl']}")
        print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
