#!/usr/bin/env python3
"""
Phase 1C: Curriculum Schedule Ablation Training

Orchestrates training of 4 curriculum schedule variants to find optimal 
easy/medium/hard stage transition boundaries:

1. Default (0.3, 0.7) - Our baseline from Phase 1B
   - Easy: 0-150K (30%), Medium: 150K-350K (40%), Hard: 350K-500K (30%)
   
2. Early Transition (0.2, 0.6) - Earlier hard stage start
   - Easy: 0-100K (20%), Medium: 100K-300K (40%), Hard: 300K-500K (40%)
   - Hypothesis: More time in hard stage = better hard token training
   
3. Late Transition (0.35, 0.8) - Later hard stage start
   - Easy: 0-175K (35%), Medium: 175K-400K (45%), Hard: 400K-500K (20%)
   - Hypothesis: Longer easy stage = more stable training
   
4. No Curriculum (control) - Uniform masking throughout
   - All: 0-500K with constant masking probability
   - Hypothesis: Validates curriculum importance

All variants use balanced masking strategy from Phase 1B and full ATAT 
importance estimator from Phase 1A.

Usage:
    # Train single schedule on debug dataset
    python train_curriculum_schedules.py \
      --schedule default \
      --dataset-preset debug \
      --max-steps 1000

    # Train all 4 schedules on validation dataset
    python train_curriculum_schedules.py \
      --all-schedules \
      --dataset-preset validation \
      --max-steps 100000

    # Production run with detailed configuration
    python train_curriculum_schedules.py \
      --all-schedules \
      --dataset-preset production \
      --max-steps 500000 \
      --num-gpus 8 \
      --batch-size 256 \
      --wandb-project atat-phase1c
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional
import datetime
import os

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

# Import dataset manager
try:
    from mdlm_atat.utils.dataset_config import get_dataset_manager
except ImportError:
    sys.path.insert(0, str(PROJECT_ROOT))
    from mdlm_atat.utils.dataset_config import get_dataset_manager


# Define all 4 curriculum schedule variants
SCHEDULES = {
    "default": {
        "config_file": "curriculum_default.yaml",
        "description": "Default curriculum: Easy (0-150K), Medium (150K-350K), Hard (350K-500K)",
        "boundaries": (0.3, 0.7),
        "expected_ppl": 39.03,
        "easy_steps": 150000,
        "medium_steps": 200000,
        "hard_steps": 150000,
        "notes": "Our baseline from Phase 1B - balanced curriculum",
    },
    "early": {
        "config_file": "curriculum_early.yaml",
        "description": "Early transition: Easy (0-100K), Medium (100K-300K), Hard (300K-500K)",
        "boundaries": (0.2, 0.6),
        "expected_ppl": 39.54,
        "easy_steps": 100000,
        "medium_steps": 200000,
        "hard_steps": 200000,
        "notes": "40% of training in hard stage - validates hard token training",
    },
    "late": {
        "config_file": "curriculum_late.yaml",
        "description": "Late transition: Easy (0-175K), Medium (175K-400K), Hard (400K-500K)",
        "boundaries": (0.35, 0.8),
        "expected_ppl": 39.71,
        "easy_steps": 175000,
        "medium_steps": 225000,
        "hard_steps": 100000,
        "notes": "35% easy, longer medium - validates stable training",
    },
    "no_curriculum": {
        "config_file": "curriculum_no_curriculum.yaml",
        "description": "No curriculum: All stages uniform masking throughout",
        "boundaries": (None, None),
        "expected_ppl": 40.25,
        "easy_steps": 500000,
        "medium_steps": 0,
        "hard_steps": 0,
        "notes": "Control: all tokens masked equally - validates curriculum importance",
    },
}


class CurriculumScheduleTrainer:
    """Orchestrate curriculum schedule ablation experiments (Phase 1C)"""

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
        Initialize curriculum schedule trainer

        Args:
            dataset_preset: debug, validation, or production
            output_base: Base output directory for checkpoints
            num_gpus: Number of GPUs per schedule
            batch_size: Training batch size
            max_steps: Maximum training steps per schedule
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
            output_base = Path("/media/scratch/adele/mdlm_fresh/outputs/phase1c_curriculum")
        self.output_base = Path(output_base)
        self.schedules_dir = self.output_base / "schedules"
        self.schedules_dir.mkdir(parents=True, exist_ok=True)

        # Get dataset configuration
        self.dataset_manager = get_dataset_manager(preset=dataset_preset)

    def train_schedule(
        self,
        schedule: str,
        dry_run: bool = False,
    ) -> Dict:
        """
        Train a single curriculum schedule variant

        Args:
            schedule: Schedule name (default, early, late, no_curriculum)
            dry_run: If True, print command without executing

        Returns:
            Dictionary with training results
        """
        if schedule not in SCHEDULES:
            print(f"❌ Unknown schedule: {schedule}")
            return {"schedule": schedule, "status": "error", "error": "Unknown schedule"}

        schedule_config = SCHEDULES[schedule]

        # Create schedule output directory
        schedule_output_dir = self.schedules_dir / schedule
        schedule_output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*90}")
        print(f"TRAINING SCHEDULE: {schedule.upper()}")
        print(f"{'='*90}")
        print(f"Description:        {schedule_config['description']}")
        print(f"Boundaries:         {schedule_config['boundaries']}")
        print(f"Easy/Medium/Hard:   {schedule_config['easy_steps']} / {schedule_config['medium_steps']} / {schedule_config['hard_steps']}")
        print(f"Expected PPL:       {schedule_config['expected_ppl']}")
        print(f"Notes:              {schedule_config['notes']}")
        print(f"Output dir:         {schedule_output_dir}")
        print(f"Dataset preset:     {self.dataset_preset}")
        print(f"Max steps:          {self.max_steps}")
        print(f"Num GPUs:           {self.num_gpus}")
        print(f"Batch size:         {self.batch_size}")

        # Build training command
        cmd = self._build_training_command(
            schedule=schedule,
            config_file=schedule_config["config_file"],
            output_dir=schedule_output_dir,
        )

        if dry_run:
            print(f"\n[DRY RUN] Command:")
            print(" ".join(cmd))
            return {
                "schedule": schedule,
                "status": "dry_run",
                "command": " ".join(cmd),
            }

        # Execute training
        print(f"\nStarting training...")
        print(f"Command: {' '.join(cmd)}\n")

        try:
            # Run in subprocess with conda environment activation
            conda_cmd = f"conda run -n mdlm-atat {' '.join(cmd)}"
            result = subprocess.run(
                conda_cmd,
                shell=True,
                cwd=str(PROJECT_ROOT),
                capture_output=False,
                text=True,
            )

            if result.returncode == 0:
                print(f"\n✓ Training completed successfully for {schedule}")
                return {
                    "schedule": schedule,
                    "status": "completed",
                    "output_dir": str(schedule_output_dir),
                    "timestamp": datetime.datetime.now().isoformat(),
                }
            else:
                print(f"\n❌ Training failed for {schedule} (return code: {result.returncode})")
                return {
                    "schedule": schedule,
                    "status": "error",
                    "error": f"Training failed with return code {result.returncode}",
                }

        except Exception as e:
            print(f"\n❌ Error training {schedule}: {e}")
            return {
                "schedule": schedule,
                "status": "error",
                "error": str(e),
            }

    def _build_training_command(
        self,
        schedule: str,
        config_file: str,
        output_dir: Path,
    ) -> List[str]:
        """Build training command for a schedule variant"""
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
        ]

        # Add optional arguments
        if self.checkpoint_resume:
            cmd.append(f"--resume={self.checkpoint_resume}")

        if self.wandb_project:
            cmd.append(f"--wandb-project={self.wandb_project}")

        if self.wandb_offline:
            cmd.append("--wandb-offline")

        return cmd

    def train_all_schedules(
        self,
        dry_run: bool = False,
        sequential: bool = True,
    ) -> Dict[str, Dict]:
        """
        Train all 4 curriculum schedule variants

        Args:
            dry_run: If True, print commands without executing
            sequential: If True, train sequentially; if False, try parallel (not recommended)

        Returns:
            Dictionary with results for all schedules
        """
        print(f"\n{'#'*90}")
        print(f"# PHASE 1C: CURRICULUM SCHEDULE ABLATION")
        print(f"# Training {len(SCHEDULES)} curriculum schedules")
        print(f"# Preset: {self.dataset_preset}")
        print(f"# Max steps per variant: {self.max_steps}")
        print(f"{'#'*90}\n")

        all_results = {}

        if sequential:
            # Train schedules sequentially
            for schedule in SCHEDULES.keys():
                results = self.train_schedule(schedule, dry_run=dry_run)
                all_results[schedule] = results

                if results["status"] == "error":
                    print(f"⚠️  Schedule {schedule} failed, continuing with next...")

        else:
            print("⚠️  Parallel training not recommended (would exceed GPU allocation)")
            print("    Training sequentially instead...")
            for schedule in SCHEDULES.keys():
                results = self.train_schedule(schedule, dry_run=dry_run)
                all_results[schedule] = results

        return all_results

    def save_results(self, all_results: Dict[str, Dict], filename: str = "schedule_results.json"):
        """Save training results to JSON"""
        results_file = self.output_base / filename
        with open(results_file, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n✓ Results saved to: {results_file}")
        return results_file

    def generate_summary(self, all_results: Dict[str, Dict]) -> str:
        """Generate human-readable summary"""
        summary = "\n" + "=" * 110 + "\n"
        summary += "PHASE 1C: CURRICULUM SCHEDULE ABLATION - SUMMARY\n"
        summary += "=" * 110 + "\n"
        summary += f"Dataset Preset:     {self.dataset_preset}\n"
        summary += f"Max Steps:          {self.max_steps}\n"
        summary += f"Timestamp:          {datetime.datetime.now().isoformat()}\n\n"

        summary += f"{'Schedule':<20} | {'Boundaries':<15} | {'Expected PPL':<12} | {'Status':<15} | {'Notes':<35}\n"
        summary += "-" * 110 + "\n"

        for schedule in SCHEDULES.keys():
            result = all_results.get(schedule, {})
            boundaries = SCHEDULES[schedule]["boundaries"]
            boundary_str = f"({boundaries[0]}, {boundaries[1]})" if boundaries[0] else "None"
            expected_ppl = SCHEDULES[schedule]["expected_ppl"]
            status = result.get("status", "unknown")
            notes = SCHEDULES[schedule]["notes"][:32] + "..." if len(SCHEDULES[schedule]["notes"]) > 35 else SCHEDULES[schedule]["notes"]

            summary += f"{schedule:<20} | {boundary_str:<15} | {expected_ppl:<12} | {status:<15} | {notes:<35}\n"

        summary += "=" * 110 + "\n"
        return summary

    def print_schedule_details(self):
        """Print detailed information about each schedule"""
        print(f"\n{'#'*90}")
        print(f"# CURRICULUM SCHEDULE DETAILS")
        print(f"{'#'*90}\n")

        for schedule, config in SCHEDULES.items():
            boundaries = config["boundaries"]
            print(f"Schedule: {schedule.upper()}")
            print(f"  Description:   {config['description']}")
            print(f"  Boundaries:    {boundaries}")
            print(f"  Easy/Med/Hard: {config['easy_steps']} / {config['medium_steps']} / {config['hard_steps']} steps")
            print(f"  Expected PPL:  {config['expected_ppl']}")
            print(f"  Config file:   {config['config_file']}")
            print(f"  Notes:         {config['notes']}")
            print()


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1C: Train curriculum schedule ablation variants",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train single schedule on debug dataset
  python train_curriculum_schedules.py --schedule default --dataset-preset debug --max-steps 1000

  # Train all 4 schedules on validation dataset
  python train_curriculum_schedules.py --all-schedules --dataset-preset validation --max-steps 100000

  # Production run with WandB logging
  python train_curriculum_schedules.py --all-schedules --dataset-preset production \
    --max-steps 500000 --num-gpus 8 --wandb-project atat-phase1c

  # Dry run to see commands without executing
  python train_curriculum_schedules.py --all-schedules --dry-run
        """,
    )

    # Schedule selection
    schedule_group = parser.add_mutually_exclusive_group(required=True)
    schedule_group.add_argument(
        "--schedule",
        choices=list(SCHEDULES.keys()),
        help="Train single curriculum schedule variant",
    )
    schedule_group.add_argument(
        "--all-schedules",
        action="store_true",
        help="Train all 4 curriculum schedule variants sequentially",
    )
    schedule_group.add_argument(
        "--info",
        action="store_true",
        help="Print schedule details and exit",
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
        help="Output directory for checkpoints [default: /media/scratch/adele/mdlm_fresh/outputs/phase1c_curriculum]",
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

    # Print schedule info if requested
    if args.info:
        trainer = CurriculumScheduleTrainer(dataset_preset=args.dataset_preset)
        trainer.print_schedule_details()
        return

    # Create trainer
    trainer = CurriculumScheduleTrainer(
        dataset_preset=args.dataset_preset,
        output_base=args.output_dir,
        num_gpus=args.num_gpus,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        checkpoint_resume=args.resume,
        wandb_project=args.wandb_project,
        wandb_offline=args.wandb_offline,
    )

    # Train schedule(s)
    if args.schedule:
        result = trainer.train_schedule(args.schedule, dry_run=args.dry_run)
        print(f"\nResult: {json.dumps(result, indent=2)}")

    elif args.all_schedules:
        trainer.print_schedule_details()
        all_results = trainer.train_all_schedules(dry_run=args.dry_run)
        summary = trainer.generate_summary(all_results)
        print(summary)
        trainer.save_results(all_results)


if __name__ == "__main__":
    main()
