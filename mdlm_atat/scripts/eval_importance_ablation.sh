#!/bin/bash
# Quick Phase 1A Evaluation Starter
# Runs eval_phase1a.py on specified GPUs

set -e

# Configuration
PROJECT_ROOT="/home/adelechinda/home/projects/mdlm/mdlm_atat"
CHECKPOINT_BASE="/media/scratch/adele/mdlm_fresh/outputs/phase1a_ablations/checkpoints"
OUTPUT_DIR="/home/adelechinda/home/projects/mdlm/report/results/phase1a"
GPUS="0,1"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Phase 1A Evaluation"
echo "=========================================="
echo "GPUs: $GPUS"
echo "Checkpoint base: $CHECKPOINT_BASE"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Set GPU environment
export CUDA_VISIBLE_DEVICES=$GPUS

# Run evaluation
echo "[$(date +'%Y-%m-%d %H:%M:%S')] Starting Phase 1A evaluation..."
python "$PROJECT_ROOT/scripts/eval_phase1a.py" \
  --checkpoint-base "$CHECKPOINT_BASE" \
  --output-dir "$OUTPUT_DIR"

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Phase 1A evaluation complete"
echo "Results saved to: $OUTPUT_DIR"
