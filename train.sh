#!/bin/bash

# Production Training Script for MDLM+ATAT with WandB
# Usage: bash train.sh

set -e

# Activate conda environment
source /home/adelechinda/miniconda3/bin/activate mdlm-atat

# Navigate to project directory
cd /home/adelechinda/home/projects/mdlm

# Print header
echo "╔════════════════════════════════════════════════════════╗"
echo "║     MDLM+ATAT Production Training with WandB           ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""
echo "Configuration:"
echo "  • Model: BERT-scale DiT with ATAT"
echo "  • Steps: 500,000"
echo "  • Batch Size: 4 per GPU × 6 GPUs = 24 global"
echo "  • Dataset: OpenWebText (40GB)"
echo "  • Precision: bf16"
echo "  • Learning Rate: 1e-4 with cosine decay"
echo ""
echo "Logging:"
echo "  • WandB Project: mdlm-atat"
echo "  • Local Logs: /media/scratch/adele/mdlm_fresh/logs/"
echo "  • Checkpoints: /media/scratch/adele/mdlm_fresh/outputs/"
echo ""
echo "Starting training in 3 seconds..."
sleep 3

# Start training
python mdlm_atat/scripts/train_atat.py \
  --config-name atat/production_training \
  --max-steps 500000 \
  --wandb-project mdlm-atat \
  --wandb-run-name production_$(date +%Y%m%d_%H%M%S) \
  --no-confirm
