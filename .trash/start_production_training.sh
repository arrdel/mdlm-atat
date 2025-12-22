#!/bin/bash
# Production Training Script for MDLM-ATAT on OpenWebText
# Usage: bash start_production_training.sh

set -e

echo "======================================================"
echo "    MDLM-ATAT Production Training on OpenWebText"
echo "======================================================"
echo ""

# Configuration
CONFIG_NAME="atat/wikitext103_validation"  # Will use same config as validation
MAX_STEPS=500000  # 10x validation (50k -> 500k)
BATCH_SIZE=4
NUM_GPUS=6
LEARNING_RATE=1e-4
CACHE_DIR="/media/scratch/adele/mdlm_fresh/data_cache"
OUTPUT_DIR="/media/scratch/adele/mdlm_fresh/outputs"
LOG_DIR="/media/scratch/adele/mdlm_fresh/logs"

# Create directories
mkdir -p "$CACHE_DIR" "$OUTPUT_DIR" "$LOG_DIR"

echo "Configuration:"
echo "  Config: $CONFIG_NAME"
echo "  Dataset: OpenWebText (40GB, streaming)"
echo "  Max Steps: $MAX_STEPS"
echo "  Batch Size: $BATCH_SIZE per GPU × $NUM_GPUS GPUs = $((BATCH_SIZE * NUM_GPUS)) global"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Estimated Duration: 6-8 days on 6x RTX 4090"
echo ""

echo "Starting training..."
cd /home/adelechinda/home/projects/mdlm/mdlm_atat

# Start training in background
/home/adelechinda/miniconda3/envs/mdlm-atat/bin/python scripts/train_atat.py \
    --config-name "$CONFIG_NAME" \
    --max-steps "$MAX_STEPS" \
    --batch-size "$BATCH_SIZE" \
    --num-gpus "$NUM_GPUS" \
    --learning-rate "$LEARNING_RATE" \
    --cache-dir "$CACHE_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --log-dir "$LOG_DIR" \
    --no-confirm \
    > "${LOG_DIR}/production_training_$(date +%Y%m%d_%H%M%S).log" 2>&1 &

PID=$!
echo "Training started with PID: $PID"
echo ""
echo "Monitor training with:"
echo "  Logs:   tail -f ${LOG_DIR}/production_training_*.log"
echo "  GPUs:   watch -n 1 nvidia-smi"
echo "  Job:    ps aux | grep train_atat.py"
echo ""
echo "Checkpoints saved to: /media/scratch/adele/mdlm_fresh/checkpoints/"
echo ""

# Optional: tail the log in foreground
# tail -f "${LOG_DIR}/production_training_$(date +%Y%m%d_%H%M%S).log"
