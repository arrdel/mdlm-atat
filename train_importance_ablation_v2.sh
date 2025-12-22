#!/bin/bash
# Importance Estimator Ablation Study - Direct Training Launcher (v2)
# Runs mdlm/main.py directly with absolute paths
#
# Usage:
#   ./train_importance_ablation_v2.sh full 0,1 500000

set -e

# Configuration
PROJECT_ROOT="/home/adelechinda/home/projects/mdlm"
PYTHON="/home/adelechinda/miniconda3/envs/mdlm-atat/bin/python"

# Arguments
VARIANT="${1:-full}"
GPUS="${2:-0,1}"
MAX_STEPS="${3:-500000}"
BATCH_SIZE=32
LEARNING_RATE=3e-4

# Derived paths
CONFIG_NAME="atat/importance_ablation_${VARIANT}"
OUTPUT_DIR="/media/scratch/adele/mdlm_fresh/outputs/importance_ablation/${VARIANT}"
LOG_DIR="/media/scratch/adele/mdlm_fresh/logs/importance_ablation/${VARIANT}"
CONFIG_PATH="$PROJECT_ROOT/mdlm_atat/configs"

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# Validate variant
VALID_VARIANTS=("full" "frequency_only" "learned_only" "uniform")
if [[ ! " ${VALID_VARIANTS[@]} " =~ " ${VARIANT} " ]]; then
    echo "Error: Invalid variant '$VARIANT'"
    echo "Valid variants: ${VALID_VARIANTS[@]}"
    exit 1
fi

echo "=========================================="
echo "Importance Estimator Ablation Study (v2)"
echo "=========================================="
echo "Variant: $VARIANT"
echo "Config: $CONFIG_NAME"
echo "GPUs: $GPUS"
echo "Batch size: $BATCH_SIZE (per GPU)"
echo "Max steps: $MAX_STEPS"
echo "Output: $OUTPUT_DIR"
echo "Logs: $LOG_DIR"
echo ""

# Set CUDA devices
export CUDA_VISIBLE_DEVICES=$GPUS

# Calculate number of GPUs
IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
NUM_GPUS=${#GPU_ARRAY[@]}
GLOBAL_BATCH_SIZE=$((BATCH_SIZE * NUM_GPUS))

echo "Starting training..."
echo "  Python: $PYTHON"
echo "  Main: $PROJECT_ROOT/mdlm/main.py"
echo "  Config path: $CONFIG_PATH"
echo "  Number of GPUs: $NUM_GPUS"
echo "  Global batch size: $GLOBAL_BATCH_SIZE"
echo ""

# Run training directly with Python (not as module)
cd "$PROJECT_ROOT"
PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH" $PYTHON mdlm/main.py \
  --config-path="$CONFIG_PATH" \
  --config-name="$CONFIG_NAME" \
  trainer.max_steps="$MAX_STEPS" \
  trainer.val_check_interval=50000 \
  trainer.log_every_n_steps=100 \
  trainer.accelerator=cuda \
  trainer.devices=$NUM_GPUS \
  loader.batch_size=$BATCH_SIZE \
  loader.global_batch_size=$GLOBAL_BATCH_SIZE \
  optim.lr=$LEARNING_RATE \
  data.cache_dir=/media/scratch/adele/mdlm_fresh/data_cache \
  hydra.run.dir="$OUTPUT_DIR" \
  hydra.job.name="importance_ablation_${VARIANT}" \
  2>&1 | tee "$LOG_DIR/train.log"

EXIT_CODE=$?
echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Training completed successfully!"
else
    echo "✗ Training failed with exit code $EXIT_CODE"
fi
echo "=========================================="
echo "Output: $OUTPUT_DIR"
echo "Logs: $LOG_DIR/train.log"
echo ""

exit $EXIT_CODE
