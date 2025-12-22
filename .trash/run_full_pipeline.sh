#!/bin/bash
#
# Run Complete ATAT Pipeline on Different Model Sizes
#
# Usage:
#   bash scripts/run_full_pipeline.sh [MODEL_SIZE] [DATASET]
#
# MODEL_SIZE: tiny, small, medium (default: small)
# DATASET: synthetic_tiny, openwebtext, wikitext103 (default: openwebtext)
#
# Examples:
#   bash scripts/run_full_pipeline.sh small openwebtext
#   bash scripts/run_full_pipeline.sh medium wikitext103
#   bash scripts/run_full_pipeline.sh tiny synthetic_tiny

set -e  # Exit on error

# Configuration
MODEL_SIZE="${1:-small}"
DATASET="${2:-openwebtext}"
NUM_GPUS="${3:-6}"
CONDA_ENV="mdlm-atat"

# Directories
PROJECT_DIR="/home/adelechinda/home/projects/mdlm/mdlm_atat"
CACHE_DIR="/media/scratch/adele/mdlm_fresh/data_cache"
OUTPUT_DIR="/media/scratch/adele/mdlm_fresh/outputs"
LOG_DIR="/media/scratch/adele/mdlm_fresh/logs"

# Model-specific configurations
case $MODEL_SIZE in
    tiny)
        MAX_STEPS=10000
        VAL_INTERVAL=500
        CONFIG_NAME="atat/tiny"
        ;;
    small)
        MAX_STEPS=100000
        VAL_INTERVAL=5000
        CONFIG_NAME="atat/small"
        ;;
    medium)
        MAX_STEPS=200000
        VAL_INTERVAL=10000
        CONFIG_NAME="atat/medium"
        ;;
    *)
        echo "Error: Unknown model size '$MODEL_SIZE'"
        echo "Valid options: tiny, small, medium"
        exit 1
        ;;
esac

# Dataset-specific configurations
case $DATASET in
    synthetic_tiny)
        TRAIN_DATA="synthetic_tiny"
        VALID_DATA="synthetic_tiny"
        # Override for synthetic data
        MAX_STEPS=200
        VAL_INTERVAL=50
        ;;
    openwebtext)
        TRAIN_DATA="openwebtext"
        VALID_DATA="wikitext103"
        ;;
    wikitext103)
        TRAIN_DATA="wikitext103"
        VALID_DATA="wikitext103"
        ;;
    *)
        echo "Error: Unknown dataset '$DATASET'"
        echo "Valid options: synthetic_tiny, openwebtext, wikitext103"
        exit 1
        ;;
esac

# Setup
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="atat_${MODEL_SIZE}_${DATASET}_${TIMESTAMP}"
PIPELINE_LOG="${LOG_DIR}/pipeline_${RUN_NAME}.log"

mkdir -p "$LOG_DIR"
mkdir -p "$OUTPUT_DIR"

echo "============================================================" | tee "$PIPELINE_LOG"
echo "  ATAT FULL PIPELINE - ${MODEL_SIZE^^} MODEL" | tee -a "$PIPELINE_LOG"
echo "============================================================" | tee -a "$PIPELINE_LOG"
echo "" | tee -a "$PIPELINE_LOG"
echo "Configuration:" | tee -a "$PIPELINE_LOG"
echo "  Model: $MODEL_SIZE" | tee -a "$PIPELINE_LOG"
echo "  Dataset: $DATASET (train=$TRAIN_DATA, valid=$VALID_DATA)" | tee -a "$PIPELINE_LOG"
echo "  GPUs: $NUM_GPUS" | tee -a "$PIPELINE_LOG"
echo "  Max steps: $MAX_STEPS" | tee -a "$PIPELINE_LOG"
echo "  Val interval: $VAL_INTERVAL" | tee -a "$PIPELINE_LOG"
echo "  Log: $PIPELINE_LOG" | tee -a "$PIPELINE_LOG"
echo "" | tee -a "$PIPELINE_LOG"

cd "$PROJECT_DIR"

# ============================================================
# STEP 1: TRAINING
# ============================================================
echo "[1/3] TRAINING" | tee -a "$PIPELINE_LOG"
echo "------------------------------------------------------------" | tee -a "$PIPELINE_LOG"

conda run -n "$CONDA_ENV" python ../mdlm/main.py \
    --config-path ../mdlm_atat/configs \
    --config-name "$CONFIG_NAME" \
    data.train="$TRAIN_DATA" \
    data.valid="$VALID_DATA" \
    trainer.devices="$NUM_GPUS" \
    trainer.max_steps="$MAX_STEPS" \
    trainer.val_check_interval="$VAL_INTERVAL" \
    wandb.name="${RUN_NAME}" \
    wandb.offline=false 2>&1 | tee -a "$PIPELINE_LOG"

TRAIN_EXIT_CODE=${PIPESTATUS[0]}

if [ $TRAIN_EXIT_CODE -ne 0 ]; then
    echo "✗ Training failed with exit code $TRAIN_EXIT_CODE" | tee -a "$PIPELINE_LOG"
    exit $TRAIN_EXIT_CODE
fi

# Find the best checkpoint
CHECKPOINT=$(find "$OUTPUT_DIR" -name "best.ckpt" -type f -mmin -120 | head -1)

if [ -z "$CHECKPOINT" ]; then
    echo "✗ Could not find checkpoint" | tee -a "$PIPELINE_LOG"
    exit 1
fi

echo "✓ Training completed" | tee -a "$PIPELINE_LOG"
echo "  Checkpoint: $CHECKPOINT" | tee -a "$PIPELINE_LOG"
echo "" | tee -a "$PIPELINE_LOG"

# ============================================================
# STEP 2: EVALUATION
# ============================================================
echo "[2/3] EVALUATION" | tee -a "$PIPELINE_LOG"
echo "------------------------------------------------------------" | tee -a "$PIPELINE_LOG"

conda run -n "$CONDA_ENV" python scripts/eval_atat.py \
    --checkpoint "$CHECKPOINT" \
    --mode both \
    --datasets "$VALID_DATA" \
    --num-samples 20 \
    --config-name "$CONFIG_NAME" \
    --num-gpus "$NUM_GPUS" \
    --cache-dir "$CACHE_DIR" \
    --log-dir "$LOG_DIR" \
    --wandb-offline 2>&1 | tee -a "$PIPELINE_LOG"

EVAL_EXIT_CODE=${PIPESTATUS[0]}

if [ $EVAL_EXIT_CODE -ne 0 ]; then
    echo "✗ Evaluation failed with exit code $EVAL_EXIT_CODE" | tee -a "$PIPELINE_LOG"
    echo "  Continuing to ablation studies..." | tee -a "$PIPELINE_LOG"
fi

echo "" | tee -a "$PIPELINE_LOG"

# ============================================================
# STEP 3: ABLATION STUDIES
# ============================================================
echo "[3/3] ABLATION STUDIES" | tee -a "$PIPELINE_LOG"
echo "------------------------------------------------------------" | tee -a "$PIPELINE_LOG"

# Scale ablation steps based on model size
ABLATION_STEPS=$((MAX_STEPS / 20))
ABLATION_VAL_INTERVAL=$((VAL_INTERVAL / 2))

conda run -n "$CONDA_ENV" python scripts/run_ablation.py \
    --study all \
    --max-steps "$ABLATION_STEPS" \
    --val-interval "$ABLATION_VAL_INTERVAL" \
    --num-gpus "$NUM_GPUS" \
    --output-dir "$OUTPUT_DIR/ablations" \
    --log-dir "$LOG_DIR" 2>&1 | tee -a "$PIPELINE_LOG"

ABLATION_EXIT_CODE=${PIPESTATUS[0]}

if [ $ABLATION_EXIT_CODE -ne 0 ]; then
    echo "✗ Ablation studies failed with exit code $ABLATION_EXIT_CODE" | tee -a "$PIPELINE_LOG"
fi

# ============================================================
# SUMMARY
# ============================================================
echo "" | tee -a "$PIPELINE_LOG"
echo "============================================================" | tee -a "$PIPELINE_LOG"
echo "                    PIPELINE COMPLETE" | tee -a "$PIPELINE_LOG"
echo "============================================================" | tee -a "$PIPELINE_LOG"
echo "" | tee -a "$PIPELINE_LOG"
echo "Results:" | tee -a "$PIPELINE_LOG"
echo "  Training: $([ $TRAIN_EXIT_CODE -eq 0 ] && echo '✓' || echo '✗')" | tee -a "$PIPELINE_LOG"
echo "  Evaluation: $([ $EVAL_EXIT_CODE -eq 0 ] && echo '✓' || echo '✗')" | tee -a "$PIPELINE_LOG"
echo "  Ablations: $([ $ABLATION_EXIT_CODE -eq 0 ] && echo '✓' || echo '✗')" | tee -a "$PIPELINE_LOG"
echo "" | tee -a "$PIPELINE_LOG"
echo "Files:" | tee -a "$PIPELINE_LOG"
echo "  Checkpoint: $CHECKPOINT" | tee -a "$PIPELINE_LOG"
echo "  Full log: $PIPELINE_LOG" | tee -a "$PIPELINE_LOG"
echo "" | tee -a "$PIPELINE_LOG"

exit 0
