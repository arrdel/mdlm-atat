#!/bin/bash
# Phase 1A: Importance Estimator Variants Ablation Study
# Launches 4 training jobs in parallel for:
# 1. Full ATAT (learned + frequency, 0.7/0.3)
# 2. Frequency-only (no learned component)
# 3. Learned-only (no frequency prior)
# 4. Uniform baseline (no importance)
#
# Expected runtime: ~12-15 days with 2 x 8-GPU nodes
# Total compute: 1,760 GPU-hours (440 GPU-hours per model)

set -e

# Configuration
PROJECT_ROOT="/home/adelechinda/home/projects/mdlm/mdlm_atat"
SCRIPT_DIR="$PROJECT_ROOT/scripts"
OUTPUT_BASE="/media/scratch/adele/mdlm_fresh/outputs/phase1a_ablations"
LOG_BASE="/media/scratch/adele/mdlm_fresh/logs/phase1a_ablations"

# Training parameters (from NEXT_STEPS.md)
CONFIG_PREFIX="atat/phase1a"
MAX_STEPS=500000
VAL_INTERVAL=50000
LOG_INTERVAL=100
NUM_GPUS=8
BATCH_SIZE=64
LEARNING_RATE=3e-4

# Create output directories
mkdir -p "$OUTPUT_BASE"
mkdir -p "$LOG_BASE"
mkdir -p "$OUTPUT_BASE/checkpoints"
mkdir -p "$OUTPUT_BASE/results"

echo "=========================================="
echo "Phase 1A: Importance Estimator Variants"
echo "=========================================="
echo "Project root: $PROJECT_ROOT"
echo "Output base: $OUTPUT_BASE"
echo "Total models: 4"
echo "GPU count per model: $NUM_GPUS"
echo "Steps per model: $MAX_STEPS"
echo "Estimated time per model: 12-15 days"
echo ""

# Function to run a single training job
run_training_job() {
    local variant=$1
    local config="$CONFIG_PREFIX"_"$variant"
    local output_dir="$OUTPUT_BASE/checkpoints/$variant"
    local log_dir="$LOG_BASE/$variant"
    
    mkdir -p "$output_dir"
    mkdir -p "$log_dir"
    
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] Starting training: $variant"
    echo "  Config: $config"
    echo "  Output: $output_dir"
    echo "  Logs: $log_dir"
    
    python "$SCRIPT_DIR/train_atat.py" \
        --config-name "$config" \
        --max-steps "$MAX_STEPS" \
        --val-interval "$VAL_INTERVAL" \
        --log-interval "$LOG_INTERVAL" \
        --num-gpus "$NUM_GPUS" \
        --batch-size "$BATCH_SIZE" \
        --learning-rate "$LEARNING_RATE" \
        --output-dir "$output_dir" \
        --log-dir "$log_dir" \
        --wandb-project "mdlm-atat-phase1a" \
        --wandb-run-name "phase1a-$variant" \
        --no-confirm \
        2>&1 | tee "$log_dir/train.log"
    
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] Completed training: $variant"
}

# Launch 4 training jobs in parallel
# Note: You may need to adjust GPU allocation based on your cluster setup
# Option 1: Serial execution (recommended for single machine)
# Option 2: Parallel execution with job scheduling

echo ""
echo "Starting training jobs..."
echo "Choose execution mode:"
echo "1. Serial (one after another) - recommended for single machine"
echo "2. Parallel (all 4 at once) - requires 2 x 8-GPU nodes"
echo ""

# Serial execution by default
PARALLEL=${1:-"serial"}

if [[ "$PARALLEL" == "parallel" ]]; then
    echo "Starting PARALLEL execution (requires 16 GPUs total)..."
    
    # Background job management
    run_training_job "full" &
    PID_FULL=$!
    
    run_training_job "frequency_only" &
    PID_FREQ=$!
    
    run_training_job "learned_only" &
    PID_LEARNED=$!
    
    run_training_job "uniform" &
    PID_UNIFORM=$!
    
    # Wait for all jobs
    echo "Waiting for all 4 jobs to complete..."
    wait $PID_FULL && echo "✓ Full ATAT completed"
    wait $PID_FREQ && echo "✓ Frequency-only completed"
    wait $PID_LEARNED && echo "✓ Learned-only completed"
    wait $PID_UNIFORM && echo "✓ Uniform baseline completed"
    
else
    echo "Starting SERIAL execution (runs one after another)..."
    echo "This will take approximately 12-15 weeks."
    echo ""
    
    run_training_job "full"
    run_training_job "frequency_only"
    run_training_job "learned_only"
    run_training_job "uniform"
fi

echo ""
echo "=========================================="
echo "Phase 1A Training Complete"
echo "=========================================="
echo "Checkpoints saved to: $OUTPUT_BASE/checkpoints"
echo "Logs saved to: $LOG_BASE"
echo "Next: Run evaluation and analysis scripts"
echo ""
