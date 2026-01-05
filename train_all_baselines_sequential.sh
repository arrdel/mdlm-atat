#!/bin/bash
# Sequential Baseline Training - All 4 Baselines (Improved Version)
# Runs D3PM and SEDD in parallel on both GPUs
# Then runs AR on first free GPU, then MDLM on next free GPU
# Usage: ./train_all_baselines_sequential.sh [max_steps] [test_mode]
# Examples:
#   ./train_all_baselines_sequential.sh 10 test    # Quick test with 10 steps and small dataset
#   ./train_all_baselines_sequential.sh 10000      # Full 10K steps with full dataset

set -e

# Set up trap to handle Ctrl+C gracefully
trap 'echo "Script interrupted. Waiting for background processes..."; wait; exit 130' INT TERM

MAX_STEPS=${1:-10000}
OUTPUT_DIR="/media/scratch/adele/mdlm_fresh/outputs"

echo "========================================================================"
echo "Sequential Baseline Training - All 4 Baselines (10K Steps on Full Data)"
echo "========================================================================"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Max steps: $MAX_STEPS"
echo ""
echo "Phase 1: AR (GPU 0) + D3PM (GPU 1) in parallel"
echo "Phase 2: SEDD (first free GPU) after one completes"
echo "Phase 3: MDLM (next free GPU) after another completes"
echo ""
echo "Checkpoints stored in:"
echo "  AR: /media/scratch/adele/mdlm_fresh/outputs/baselines/ar_transformer/"
echo "  D3PM: /media/scratch/adele/mdlm_fresh/outputs/baselines/d3pm_small/"
echo "  SEDD: /media/scratch/adele/mdlm_fresh/outputs/baselines/sedd/"
echo "  MDLM: /media/scratch/adele/mdlm_fresh/outputs/baselines/mdlm_uniform/"
echo "========================================================================"

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# Navigate to project root
cd /home/adelechinda/home/projects/mdlm

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mdlm-atat

echo "Using FULL datasets (no --max-examples limit)"

echo ""
echo "Phase 1: Starting AR and D3PM in parallel..."
echo "========================================================================"

# Change to mdlm_atat directory for AR
cd /home/adelechinda/home/projects/mdlm/mdlm_atat

# Launch AR on GPU 0
echo "Launching AR Transformer on GPU 0..."
CUDA_VISIBLE_DEVICES=0 python baselines/ar_transformer/train_ar_simple.py \
  --max-steps $MAX_STEPS \
  --batch-size 4 \
  --num-gpus 1 \
  --no-wandb \
  --data-path /media/scratch/adele/mdlm_fresh/data_cache/openwebtext-train_train_bs1024_wrapped.dat \
  > "$OUTPUT_DIR/ar_latest.log" 2>&1 &
AR_PID=$!
echo "  ✓ AR started (PID: $AR_PID, GPU 0)"

# Go back to project root for D3PM
cd /home/adelechinda/home/projects/mdlm

# Launch D3PM on GPU 1
echo "Launching D3PM on GPU 1..."
CUDA_VISIBLE_DEVICES=1 python mdlm_atat/baselines/d3pm/train_d3pm.py \
  --config d3pm_small_config.yaml \
  --max-steps $MAX_STEPS \
  --num-gpus 1 \
  --no-wandb \
  > "$OUTPUT_DIR/d3pm_latest.log" 2>&1 &
D3PM_PID=$!
echo "  ✓ D3PM started (PID: $D3PM_PID, GPU 1)"

echo ""
echo "Monitoring Phase 1 progress..."
echo "  AR log: tail -f $OUTPUT_DIR/ar_latest.log"
echo "  D3PM log: tail -f $OUTPUT_DIR/d3pm_latest.log"
echo ""

# Monitor which process finishes first
echo "Waiting for first baseline to complete..."
FIRST_FREE_GPU=""
AR_EXIT=0
D3PM_EXIT=0

while [[ -z "$FIRST_FREE_GPU" ]]; do
    # Check if AR finished
    if ! kill -0 $AR_PID 2>/dev/null; then
        wait $AR_PID 2>/dev/null || true
        AR_EXIT=$?
        FIRST_FREE_GPU=0
        echo "  ✓ AR completed (exit code: $AR_EXIT) - GPU 0 free"
        break
    fi
    
    # Check if D3PM finished
    if ! kill -0 $D3PM_PID 2>/dev/null; then
        wait $D3PM_PID 2>/dev/null || true
        D3PM_EXIT=$?
        FIRST_FREE_GPU=1
        echo "  ✓ D3PM completed (exit code: $D3PM_EXIT) - GPU 1 free"
        break
    fi
    
    sleep 2
done

echo ""
echo "Phase 2: Starting SEDD on GPU $FIRST_FREE_GPU..."
echo "========================================================================"

# Launch SEDD on first free GPU
echo "Launching SEDD on GPU $FIRST_FREE_GPU..."
CUDA_VISIBLE_DEVICES=$FIRST_FREE_GPU python mdlm_atat/baselines/sedd/train_sedd.py \
  --max-steps $MAX_STEPS \
  --num-gpus 1 \
  --no-wandb \
  > "$OUTPUT_DIR/sedd_latest.log" 2>&1 &
SEDD_PID=$!
echo "  ✓ SEDD started (PID: $SEDD_PID, GPU $FIRST_FREE_GPU)"

echo ""
echo "Monitoring Phase 2 progress..."
echo "  AR log: tail -f $OUTPUT_DIR/ar_latest.log"
echo ""

# Wait for second GPU to become free
echo "Waiting for second baseline to complete..."
SECOND_FREE_GPU=""
SEDD_EXIT=0

if [[ "$FIRST_FREE_GPU" == "0" ]]; then
    # AR finished first, wait for D3PM
    while kill -0 $D3PM_PID 2>/dev/null; do
        sleep 2
    done
    wait $D3PM_PID 2>/dev/null || true
    D3PM_EXIT=$?
    SECOND_FREE_GPU=1
    echo "  ✓ D3PM completed (exit code: $D3PM_EXIT) - GPU 1 free"
else
    # D3PM finished first, wait for AR
    while kill -0 $AR_PID 2>/dev/null; do
        sleep 2
    done
    wait $AR_PID 2>/dev/null || true
    AR_EXIT=$?
    SECOND_FREE_GPU=0
    echo "  ✓ AR completed (exit code: $AR_EXIT) - GPU 0 free"
fi

echo ""
echo "Phase 3: Starting MDLM on GPU $SECOND_FREE_GPU..."
echo "========================================================================"

# Go back to project root for MDLM
cd /home/adelechinda/home/projects/mdlm

# Launch MDLM on second free GPU
echo "Launching MDLM on GPU $SECOND_FREE_GPU..."
CUDA_VISIBLE_DEVICES=$SECOND_FREE_GPU python mdlm_atat/baselines/mdlm/train_mdlm_baseline.py \
  --max-steps $MAX_STEPS \
  --num-gpus 1 \
  --no-wandb \
  > "$OUTPUT_DIR/mdlm_latest.log" 2>&1 &
MDLM_PID=$!
echo "  ✓ MDLM started (PID: $MDLM_PID, GPU $SECOND_FREE_GPU)"

echo ""
echo "Monitoring Phase 3 progress..."
echo "  SEDD log: tail -f $OUTPUT_DIR/sedd_latest.log"
echo "  MDLM log: tail -f $OUTPUT_DIR/mdlm_latest.log"
echo ""

# Wait for all remaining processes to complete
echo "Waiting for SEDD and MDLM to complete..."
while kill -0 $SEDD_PID 2>/dev/null; do
    sleep 2
done
wait $SEDD_PID 2>/dev/null || true
SEDD_EXIT=$?
echo "  ✓ SEDD completed (exit code: $SEDD_EXIT)"

while kill -0 $MDLM_PID 2>/dev/null; do
    sleep 2
done
wait $MDLM_PID 2>/dev/null || true
MDLM_EXIT=$?
echo "  ✓ MDLM completed (exit code: $MDLM_EXIT)"

# Final status
echo ""
echo "========================================================================"
echo "All Baselines Training Complete!"
echo "========================================================================"
echo "Completed: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "Exit codes:"
echo "  D3PM: $D3PM_EXIT"
echo "  SEDD: $SEDD_EXIT"
echo "  AR:   $AR_EXIT"
echo "  MDLM: $MDLM_EXIT"
echo ""

# Check for any failures
FAILED=0
if [[ $D3PM_EXIT -ne 0 ]]; then
    echo "❌ D3PM failed with exit code $D3PM_EXIT"
    FAILED=1
else
    echo "✓ D3PM succeeded"
fi

if [[ $SEDD_EXIT -ne 0 ]]; then
    echo "❌ SEDD failed with exit code $SEDD_EXIT"
    FAILED=1
else
    echo "✓ SEDD succeeded"
fi

if [[ $AR_EXIT -ne 0 ]]; then
    echo "❌ AR failed with exit code $AR_EXIT"
    FAILED=1
else
    echo "✓ AR succeeded"
fi

if [[ $MDLM_EXIT -ne 0 ]]; then
    echo "❌ MDLM failed with exit code $MDLM_EXIT"
    FAILED=1
else
    echo "✓ MDLM succeeded"
fi

echo ""
echo "Log files (always use 'latest' naming - overwritten each run):"
echo "  D3PM: $OUTPUT_DIR/d3pm_latest.log"
echo "  SEDD: $OUTPUT_DIR/sedd_latest.log"
echo "  AR:   $OUTPUT_DIR/ar_latest.log"
echo "  MDLM: $OUTPUT_DIR/mdlm_latest.log"
echo ""
echo "Checkpoints saved in:"
echo "  D3PM: /media/scratch/adele/mdlm_fresh/outputs/baselines/d3pm_small/"
echo "  SEDD: /media/scratch/adele/mdlm_fresh/outputs/baselines/sedd/"
echo "  AR:   /media/scratch/adele/mdlm_fresh/outputs/baselines/ar_transformer/"
echo "  MDLM: /media/scratch/adele/mdlm_fresh/outputs/baselines/mdlm_uniform/"
echo "========================================================================"

if [[ $FAILED -eq 0 ]]; then
    echo "✓ All baselines completed successfully!"
    exit 0
else
    echo "❌ Some baselines failed. Check logs above."
    exit 1
fi

