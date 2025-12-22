#!/bin/bash
# Importance Ablation Training - Complete Study
# Trains 4 variants of importance estimator on OpenWebText
# 
# Usage:
#   bash run_importance_ablation.sh full "0,1"        # Run full variant on GPUs 0,1
#   bash run_importance_ablation.sh frequency_only "0,1"
#   bash run_importance_ablation.sh learned_only "0,1"
#   bash run_importance_ablation.sh uniform "0,1"
#
# Or run all 4 serially:
#   bash run_importance_ablation.sh all "0,1"

set -e

VARIANT=${1:-"full"}
GPUS=${2:-"0,1"}

PROJECT_ROOT="/home/adelechinda/home/projects/mdlm"
PYTHON="/home/adelechinda/miniconda3/envs/mdlm-atat/bin/python"
CONFIG_PATH="$PROJECT_ROOT/mdlm_atat/configs"

# Use consistent naming: importance_ablation
OUTPUT_BASE="/media/scratch/adele/mdlm_fresh/importance_ablation/outputs"
LOG_BASE="/media/scratch/adele/mdlm_fresh/importance_ablation/logs"

# Helper function to run a single variant
run_variant() {
    local v=$1
    local gpus=$2
    
    local variant_out="$OUTPUT_BASE/$v"
    local variant_log="$LOG_BASE/$v"
    
    mkdir -p "$variant_out" "$variant_log"
    
    echo "=========================================="
    echo "Training: Importance Ablation - $v"
    echo "=========================================="
    echo "Variant: $v"
    echo "GPUs: $gpus"
    echo "Config: atat/importance_ablation_$v"
    echo "Output: $variant_out"
    echo "Logs: $variant_log"
    echo ""
    
    # Set environment
    export CUDA_VISIBLE_DEVICES=$gpus
    export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
    
    # Change to project root
    cd "$PROJECT_ROOT"
    
    # Run training
    $PYTHON -c "
import os
import sys
os.chdir('$PROJECT_ROOT')
from hydra.core.global_hydra import GlobalHydra
GlobalHydra.instance().clear()
from mdlm.main import main

sys.argv = [
    'mdlm.main',
    '--config-path=$CONFIG_PATH',
    '--config-name=atat/importance_ablation_$v',
    'trainer.max_steps=500000',
    'trainer.val_check_interval=50000',
    'trainer.log_every_n_steps=100',
    'trainer.accelerator=cuda',
    'trainer.devices=2',
    'loader.batch_size=32',
    'loader.global_batch_size=64',
    'optim.lr=3e-4',
    'data.cache_dir=/media/scratch/adele/mdlm_fresh/data_cache',
    'hydra.run.dir=$variant_out',
    'hydra.job.chdir=false',
]

main()
" 2>&1 | tee "$variant_log/train.log"

    echo ""
    echo "✓ Training completed: $v"
    echo "  Output: $variant_out"
    echo "  Logs: $variant_log"
    echo ""
}

# Main execution
if [[ "$VARIANT" == "all" ]]; then
    echo "Running all 4 importance ablation variants serially..."
    echo "Expected time: ~12-15 days total (3-4 days per variant on 2 GPUs)"
    echo ""
    
    run_variant "full" "$GPUS"
    run_variant "frequency_only" "$GPUS"
    run_variant "learned_only" "$GPUS"
    run_variant "uniform" "$GPUS"
    
    echo "=========================================="
    echo "All training complete!"
    echo "=========================================="
    echo "Results saved to: $OUTPUT_BASE"
    echo "Logs saved to: $LOG_BASE"
    echo ""
    echo "Next step: Run evaluation"
    echo "  bash mdlm_atat/scripts/eval_importance_ablation.sh"
else
    # Single variant
    run_variant "$VARIANT" "$GPUS"
fi
