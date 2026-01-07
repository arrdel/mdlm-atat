#!/bin/bash
# Importance Estimator Ablation - Training Runner
# 
# Orchestrates training of 4 importance estimator variants:
# 1. Full (ours): E-ATAT (exponential importance weighted by frequency)
# 2. Frequency-Only: Uses only token frequency (no learning)
# 3. Learned-Only: Uses only learned importance scores (no frequency)
# 4. Uniform (control): No importance signal, uniform masking
#
# Usage:
#   ./run_importance_estimator_training.sh [OPTIONS]
#
# Examples:
#   # Train single variant on debug dataset
#   ./run_importance_estimator_training.sh --variant full --dataset-preset debug
#
#   # Train all variants on validation dataset
#   ./run_importance_estimator_training.sh --all-variants --dataset-preset validation
#
#   # Production run with custom settings
#   ./run_importance_estimator_training.sh --all-variants --dataset-preset production \
#     --max-steps 500000 --num-gpus 8 --batch-size 256

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
VARIANT=""
ALL_VARIANTS=false
DATASET_PRESET="debug"
MAX_STEPS=500000
NUM_GPUS=2
BATCH_SIZE=256
OUTPUT_DIR=""
RESUME=false
USE_WANDB=false
DRY_RUN=false

# Parse arguments
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --variant)
                VARIANT="$2"
                shift 2
                ;;
            --all-variants)
                ALL_VARIANTS=true
                shift
                ;;
            --dataset-preset)
                DATASET_PRESET="$2"
                shift 2
                ;;
            --max-steps)
                MAX_STEPS="$2"
                shift 2
                ;;
            --num-gpus)
                NUM_GPUS="$2"
                shift 2
                ;;
            --batch-size)
                BATCH_SIZE="$2"
                shift 2
                ;;
            --output-dir)
                OUTPUT_DIR="$2"
                shift 2
                ;;
            --resume)
                RESUME=true
                shift
                ;;
            --use-wandb)
                USE_WANDB=true
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --help)
                print_help
                exit 0
                ;;
            *)
                echo -e "${RED}Unknown option: $1${NC}"
                print_help
                exit 1
                ;;
        esac
    done
}

print_help() {
    cat << EOF
Phase 1A: Importance Estimator Ablation - Training Runner

Usage:
    ./run_phase1a_training.sh [OPTIONS]

Options:
    --variant VARIANT             Train single variant (full, frequency_only, learned_only, uniform)
    --all-variants               Train all 4 importance variants
    --dataset-preset PRESET      Dataset preset: debug, validation, production [default: debug]
    --max-steps STEPS            Maximum training steps [default: 500000]
    --num-gpus NUM               Number of GPUs [default: 2]
    --batch-size SIZE            Batch size [default: 256]
    --output-dir DIR             Output directory for checkpoints
    --resume                     Resume from checkpoint
    --use-wandb                  Enable Weights & Biases logging
    --dry-run                    Print commands without executing
    --help                       Print this help message

Examples:
    # Train single variant
    ./run_phase1a_training.sh --variant full --dataset-preset validation

    # Train all variants
    ./run_phase1a_training.sh --all-variants --dataset-preset validation

    # Production run
    ./run_phase1a_training.sh --all-variants --dataset-preset production \
      --max-steps 500000 --num-gpus 8 --batch-size 256

EOF
}

run_command() {
    local cmd="$1"
    if [ "$DRY_RUN" = true ]; then
        echo -e "${BLUE}[DRY-RUN]${NC} $cmd"
    else
        echo -e "${GREEN}[RUNNING]${NC} $cmd"
        eval "$cmd"
    fi
}

print_header() {
    local message="$1"
    echo ""
    echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}${message}${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
    echo ""
}

main() {
    parse_arguments "$@"

    # Validate arguments
    if [ "$VARIANT" == "" ] && [ "$ALL_VARIANTS" = false ]; then
        echo -e "${RED}Error: Must specify --variant or --all-variants${NC}"
        print_help
        exit 1
    fi

    # Adjust batch size based on dataset preset if not explicitly set
    if [ "$BATCH_SIZE" == "256" ]; then  # Only if using default batch size
        case $DATASET_PRESET in
            debug)
                BATCH_SIZE=8
                ;;
            validation)
                BATCH_SIZE=32
                ;;
            production)
                BATCH_SIZE=256
                ;;
        esac
    fi

    # Construct Python command
    PYTHON_CMD="python $SCRIPT_DIR/training/train_importance_estimator.py"
    
    # Add arguments
    if [ "$ALL_VARIANTS" = true ]; then
        PYTHON_CMD="$PYTHON_CMD --all"
    else
        PYTHON_CMD="$PYTHON_CMD --variant $VARIANT"
    fi

    PYTHON_CMD="$PYTHON_CMD --dataset-preset $DATASET_PRESET"
    PYTHON_CMD="$PYTHON_CMD --max-steps $MAX_STEPS"
    PYTHON_CMD="$PYTHON_CMD --num-gpus $NUM_GPUS"
    PYTHON_CMD="$PYTHON_CMD --batch-size $BATCH_SIZE"

    if [ ! -z "$OUTPUT_DIR" ]; then
        PYTHON_CMD="$PYTHON_CMD --output-dir $OUTPUT_DIR"
    fi

    if [ "$RESUME" = true ]; then
        PYTHON_CMD="$PYTHON_CMD --resume"
    fi

    if [ "$USE_WANDB" = true ]; then
        PYTHON_CMD="$PYTHON_CMD --wandb-project phase1a-importance"
    fi

    # Print summary
    print_header "IMPORTANCE ESTIMATOR ABLATION - TRAINING"
    echo -e "${YELLOW}Configuration:${NC}"
    echo "  Variants:          $([ "$ALL_VARIANTS" = true ] && echo "All (full, frequency_only, learned_only, uniform)" || echo "$VARIANT")"
    echo "  Dataset preset:    $DATASET_PRESET"
    echo "  Max steps:         $MAX_STEPS"
    echo "  Batch size:        $BATCH_SIZE"
    echo "  Number of GPUs:    $NUM_GPUS"
    echo "  WandB logging:     $([ "$USE_WANDB" = true ] && echo "Enabled" || echo "Disabled")"
    echo ""

    # Run training
    run_command "$PYTHON_CMD"

    echo ""
    echo -e "${GREEN}✓ Importance Estimator training completed${NC}"
}

main "$@"
