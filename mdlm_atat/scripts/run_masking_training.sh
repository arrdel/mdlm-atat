#!/bin/bash
# Masking Strategy Ablation - Training Runner
#
# Orchestrates training of 4 masking strategy variants:
# 1. Balanced (ours): (1-t)*g_inv + t*g_prop
# 2. Importance-Proportional: Always mask important tokens more
# 3. Importance-Inverse: Always preserve important tokens
# 4. Time-Only (control): Uniform masking, no importance
#
# All variants use full ATAT importance estimator from importance_estimator phase.
#
# Usage:
#   ./run_masking_training.sh [OPTIONS]
#
# Examples:
#   # Train single strategy on debug dataset
#   ./run_masking_training.sh --strategy balanced --dataset-preset debug
#
#   # Train all strategies on validation dataset
#   ./run_masking_training.sh --all-strategies --dataset-preset validation
#
#   # Production run with custom settings
#   ./run_masking_training.sh --all-strategies --dataset-preset production \
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
STRATEGY=""
ALL_STRATEGIES=false
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
            --strategy)
                STRATEGY="$2"
                shift 2
                ;;
            --all-strategies)
                ALL_STRATEGIES=true
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
Phase 1B: Masking Strategy Ablation - Training Runner

Usage:
    ./run_phase1b_training.sh [OPTIONS]

Options:
    --strategy STRATEGY          Train single strategy (balanced, proportional, inverse, time_only)
    --all-strategies            Train all 4 masking strategies
    --dataset-preset PRESET     Dataset preset: debug, validation, production [default: debug]
    --max-steps STEPS           Maximum training steps [default: 500000]
    --num-gpus NUM              Number of GPUs [default: 2]
    --batch-size SIZE           Batch size [default: 256]
    --output-dir DIR            Output directory for checkpoints
    --resume                    Resume from checkpoint
    --use-wandb                 Enable Weights & Biases logging
    --dry-run                   Print commands without executing
    --help                      Print this help message

Examples:
    # Train single strategy
    ./run_phase1b_training.sh --strategy balanced --dataset-preset validation

    # Train all strategies
    ./run_phase1b_training.sh --all-strategies --dataset-preset validation

    # Production run
    ./run_phase1b_training.sh --all-strategies --dataset-preset production \
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
    if [ "$STRATEGY" == "" ] && [ "$ALL_STRATEGIES" = false ]; then
        echo -e "${RED}Error: Must specify --strategy or --all-strategies${NC}"
        print_help
        exit 1
    fi

    # Construct Python command with conda environment
    PYTHON_CMD="conda run -n mdlm-atat python $SCRIPT_DIR/training/train_masking.py"
    
    # Add arguments
    if [ "$ALL_STRATEGIES" = true ]; then
        PYTHON_CMD="$PYTHON_CMD --all-strategies"
    else
        PYTHON_CMD="$PYTHON_CMD --strategy $STRATEGY"
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
        PYTHON_CMD="$PYTHON_CMD --wandb-project phase1b-masking"
    fi

    # Print summary
    print_header "MASKING STRATEGY ABLATION - TRAINING"
    echo -e "${YELLOW}Configuration:${NC}"
    echo "  Strategies:        $([ "$ALL_STRATEGIES" = true ] && echo "All (balanced, proportional, inverse, time_only)" || echo "$STRATEGY")"
    echo "  Dataset preset:    $DATASET_PRESET"
    echo "  Max steps:         $MAX_STEPS"
    echo "  Batch size:        $BATCH_SIZE"
    echo "  Number of GPUs:    $NUM_GPUS"
    echo "  WandB logging:     $([ "$USE_WANDB" = true ] && echo "Enabled" || echo "Disabled")"
    echo ""

    # Run training
    run_command "$PYTHON_CMD"

    echo ""
    echo -e "${GREEN}✓ Masking Strategy training completed${NC}"
}

main "$@"
