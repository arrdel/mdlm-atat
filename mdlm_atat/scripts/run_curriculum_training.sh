#!/bin/bash
# Curriculum Schedule Ablation - Training Runner
#
# Orchestrates training of 4 curriculum schedule variants:
# 1. Default: 0.3-0.7 boundaries (30% easy, 70% hard)
# 2. Early: 0.2-0.6 boundaries (20% easy, 60% hard)
# 3. Late: 0.35-0.8 boundaries (35% easy, 80% hard)
# 4. No Curriculum: No curriculum control, uniform masking
#
# All variants use balanced masking strategy from masking phase and
# full ATAT importance estimator from importance_estimator phase.
#
# Usage:
#   ./run_curriculum_training.sh [OPTIONS]
#
# Examples:
#   # Train single schedule on debug dataset
#   ./run_curriculum_training.sh --schedule default --dataset-preset debug
#
#   # Train all schedules on validation dataset
#   ./run_curriculum_training.sh --all-schedules --dataset-preset validation
#
#   # Production run with custom settings
#   ./run_curriculum_training.sh --all-schedules --dataset-preset production \
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
SCHEDULE=""
ALL_SCHEDULES=false
DATASET_PRESET="debug"
MAX_STEPS=500000
NUM_GPUS=2
BATCH_SIZE=256
LEARNING_RATE=0.001
WARMUP_STEPS=10000
OUTPUT_DIR=""
RESUME=false
USE_WANDB=false
DRY_RUN=false

# Parse arguments
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --schedule)
                SCHEDULE="$2"
                shift 2
                ;;
            --all-schedules)
                ALL_SCHEDULES=true
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
            --learning-rate)
                LEARNING_RATE="$2"
                shift 2
                ;;
            --warmup-steps)
                WARMUP_STEPS="$2"
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
Phase 1C: Curriculum Schedule Ablation - Training Runner

Usage:
    ./run_phase1c_training.sh [OPTIONS]

Options:
    --schedule SCHEDULE         Train single schedule (default, early, late, no_curriculum)
    --all-schedules            Train all 4 curriculum schedules
    --dataset-preset PRESET    Dataset preset: debug, validation, production [default: debug]
    --max-steps STEPS          Maximum training steps [default: 500000]
    --num-gpus NUM             Number of GPUs [default: 2]
    --batch-size SIZE          Batch size [default: 256]
    --learning-rate LR         Learning rate [default: 0.001]
    --warmup-steps STEPS       Warmup steps [default: 10000]
    --output-dir DIR           Output directory for checkpoints
    --resume                   Resume from checkpoint
    --use-wandb                Enable Weights & Biases logging
    --dry-run                  Print commands without executing
    --help                     Print this help message

Examples:
    # Train single schedule
    ./run_phase1c_training.sh --schedule default --dataset-preset validation

    # Train all schedules
    ./run_phase1c_training.sh --all-schedules --dataset-preset validation

    # Production run
    ./run_phase1c_training.sh --all-schedules --dataset-preset production \
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
    if [ "$SCHEDULE" == "" ] && [ "$ALL_SCHEDULES" = false ]; then
        echo -e "${RED}Error: Must specify --schedule or --all-schedules${NC}"
        print_help
        exit 1
    fi

    # Construct Python command
    PYTHON_CMD="python $SCRIPT_DIR/training/train_curriculum_schedules.py"
    
    # Add arguments
    if [ "$ALL_SCHEDULES" = true ]; then
        PYTHON_CMD="$PYTHON_CMD --all-schedules"
    else
        PYTHON_CMD="$PYTHON_CMD --schedule $SCHEDULE"
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
        PYTHON_CMD="$PYTHON_CMD --wandb-project phase1c-curriculum"
    fi

    # Print summary
    print_header "CURRICULUM SCHEDULE ABLATION - TRAINING"
    echo -e "${YELLOW}Configuration:${NC}"
    echo "  Schedules:         $([ "$ALL_SCHEDULES" = true ] && echo "All (default, early, late, no_curriculum)" || echo "$SCHEDULE")"
    echo "  Dataset preset:    $DATASET_PRESET"
    echo "  Max steps:         $MAX_STEPS"
    echo "  Batch size:        $BATCH_SIZE"
    echo "  Number of GPUs:    $NUM_GPUS"
    echo "  WandB logging:     $([ "$USE_WANDB" = true ] && echo "Enabled" || echo "Disabled")"
    echo ""

    # Run training
    run_command "$PYTHON_CMD"

    echo ""
    echo -e "${GREEN}✓ Curriculum Schedule training completed${NC}"
}

main "$@"
