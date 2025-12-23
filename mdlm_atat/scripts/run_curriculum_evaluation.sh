#!/bin/bash
# Curriculum Schedule Ablation - Evaluation Runner
#
# Comprehensive evaluation of all curriculum schedule variants with:
# - Validation PPL and BPD metrics
# - Per-stage loss analysis (easy, medium, hard)
# - Curriculum boundary transition effects
# - Training stability metrics (loss variance, gradient norms)
# - Convergence speed metrics
# - Real-time progress monitoring
#
# Usage:
#   ./run_curriculum_evaluation.sh [OPTIONS]
#
# Examples:
#   # Evaluate single schedule
#   ./run_curriculum_evaluation.sh --schedule default --checkpoint-dir ./checkpoints
#
#   # Evaluate all schedules with detailed analysis
#   ./run_curriculum_evaluation.sh --all-schedules --checkpoint-dir ./checkpoints \
#     --analyze-boundaries --stability-analysis
#
#   # Real-time monitoring
#   ./run_curriculum_evaluation.sh --all-schedules --checkpoint-dir ./checkpoints --watch

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
WATCH_MODE=false
CHECKPOINT_DIR=""
OUTPUT_DIR=""
DATASET_PRESET="debug"
ANALYZE_BOUNDARIES=false
STABILITY_ANALYSIS=false
DRY_RUN=false
INTERVAL=30

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
            --checkpoint-dir)
                CHECKPOINT_DIR="$2"
                shift 2
                ;;
            --output-dir)
                OUTPUT_DIR="$2"
                shift 2
                ;;
            --dataset-preset)
                DATASET_PRESET="$2"
                shift 2
                ;;
            --watch)
                WATCH_MODE=true
                shift
                ;;
            --interval)
                INTERVAL="$2"
                shift 2
                ;;
            --analyze-boundaries)
                ANALYZE_BOUNDARIES=true
                shift
                ;;
            --stability-analysis)
                STABILITY_ANALYSIS=true
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
Phase 1C: Curriculum Schedule Ablation - Evaluation Runner

Usage:
    ./run_phase1c_evaluation.sh [OPTIONS]

Options:
    --schedule SCHEDULE          Evaluate single schedule (default, early, late, no_curriculum)
    --all-schedules             Evaluate all 4 curriculum schedules
    --checkpoint-dir DIR        Directory containing schedule checkpoints [required]
    --output-dir DIR            Output directory for results
    --dataset-preset PRESET     Dataset preset: debug, validation, production [default: debug]
    --watch                     Real-time monitoring mode
    --interval SECONDS          Refresh interval in watch mode [default: 30]
    --analyze-boundaries        Compute detailed curriculum boundary analysis
    --stability-analysis        Compute training stability analysis
    --dry-run                   Print commands without executing
    --help                      Print this help message

Examples:
    # Evaluate single schedule
    ./run_phase1c_evaluation.sh --schedule default --checkpoint-dir ./checkpoints

    # Evaluate all schedules with detailed analysis
    ./run_phase1c_evaluation.sh --all-schedules --checkpoint-dir ./checkpoints \
      --analyze-boundaries --stability-analysis

    # Real-time monitoring
    ./run_phase1c_evaluation.sh --all-schedules --checkpoint-dir ./checkpoints --watch

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
    if [ -z "$CHECKPOINT_DIR" ]; then
        echo -e "${RED}Error: --checkpoint-dir is required${NC}"
        print_help
        exit 1
    fi

    if [ "$SCHEDULE" == "" ] && [ "$ALL_SCHEDULES" = false ] && [ "$WATCH_MODE" = false ]; then
        echo -e "${RED}Error: Must specify --schedule, --all-schedules, or --watch${NC}"
        print_help
        exit 1
    fi

    # Construct Python command
    PYTHON_CMD="python $SCRIPT_DIR/evaluation/evaluate_curriculum.py"
    PYTHON_CMD="$PYTHON_CMD --checkpoint-dir $CHECKPOINT_DIR"

    # Add arguments
    if [ "$WATCH_MODE" = true ]; then
        PYTHON_CMD="$PYTHON_CMD --watch"
        PYTHON_CMD="$PYTHON_CMD --all-schedules"
    elif [ "$ALL_SCHEDULES" = true ]; then
        PYTHON_CMD="$PYTHON_CMD --all-schedules"
    else
        PYTHON_CMD="$PYTHON_CMD --schedule $SCHEDULE"
    fi

    if [ ! -z "$OUTPUT_DIR" ]; then
        PYTHON_CMD="$PYTHON_CMD --output-dir $OUTPUT_DIR"
    fi

    PYTHON_CMD="$PYTHON_CMD --dataset-preset $DATASET_PRESET"

    if [ "$ANALYZE_BOUNDARIES" = true ]; then
        PYTHON_CMD="$PYTHON_CMD --analyze-boundaries"
    fi

    if [ "$STABILITY_ANALYSIS" = true ]; then
        PYTHON_CMD="$PYTHON_CMD --stability-analysis"
    fi

    if [ "$WATCH_MODE" = true ]; then
        PYTHON_CMD="$PYTHON_CMD --interval $INTERVAL"
    fi

    # Print summary
    if [ "$WATCH_MODE" = true ]; then
        print_header "CURRICULUM SCHEDULE ABLATION - EVALUATION (WATCH MODE)"
    else
        print_header "CURRICULUM SCHEDULE ABLATION - EVALUATION"
    fi

    echo -e "${YELLOW}Configuration:${NC}"
    if [ "$WATCH_MODE" = true ]; then
        echo "  Mode:                Real-time monitoring"
        echo "  Refresh interval:    $INTERVAL seconds"
    else
        echo "  Schedules:           $([ "$ALL_SCHEDULES" = true ] && echo "All (default, early, late, no_curriculum)" || echo "$SCHEDULE")"
    fi
    echo "  Checkpoint dir:      $CHECKPOINT_DIR"
    echo "  Dataset preset:      $DATASET_PRESET"
    echo "  Analyze boundaries:  $([ "$ANALYZE_BOUNDARIES" = true ] && echo "Enabled" || echo "Disabled")"
    echo "  Stability analysis:  $([ "$STABILITY_ANALYSIS" = true ] && echo "Enabled" || echo "Disabled")"
    echo ""

    # Run evaluation
    run_command "$PYTHON_CMD"

    if [ "$WATCH_MODE" = false ]; then
        echo ""
        echo -e "${GREEN}✓ Curriculum Schedule evaluation completed${NC}"
        if [ ! -z "$OUTPUT_DIR" ]; then
            echo -e "${YELLOW}Results saved to:${NC} $OUTPUT_DIR"
        fi
    fi
}

main "$@"
