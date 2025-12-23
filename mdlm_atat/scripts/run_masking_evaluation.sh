#!/bin/bash
# Masking Strategy Ablation - Evaluation Runner
#
# Comprehensive evaluation of all masking strategy variants with:
# - Validation PPL and BPD metrics
# - Per-stage loss analysis (easy, medium, hard)
# - Curriculum boundary transition effects
# - Gradient norm statistics by importance quantile
# - Convergence speed metrics
# - Real-time progress monitoring
#
# Usage:
#   ./run_masking_evaluation.sh [OPTIONS]
#
# Examples:
#   # Evaluate single strategy
#   ./run_masking_evaluation.sh --strategy balanced --checkpoint-dir ./checkpoints
#
#   # Evaluate all strategies with detailed analysis
#   ./run_masking_evaluation.sh --all-strategies --checkpoint-dir ./checkpoints \
#     --analyze-curriculum --gradient-analysis
#
#   # Real-time monitoring
#   ./run_masking_evaluation.sh --all-strategies --checkpoint-dir ./checkpoints --watch

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
WATCH_MODE=false
CHECKPOINT_DIR=""
OUTPUT_DIR=""
DATASET_PRESET="debug"
ANALYZE_CURRICULUM=false
IMPORTANCE_QUANTILES=false
GRADIENT_ANALYSIS=false
DRY_RUN=false
INTERVAL=30

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
            --analyze-curriculum)
                ANALYZE_CURRICULUM=true
                shift
                ;;
            --importance-quantiles)
                IMPORTANCE_QUANTILES=true
                shift
                ;;
            --gradient-analysis)
                GRADIENT_ANALYSIS=true
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
Phase 1B: Masking Strategy Ablation - Evaluation Runner

Usage:
    ./run_phase1b_evaluation.sh [OPTIONS]

Options:
    --strategy STRATEGY          Evaluate single strategy (balanced, proportional, inverse, time_only)
    --all-strategies            Evaluate all 4 masking strategies
    --checkpoint-dir DIR        Directory containing strategy checkpoints [required]
    --output-dir DIR            Output directory for results
    --dataset-preset PRESET     Dataset preset: debug, validation, production [default: debug]
    --watch                     Real-time monitoring mode
    --interval SECONDS          Refresh interval in watch mode [default: 30]
    --analyze-curriculum        Compute detailed curriculum analysis
    --importance-quantiles      Compute per-importance-quantile analysis
    --gradient-analysis         Compute gradient norm analysis
    --dry-run                   Print commands without executing
    --help                      Print this help message

Examples:
    # Evaluate single strategy
    ./run_phase1b_evaluation.sh --strategy balanced --checkpoint-dir ./checkpoints

    # Evaluate all strategies with detailed analysis
    ./run_phase1b_evaluation.sh --all-strategies --checkpoint-dir ./checkpoints \
      --analyze-curriculum --gradient-analysis

    # Real-time monitoring
    ./run_phase1b_evaluation.sh --all-strategies --checkpoint-dir ./checkpoints --watch

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

    if [ "$STRATEGY" == "" ] && [ "$ALL_STRATEGIES" = false ] && [ "$WATCH_MODE" = false ]; then
        echo -e "${RED}Error: Must specify --strategy, --all-strategies, or --watch${NC}"
        print_help
        exit 1
    fi

    # Construct Python command
    PYTHON_CMD="python $SCRIPT_DIR/evaluation/evaluate_masking.py"
    PYTHON_CMD="$PYTHON_CMD --checkpoint-dir $CHECKPOINT_DIR"

    # Add arguments
    if [ "$WATCH_MODE" = true ]; then
        PYTHON_CMD="$PYTHON_CMD --watch"
        PYTHON_CMD="$PYTHON_CMD --all-strategies"
    elif [ "$ALL_STRATEGIES" = true ]; then
        PYTHON_CMD="$PYTHON_CMD --all-strategies"
    else
        PYTHON_CMD="$PYTHON_CMD --strategy $STRATEGY"
    fi

    if [ ! -z "$OUTPUT_DIR" ]; then
        PYTHON_CMD="$PYTHON_CMD --output-dir $OUTPUT_DIR"
    fi

    PYTHON_CMD="$PYTHON_CMD --dataset-preset $DATASET_PRESET"

    if [ "$ANALYZE_CURRICULUM" = true ]; then
        PYTHON_CMD="$PYTHON_CMD --analyze-curriculum"
    fi

    if [ "$IMPORTANCE_QUANTILES" = true ]; then
        PYTHON_CMD="$PYTHON_CMD --importance-quantiles"
    fi

    if [ "$GRADIENT_ANALYSIS" = true ]; then
        PYTHON_CMD="$PYTHON_CMD --gradient-analysis"
    fi

    if [ "$WATCH_MODE" = true ]; then
        PYTHON_CMD="$PYTHON_CMD --interval $INTERVAL"
    fi

    # Print summary
    if [ "$WATCH_MODE" = true ]; then
        print_header "MASKING STRATEGY ABLATION - EVALUATION (WATCH MODE)"
    else
        print_header "MASKING STRATEGY ABLATION - EVALUATION"
    fi

    echo -e "${YELLOW}Configuration:${NC}"
    if [ "$WATCH_MODE" = true ]; then
        echo "  Mode:                Real-time monitoring"
        echo "  Refresh interval:    $INTERVAL seconds"
    else
        echo "  Strategies:          $([ "$ALL_STRATEGIES" = true ] && echo "All (balanced, proportional, inverse, time_only)" || echo "$STRATEGY")"
    fi
    echo "  Checkpoint dir:      $CHECKPOINT_DIR"
    echo "  Dataset preset:      $DATASET_PRESET"
    echo "  Analyze curriculum:  $([ "$ANALYZE_CURRICULUM" = true ] && echo "Enabled" || echo "Disabled")"
    echo "  Quantile analysis:   $([ "$IMPORTANCE_QUANTILES" = true ] && echo "Enabled" || echo "Disabled")"
    echo "  Gradient analysis:   $([ "$GRADIENT_ANALYSIS" = true ] && echo "Enabled" || echo "Disabled")"
    echo ""

    # Run evaluation
    run_command "$PYTHON_CMD"

    if [ "$WATCH_MODE" = false ]; then
        echo ""
        echo -e "${GREEN}✓ Masking Strategy evaluation completed${NC}"
        if [ ! -z "$OUTPUT_DIR" ]; then
            echo -e "${YELLOW}Results saved to:${NC} $OUTPUT_DIR"
        fi
    fi
}

main "$@"
