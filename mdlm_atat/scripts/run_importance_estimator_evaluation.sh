#!/bin/bash
# Importance Estimator Ablation - Evaluation Runner
#
# Comprehensive evaluation of all importance estimator variants with:
# - Validation PPL and BPD metrics
# - Per-token loss by frequency quartile
# - Oracle importance correlation analysis
# - Importance distribution analysis
# - Convergence speed metrics
# - Real-time progress monitoring
#
# Usage:
#   ./run_importance_estimator_evaluation.sh [OPTIONS]
#
# Examples:
#   # Evaluate single variant
#   ./run_importance_estimator_evaluation.sh --variant full --checkpoint-dir ./checkpoints
#
#   # Evaluate all variants with detailed analysis
#   ./run_importance_estimator_evaluation.sh --all-variants --checkpoint-dir ./checkpoints \
#     --analyze-importance --oracle-model bert-base-uncased
#
#   # Real-time monitoring
#   ./run_importance_estimator_evaluation.sh --all-variants --checkpoint-dir ./checkpoints --watch

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
WATCH_MODE=false
CHECKPOINT_DIR=""
OUTPUT_DIR=""
DATASET_PRESET="debug"
ANALYZE_IMPORTANCE=false
ORACLE_MODEL=""
DRY_RUN=false
INTERVAL=30

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
            --analyze-importance)
                ANALYZE_IMPORTANCE=true
                shift
                ;;
            --oracle-model)
                ORACLE_MODEL="$2"
                shift 2
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
Phase 1A: Importance Estimator Ablation - Evaluation Runner

Usage:
    ./run_phase1a_evaluation.sh [OPTIONS]

Options:
    --variant VARIANT             Evaluate single variant (full, frequency_only, learned_only, uniform)
    --all-variants               Evaluate all 4 importance variants
    --checkpoint-dir DIR         Directory containing strategy checkpoints [required]
    --output-dir DIR             Output directory for results
    --dataset-preset PRESET      Dataset preset: debug, validation, production [default: debug]
    --watch                      Real-time monitoring mode
    --interval SECONDS           Refresh interval in watch mode [default: 30]
    --analyze-importance         Compute detailed importance analysis
    --oracle-model MODEL         Oracle model for importance correlation
    --dry-run                    Print commands without executing
    --help                       Print this help message

Examples:
    # Evaluate single variant
    ./run_phase1a_evaluation.sh --variant full --checkpoint-dir ./checkpoints

    # Evaluate all variants with detailed analysis
    ./run_phase1a_evaluation.sh --all-variants --checkpoint-dir ./checkpoints \
      --analyze-importance --oracle-model bert-base-uncased

    # Real-time monitoring
    ./run_phase1a_evaluation.sh --all-variants --checkpoint-dir ./checkpoints --watch

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

    if [ "$VARIANT" == "" ] && [ "$ALL_VARIANTS" = false ] && [ "$WATCH_MODE" = false ]; then
        echo -e "${RED}Error: Must specify --variant, --all-variants, or --watch${NC}"
        print_help
        exit 1
    fi

    # Construct Python command
    PYTHON_CMD="python $SCRIPT_DIR/evaluation/evaluate_importance_estimator.py"
    PYTHON_CMD="$PYTHON_CMD --checkpoint-dir $CHECKPOINT_DIR"

    # Add arguments
    if [ "$WATCH_MODE" = true ]; then
        PYTHON_CMD="$PYTHON_CMD --watch"
        PYTHON_CMD="$PYTHON_CMD --all-variants"
    elif [ "$ALL_VARIANTS" = true ]; then
        PYTHON_CMD="$PYTHON_CMD --all-variants"
    else
        PYTHON_CMD="$PYTHON_CMD --strategy $VARIANT"
    fi

    if [ ! -z "$OUTPUT_DIR" ]; then
        PYTHON_CMD="$PYTHON_CMD --output-dir $OUTPUT_DIR"
    fi

    PYTHON_CMD="$PYTHON_CMD --dataset-preset $DATASET_PRESET"

    if [ "$ANALYZE_IMPORTANCE" = true ]; then
        PYTHON_CMD="$PYTHON_CMD --analyze-importance"
    fi

    if [ ! -z "$ORACLE_MODEL" ]; then
        PYTHON_CMD="$PYTHON_CMD --oracle-model $ORACLE_MODEL"
    fi

    if [ "$WATCH_MODE" = true ]; then
        PYTHON_CMD="$PYTHON_CMD --interval $INTERVAL"
    fi

    # Print summary
    if [ "$WATCH_MODE" = true ]; then
        print_header "IMPORTANCE ESTIMATOR ABLATION - EVALUATION (WATCH MODE)"
    else
        print_header "IMPORTANCE ESTIMATOR ABLATION - EVALUATION"
    fi

    echo -e "${YELLOW}Configuration:${NC}"
    if [ "$WATCH_MODE" = true ]; then
        echo "  Mode:              Real-time monitoring"
        echo "  Refresh interval:  $INTERVAL seconds"
    else
        echo "  Variants:          $([ "$ALL_VARIANTS" = true ] && echo "All (full, frequency_only, learned_only, uniform)" || echo "$VARIANT")"
    fi
    echo "  Checkpoint dir:    $CHECKPOINT_DIR"
    echo "  Dataset preset:    $DATASET_PRESET"
    echo "  Analyze importance:$([ "$ANALYZE_IMPORTANCE" = true ] && echo "Enabled" || echo "Disabled")"
    echo ""

    # Run evaluation
    run_command "$PYTHON_CMD"

    if [ "$WATCH_MODE" = false ]; then
        echo ""
        echo -e "${GREEN}✓ Importance Estimator evaluation completed${NC}"
        if [ ! -z "$OUTPUT_DIR" ]; then
            echo -e "${YELLOW}Results saved to:${NC} $OUTPUT_DIR"
        fi
    fi
}

main "$@"
