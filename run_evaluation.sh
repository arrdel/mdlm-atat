#!/bin/bash

# ATAT Checkpoint Evaluation Script
# Runs comprehensive evaluation on the best saved checkpoint
# 
# Metrics evaluated:
#   1. Perplexity (PPL): Language modeling quality
#   2. Negative Log-Likelihood (NLL): Standardized quality metric  
#   3. Top-1 Accuracy: Prediction correctness
#
# Usage:
#   bash run_evaluation.sh [dataset] [batch_size] [num_batches]
#
# Examples:
#   bash run_evaluation.sh                    # Default: WikiText-103, batch_size=32
#   bash run_evaluation.sh wikitext103 32     # Explicit WikiText-103
#   bash run_evaluation.sh openwebtext 64 100 # OpenWebText with 100 batches

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse arguments
DATASET="${1:-wikitext103}"
BATCH_SIZE="${2:-32}"
NUM_BATCHES="${3:-}"

# Paths
CHECKPOINT_DIR="/media/scratch/adele/mdlm_fresh/outputs/checkpoints"
BEST_CHECKPOINT="$CHECKPOINT_DIR/last.ckpt"
EVAL_SCRIPT="/home/adelechinda/home/projects/mdlm/mdlm_atat/scripts/eval_production.py"
RESULTS_DIR="/media/scratch/adele/mdlm_fresh/outputs/evaluation_results"

# Create results directory
mkdir -p "$RESULTS_DIR"

# Print header
echo ""
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║         ATAT CHECKPOINT EVALUATION - CORE METRICS TEST            ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

# Check checkpoint exists
if [ ! -f "$BEST_CHECKPOINT" ]; then
    echo -e "${RED}✗ Error: Checkpoint not found: $BEST_CHECKPOINT${NC}"
    echo ""
    echo "Available checkpoints in $CHECKPOINT_DIR:"
    ls -lh "$CHECKPOINT_DIR"/*.ckpt 2>/dev/null || echo "No checkpoints found"
    exit 1
fi

echo -e "${GREEN}✓ Checkpoint found:${NC} $(basename $BEST_CHECKPOINT)"
echo -e "  Size: $(du -h $BEST_CHECKPOINT | cut -f1)"
echo ""

# Print configuration
echo -e "${BLUE}Evaluation Configuration:${NC}"
echo "  Checkpoint:  $BEST_CHECKPOINT"
echo "  Dataset:     $DATASET"
echo "  Batch Size:  $BATCH_SIZE"
if [ -z "$NUM_BATCHES" ]; then
    echo "  Max Batches: All (evaluate full dataset)"
else
    echo "  Max Batches: $NUM_BATCHES"
fi
echo "  Results Dir: $RESULTS_DIR"
echo ""

# Activate conda environment
echo -e "${YELLOW}Activating conda environment...${NC}"
source /home/adelechinda/miniconda3/bin/activate mdlm-atat

# Print metrics being evaluated
echo ""
echo -e "${BLUE}Metrics to be evaluated:${NC}"
echo "  1. 📊 Perplexity (PPL) - Language modeling quality"
echo "     → Lower is better (target: < 50 for good models)"
echo ""
echo "  2. 📈 Negative Log-Likelihood (NLL) - Standardized quality"
echo "     → Lower is better (allows cross-dataset comparison)"
echo ""
echo "  3. ✨ Top-1 Accuracy - Prediction correctness"
echo "     → Higher is better (% of correctly predicted tokens)"
echo ""

# Build evaluation command
CMD="python $EVAL_SCRIPT \
    --checkpoint $BEST_CHECKPOINT \
    --dataset $DATASET \
    --batch-size $BATCH_SIZE"

if [ -n "$NUM_BATCHES" ]; then
    CMD="$CMD --num-batches $NUM_BATCHES"
fi

echo -e "${YELLOW}Starting evaluation...${NC}"
echo "Command: $CMD"
echo ""

# Run evaluation
$CMD

# Print completion
echo ""
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║                   EVALUATION COMPLETE                             ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""
echo -e "${GREEN}✓ Results saved to:${NC} $RESULTS_DIR"
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo "  • Check results JSON: ls -lh $RESULTS_DIR/eval_*.json | tail -1"
echo "  • View latest results: cat $RESULTS_DIR/eval_*.json | tail -50"
echo "  • Compare with baseline: python compare_results.py"
echo ""
