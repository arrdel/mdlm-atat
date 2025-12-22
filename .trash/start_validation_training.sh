#!/bin/bash
# Quick Validation Training Startup Script
# Train BERT backbone + Importance Estimator on WikiText-103
# Duration: 12-24 hours
# CRITICAL: Uses discrete MASKING diffusion (not Gaussian)

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="/home/adelechinda/home/projects/mdlm"
DATASET_CACHE="/media/scratch/adele/mdlm_data_cache"
OUTPUTS_DIR="/media/scratch/adele/mdlm_fresh/outputs"
LOGS_DIR="/media/scratch/adele/mdlm_fresh/logs"
CHECKPOINT_DIR="/media/scratch/adele/mdlm_fresh/checkpoints"

# Defaults
CONFIG_NAME="atat/wikitext103_validation"
MAX_STEPS=50000
SANITY_STEPS=100
RUN_SANITY=true
WANDB_PROJECT="mdlm-atat"
NO_CONFIRM=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-sanity)
            RUN_SANITY=false
            shift
            ;;
        --sanity-only)
            MAX_STEPS=$SANITY_STEPS
            RUN_SANITY=false
            shift
            ;;
        --max-steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        --no-confirm)
            NO_CONFIRM=true
            shift
            ;;
        --no-wandb)
            WANDB_OFFLINE="--wandb-offline"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}MDLM-ATAT Quick Validation${NC}"
echo -e "${BLUE}Training Setup Script${NC}"
echo -e "${BLUE}================================${NC}"
echo ""

# Step 1: Verify environment
echo -e "${YELLOW}[1/6]${NC} Verifying environment..."
echo "  Project: $PROJECT_ROOT"
echo "  Dataset cache: $DATASET_CACHE"
echo "  Logs: $LOGS_DIR"

if [ ! -d "$PROJECT_ROOT" ]; then
    echo -e "${RED}❌ Project directory not found: $PROJECT_ROOT${NC}"
    exit 1
fi

if [ ! -f "$PROJECT_ROOT/mdlm_atat/scripts/train_atat.py" ]; then
    echo -e "${RED}❌ Training script not found${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Project structure OK${NC}"
echo ""

# Step 2: Check Python environment
echo -e "${YELLOW}[2/6]${NC} Checking Python environment..."

cd "$PROJECT_ROOT"

if ! command -v python &> /dev/null; then
    echo -e "${RED}❌ Python not found${NC}"
    exit 1
fi

PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo "  Python: $PYTHON_VERSION"

# Check key packages
python -c "import torch; print(f'  PyTorch: {torch.__version__}')" 2>/dev/null || {
    echo -e "${RED}❌ PyTorch not installed${NC}"
    exit 1
}

python -c "import transformers; print(f'  Transformers: {transformers.__version__}')" 2>/dev/null || {
    echo -e "${RED}❌ Transformers not installed${NC}"
    exit 1
}

python -c "import datasets; print(f'  Datasets: {datasets.__version__}')" 2>/dev/null || {
    echo -e "${RED}❌ Datasets not installed${NC}"
    exit 1
}

echo -e "${GREEN}✓ Python packages OK${NC}"
echo ""

# Step 3: Check GPUs
echo -e "${YELLOW}[3/6]${NC} Checking GPU availability..."

GPU_COUNT=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
if [ "$GPU_COUNT" -eq 0 ]; then
    echo -e "${RED}❌ No CUDA-capable GPUs found${NC}"
    exit 1
fi

echo "  GPUs available: $GPU_COUNT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | while read line; do
    echo "    - $line"
done

GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
if [ "$GPU_MEMORY" -lt 30000 ]; then
    echo -e "${YELLOW}⚠ Low GPU memory (~${GPU_MEMORY}MB). May need to reduce batch size.${NC}"
fi

echo -e "${GREEN}✓ GPU setup OK${NC}"
echo ""

# Step 4: Prepare data directories
echo -e "${YELLOW}[4/6]${NC} Preparing data directories..."

mkdir -p "$DATASET_CACHE"
mkdir -p "$OUTPUTS_DIR"
mkdir -p "$LOGS_DIR"
mkdir -p "$CHECKPOINT_DIR"

echo "  Created: $DATASET_CACHE"
echo "  Created: $OUTPUTS_DIR"
echo "  Created: $LOGS_DIR"
echo "  Created: $CHECKPOINT_DIR"

echo -e "${GREEN}✓ Directories ready${NC}"
echo ""

# Step 5: Download WikiText-103
echo -e "${YELLOW}[5/6]${NC} Checking WikiText-103 dataset..."

python -c "
import os
from datasets import load_dataset

cache_dir = '$DATASET_CACHE'
os.makedirs(cache_dir, exist_ok=True)

try:
    print('  Loading WikiText-103...')
    dataset = load_dataset('wikitext', 'wikitext-103-v1', cache_dir=cache_dir)
    print(f'  Train: {len(dataset[\"train\"])} samples')
    print(f'  Validation: {len(dataset[\"validation\"])} samples')
except Exception as e:
    print(f'ERROR: {e}')
    exit(1)
" || exit 1

echo -e "${GREEN}✓ WikiText-103 ready${NC}"
echo ""

# Step 6: Show training info
echo -e "${YELLOW}[6/6]${NC} Training Configuration..."
echo ""
echo -e "${BLUE}Training Setup:${NC}"
echo "  Config: $CONFIG_NAME"
echo "  Max steps: $MAX_STEPS"
echo "  Dataset: WikiText-103 (validation)"
echo "  Duration: 12-24 hours"
echo ""
echo -e "${BLUE}Key Diffusion Parameters:${NC}"
echo "  Mode: discrete masking diffusion"
echo "  Parameterization: substitution (token masking)"
echo "  Noise schedule: loglinear"
echo "  Sampling steps: 128"
echo ""
echo -e "${BLUE}ATAT Components:${NC}"
echo "  ✓ BERT backbone (768D, 12L)"
echo "  ✓ Importance Estimator (256D, 2L)"
echo "  ✓ Adaptive masking"
echo "  ✓ Curriculum learning (3 stages)"
echo "  ✓ Uncertainty sampler (inference)"
echo ""

if [ "$RUN_SANITY" = true ]; then
    echo -e "${YELLOW}Will run sanity check first (100 steps, ~5-10 min)${NC}"
    echo ""
fi

# Confirmation prompt
if [ "$NO_CONFIRM" = false ]; then
    echo -e "${YELLOW}Ready to start training?${NC} (y/n)"
    read -r response
    if [ "$response" != "y" ]; then
        echo "Training cancelled."
        exit 0
    fi
fi

echo ""
echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}Starting Training${NC}"
echo -e "${GREEN}================================${NC}"
echo ""

# Run sanity check if enabled
if [ "$RUN_SANITY" = true ]; then
    echo -e "${YELLOW}Running sanity check (100 steps)...${NC}"
    echo ""
    
    SANITY_LOG="$LOGS_DIR/sanity_check_$(date +%Y%m%d_%H%M%S).log"
    
    python mdlm_atat/scripts/train_atat.py \
        --config-name "$CONFIG_NAME" \
        --max-steps 100 \
        --log-dir "$LOGS_DIR/sanity_check_$(date +%Y%m%d_%H%M%S)" \
        $WANDB_OFFLINE \
        2>&1 | tee "$SANITY_LOG"
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ Sanity check failed!${NC}"
        echo "Check logs: $SANITY_LOG"
        exit 1
    fi
    
    echo -e "${GREEN}✓ Sanity check passed!${NC}"
    echo ""
fi

# Run main training
echo -e "${YELLOW}Starting validation training ($MAX_STEPS steps)...${NC}"
echo -e "${YELLOW}Estimated duration: 12-24 hours${NC}"
echo ""

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$LOGS_DIR/wikitext_validation_$TIMESTAMP"

python mdlm_atat/scripts/train_atat.py \
    --config-name "$CONFIG_NAME" \
    --max-steps "$MAX_STEPS" \
    --log-dir "$LOG_DIR" \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-run-name "wikitext_validation_$TIMESTAMP" \
    $WANDB_OFFLINE \
    2>&1 | tee "$LOG_DIR/training.log"

TRAIN_EXIT_CODE=$?

echo ""
echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}Training Complete${NC}"
echo -e "${GREEN}================================${NC}"
echo ""

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓ Training completed successfully!${NC}"
    echo ""
    echo "Results saved to: $LOG_DIR"
    echo ""
    echo "Next steps:"
    echo "  1. Check final metrics: tail $LOG_DIR/training.log"
    echo "  2. View checkpoint: ls -lh $LOG_DIR/checkpoints/"
    echo "  3. Ready for OpenWebText training? (1-2 weeks)"
    echo ""
else
    echo -e "${RED}❌ Training failed with exit code $TRAIN_EXIT_CODE${NC}"
    echo ""
    echo "Check logs for errors:"
    echo "  tail -100 $LOG_DIR/training.log"
    exit 1
fi

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}For detailed info, see:${NC}"
echo -e "${BLUE}  docs/QUICK_VALIDATION_TRAINING.md${NC}"
echo -e "${BLUE}================================${NC}"
