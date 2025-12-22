#!/bin/bash

# WandB Setup and Training Script
# Automatically configures WandB and starts training

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${BLUE}═══════════════════════════════════════════════════${NC}"
echo -e "${BLUE}   MDLM+ATAT Production Training with WandB${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════${NC}\n"

# Check if WandB is installed
if ! python -c "import wandb" 2>/dev/null; then
    echo -e "${YELLOW}WandB not installed. Installing...${NC}"
    pip install wandb
fi

# Check if logged in to WandB
echo -e "${CYAN}Checking WandB authentication...${NC}"
if wandb login --relogin 2>/dev/null; then
    echo -e "${GREEN}✓ WandB authenticated${NC}\n"
else
    echo -e "${YELLOW}⚠ WandB login failed, running in offline mode${NC}"
    OFFLINE_MODE="--wandb-offline"
fi

# Get configuration options
read -p "$(echo -e ${CYAN}Enter WandB project name [mdlm-atat]:${NC} )" WANDB_PROJECT
WANDB_PROJECT=${WANDB_PROJECT:-"mdlm-atat"}

# Generate run name with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
read -p "$(echo -e ${CYAN}Enter WandB run name [mdlm_production_$TIMESTAMP]:${NC} )" WANDB_RUN_NAME
WANDB_RUN_NAME=${WANDB_RUN_NAME:-"mdlm_production_$TIMESTAMP"}

# Check dataset
CACHE_DIR="/media/scratch/adele/mdlm_fresh/data_cache"
if [ ! -d "$CACHE_DIR/openwebtext" ]; then
    echo -e "\n${YELLOW}OpenWebText dataset not found.${NC}"
    read -p "$(echo -e ${CYAN}Download now? [Y/n]:${NC} )" DOWNLOAD
    if [ "$DOWNLOAD" != "n" ]; then
        echo -e "${BLUE}Downloading dataset...${NC}"
        cd /home/adelechinda/home/projects/mdlm
        python mdlm_atat/scripts/download_datasets.py --datasets openwebtext
    fi
else
    echo -e "${GREEN}✓ Dataset found${NC}\n"
fi

# Confirm training parameters
echo -e "\n${BLUE}Training Parameters:${NC}"
echo -e "  Project: ${GREEN}$WANDB_PROJECT${NC}"
echo -e "  Run Name: ${GREEN}$WANDB_RUN_NAME${NC}"
echo -e "  Max Steps: ${GREEN}500000${NC}"
echo -e "  Config: ${GREEN}atat/production_training${NC}"
echo -e "  Dataset: ${GREEN}OpenWebText${NC}"

read -p "$(echo -e ${CYAN}Start training? [Y/n]:${NC} )" CONFIRM
if [ "$CONFIRM" = "n" ]; then
    echo -e "${RED}Training cancelled${NC}"
    exit 0
fi

# Change to project directory
cd /home/adelechinda/home/projects/mdlm

echo -e "\n${BLUE}═══════════════════════════════════════════════════${NC}"
echo -e "${BLUE}   Starting Training...${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════${NC}\n"

# Start training
python mdlm_atat/scripts/train_atat.py \
  --config-name atat/production_training \
  --max-steps 500000 \
  --wandb-project "$WANDB_PROJECT" \
  --wandb-run-name "$WANDB_RUN_NAME" \
  ${OFFLINE_MODE} \
  --no-confirm

EXIT_CODE=$?

# Print final message
echo -e "\n${BLUE}═══════════════════════════════════════════════════${NC}"
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}   Training Completed Successfully!${NC}"
else
    echo -e "${RED}   Training Failed${NC}"
fi
echo -e "${BLUE}═══════════════════════════════════════════════════${NC}\n"

echo -e "${CYAN}View results at:${NC} https://wandb.ai\n"

exit $EXIT_CODE
