#!/bin/bash

# MDLM+ATAT Production Training with WandB in tmux
# This script sets up a tmux session for production training with real-time monitoring

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

PROJECT_DIR="/home/adelechinda/home/projects/mdlm"
SESSION_NAME="mdlm-training"
CACHE_DIR="/media/scratch/adele/mdlm_fresh/data_cache"
LOG_DIR="/media/scratch/adele/mdlm_fresh/logs"
CHECKPOINT_DIR="/media/scratch/adele/mdlm_fresh/checkpoints"

echo -e "${BLUE}╔═══════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   MDLM+ATAT Production Training Setup${NC}"
echo -e "${BLUE}║   with WandB & tmux${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════╝${NC}\n"

# Check if session already exists
if tmux list-sessions | grep -q "^$SESSION_NAME"; then
    echo -e "${YELLOW}⚠ Session '$SESSION_NAME' already exists${NC}"
    read -p "$(echo -e ${CYAN}Attach to existing session? [Y/n]:${NC} )" ATTACH
    if [ "$ATTACH" != "n" ]; then
        tmux attach-session -t "$SESSION_NAME"
        exit 0
    fi
    echo -e "${YELLOW}Killing existing session...${NC}"
    tmux kill-session -t "$SESSION_NAME"
fi

# Verify dataset
if [ ! -d "$CACHE_DIR/openwebtext" ]; then
    echo -e "${RED}✗ OpenWebText dataset not found!${NC}"
    echo -e "${YELLOW}Run this first:${NC}"
    echo -e "  python mdlm_atat/scripts/download_datasets.py --datasets openwebtext"
    exit 1
fi
echo -e "${GREEN}✓ Dataset verified${NC}"

# Create directories
mkdir -p "$LOG_DIR" "$CHECKPOINT_DIR"
echo -e "${GREEN}✓ Directories ready${NC}\n"

# Create tmux session
echo -e "${CYAN}Creating tmux session: $SESSION_NAME${NC}"
tmux new-session -d -s "$SESSION_NAME" -x 240 -y 60

# Window 1: Training
echo -e "${CYAN}Setting up Window 1: Training${NC}"
tmux send-keys -t "$SESSION_NAME" "cd $PROJECT_DIR && clear" C-m
tmux send-keys -t "$SESSION_NAME" "echo 'MDLM+ATAT Production Training with WandB' && echo '========================================='" C-m
tmux send-keys -t "$SESSION_NAME" "python mdlm_atat/scripts/train_atat.py --config-name atat/production_training --max-steps 500000 --wandb-project mdlm-atat --wandb-run-name production_$(date +%Y%m%d_%H%M%S) --no-confirm" C-m

# Window 2: GPU Monitoring
echo -e "${CYAN}Setting up Window 2: GPU Monitoring${NC}"
tmux new-window -t "$SESSION_NAME" -n "gpu"
tmux send-keys -t "$SESSION_NAME:gpu" "watch -n 1 nvidia-smi" C-m

# Window 3: Training Logs
echo -e "${CYAN}Setting up Window 3: Training Logs${NC}"
tmux new-window -t "$SESSION_NAME" -n "logs"
tmux send-keys -t "$SESSION_NAME:logs" "cd $LOG_DIR && clear && echo 'Waiting for training to start...' && sleep 3 && tail -f training_*.log" C-m

# Window 4: Checkpoint Monitor
echo -e "${CYAN}Setting up Window 4: Checkpoint Monitor${NC}"
tmux new-window -t "$SESSION_NAME" -n "checkpoints"
tmux send-keys -t "$SESSION_NAME:checkpoints" "cd $CHECKPOINT_DIR && watch -n 5 'ls -lht | head -15'" C-m

# Window 5: System Monitor
echo -e "${CYAN}Setting up Window 5: System Monitor${NC}"
tmux new-window -t "$SESSION_NAME" -n "system"
tmux send-keys -t "$SESSION_NAME:system" "clear && echo 'System Resource Monitor' && watch -n 2 'free -h; echo; df -h /media/scratch/adele; echo; ps aux | grep python | grep -v grep | wc -l'" C-m

# Window 6: WandB & Commands
echo -e "${CYAN}Setting up Window 6: Commands Reference${NC}"
tmux new-window -t "$SESSION_NAME" -n "commands"
tmux send-keys -t "$SESSION_NAME:commands" "cat << 'CMDS'\n\n${CYAN}═══════════════════════════════════════════════════${NC}\n${BLUE}MDLM+ATAT Training - Command Reference${NC}\n${CYAN}═══════════════════════════════════════════════════${NC}\n\n${GREEN}Navigation:${NC}\n  tmux select-window -t $SESSION_NAME:0   # Training\n  tmux select-window -t $SESSION_NAME:1   # GPU Monitor\n  tmux select-window -t $SESSION_NAME:2   # Logs\n  tmux select-window -t $SESSION_NAME:3   # Checkpoints\n  tmux select-window -t $SESSION_NAME:4   # System\n  tmux select-window -t $SESSION_NAME:5   # Commands\n\n${GREEN}tmux Shortcuts (from inside tmux):${NC}\n  Ctrl-b c              # Create new window\n  Ctrl-b n              # Next window\n  Ctrl-b p              # Previous window\n  Ctrl-b [0-5]          # Select window by number\n  Ctrl-b d              # Detach session\n  Ctrl-b ?              # Show keybindings\n\n${GREEN}View Monitoring:${NC}\n  WandB Dashboard:      https://wandb.ai\n  GPU Usage:            Window 1 (gpu)\n  Training Logs:        Window 2 (logs)\n  Checkpoints:          Window 3 (checkpoints)\n  System Resources:     Window 4 (system)\n\n${GREEN}Training Status:${NC}\n  Watch logs in real-time (Window 2)\n  Check GPU usage (Window 1)\n  Monitor checkpoints (Window 3)\n\n${YELLOW}Stop Training:${NC}\n  Go to Window 0 (Training)\n  Press Ctrl-C to interrupt\n\n${CYAN}═══════════════════════════════════════════════════${NC}\n\nCMDS" C-m

# Select first window (training)
tmux select-window -t "$SESSION_NAME:0"

echo -e "\n${BLUE}╔═══════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   Session Created: $SESSION_NAME${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════╝${NC}\n"

echo -e "${GREEN}✓ tmux session created with 6 windows:${NC}\n"
echo -e "  ${CYAN}Window 0: training${NC}        → Main training process"
echo -e "  ${CYAN}Window 1: gpu${NC}             → GPU monitoring (nvidia-smi)"
echo -e "  ${CYAN}Window 2: logs${NC}            → Real-time training logs"
echo -e "  ${CYAN}Window 3: checkpoints${NC}     → Checkpoint directory monitor"
echo -e "  ${CYAN}Window 4: system${NC}          → System resource monitor"
echo -e "  ${CYAN}Window 5: commands${NC}        → Command reference\n"

echo -e "${YELLOW}Attaching to session in 3 seconds...${NC}\n"
sleep 3

# Attach to session
tmux attach-session -t "$SESSION_NAME"
