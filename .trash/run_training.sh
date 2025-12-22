#!/bin/bash
# Simple training launcher for ATAT

cd /home/adelechinda/home/projects/mdlm/mdlm_atat

python ../mdlm/main.py \
  --config-path ../mdlm_atat/configs \
  --config-name atat/tiny \
  trainer.max_steps=10000 \
  trainer.devices=2 \
  data.cache_dir=/media/scratch/adele/mdlm_fresh/data_cache \
  wandb.offline=true
