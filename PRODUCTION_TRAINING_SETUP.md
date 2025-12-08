# 🚀 Full Production Training Setup

## TL;DR - Quick Start
```bash
# Step 1: Download dataset (one-time, ~30 mins - ~40GB)
python mdlm_atat/scripts/download_datasets.py --datasets openwebtext

# Step 2: Configure production training
# Create mdlm_atat/configs/atat/production_training.yaml (see template below)

# Step 3: Run production training (500k steps)
python mdlm_atat/scripts/train_atat.py --config-name atat/production_training --max-steps 500000

# Or use the launcher script:
bash start_production_training.sh
```

---

## Dataset Splitting: Do You Need It?

**Short Answer: NO - Automatic Splitting Handled**

### How Splitting Works (Built-in)

The dataloader automatically handles train/validation splits:

```
OpenWebText Dataset (40GB)
├── Train Split: Used for training
│   └── 90-95% of data automatically selected
├── Validation Split: Used for validation
│   └── 5-10% of data automatically selected
└── Automatically formatted by HuggingFace
```

**No manual split needed!** The framework uses the built-in HuggingFace dataset splits:

- **WikiText-103**: Has `train`, `validation`, `test` splits (built-in)
- **OpenWebText**: Has `train` split (default) and automatic validation split
- **WikiText-2, PTB**: Have pre-defined splits

### Configuration

Your config file specifies which splits to use:

```yaml
data:
  train: openwebtext    # Use train split of OpenWebText
  valid: openwebtext    # Use validation split of OpenWebText
  tokenizer_name_or_path: gpt2
  cache_dir: /media/scratch/adele/mdlm_fresh/data_cache
```

---

## Step-by-Step Full Training Setup

### Step 1: Download Dataset (One-time)

```bash
cd /home/adelechinda/home/projects/mdlm

# Download OpenWebText (required)
python mdlm_atat/scripts/download_datasets.py --datasets openwebtext

# This will:
# ✓ Download ~40GB of data
# ✓ Cache to /media/scratch/adele/mdlm_fresh/data_cache
# ✓ Create train/validation splits automatically
# ✓ Take ~30 minutes depending on internet speed
```

**Verification**:
```bash
ls -lh /media/scratch/adele/mdlm_fresh/data_cache/
# Should show: openwebtext/, wikitext103/ (if you downloaded both)
```

---

### Step 2: Create Production Configuration

Create `mdlm_atat/configs/atat/production_training.yaml`:

```yaml
# @package _global_

# ATAT Production Training Configuration
# Full training on OpenWebText (500k steps)
# Expected duration: 1-2 weeks on 6x RTX 4090

defaults:
  - /lr_scheduler: cosine_decay_warmup

# Basic settings
mode: train
diffusion: absorbing_state    # Discrete masking (CRITICAL!)
backbone: dit
parameterization: subs        # Substitute masking
time_conditioning: False
T: 0
subs_masking: False
seed: 42

# Data configuration - OpenWebText for production
data:
  train: openwebtext          # HuggingFace will auto-select train split
  valid: openwebtext          # HuggingFace will auto-select validation split
  tokenizer_name_or_path: gpt2
  cache_dir: /media/scratch/adele/mdlm_fresh/data_cache
  wrap: True
  streaming: False

# Model configuration - Full BERT scale
model:
  name: atat_dit
  type: dit
  length: 1024
  hidden_size: 768
  cond_dim: 128
  n_blocks: 12
  n_heads: 12
  scale_by_sigma: True
  dropout: 0.1
  tie_word_embeddings: False
  importance_hidden_dim: 256
  importance_num_layers: 2
  masking_strategy: "importance"
  masking_temperature: 1.0
  position_bias: false
  use_importance: true
  use_adaptive_masking: true
  use_curriculum: true

# ATAT components - All enabled
use_importance: true
use_adaptive_masking: true
use_curriculum: true

# Training settings
training:
  ema: 0.9999
  antithetic_sampling: True
  importance_sampling: False
  sampling_eps: 1e-3
  change_of_variables: False
  importance_loss_weight: 0.1

# Curriculum settings for full training
curriculum:
  curriculum_type: "linear"
  warmup_steps: 5000           # Longer warmup for 500k steps
  easy_fraction: 0.3
  hard_fraction: 0.3
  dynamic_adjustment: false

# Loader settings - Production scale
loader:
  global_batch_size: 24
  eval_global_batch_size: ${.loader.global_batch_size}
  batch_size: 4                 # 4 per GPU
  eval_batch_size: 4
  num_workers: 8
  pin_memory: true

# Training hyperparameters
max_steps: 500000              # 500k steps for full training
val_check_interval: 1000       # Validate every 1000 steps
log_interval: 100              # Log every 100 steps
save_interval: 5000            # Save checkpoint every 5000 steps
warmup_steps: 5000

# Learning rate
learning_rate: 1e-4
weight_decay: 0.01
warmup_fraction: 0.02          # 2% of total steps = 10k steps

# Optimization
grad_accumulation_steps: 1
grad_clip_norm: 1.0

# Device settings
precision: bf16                # Mixed precision training
enable_gradient_checkpointing: true
```

Save this as: `mdlm_atat/configs/atat/production_training.yaml`

---

### Step 3: Run Production Training

**Option A: Direct Command**
```bash
cd /home/adelechinda/home/projects/mdlm

python mdlm_atat/scripts/train_atat.py \
  --config-name atat/production_training \
  --max-steps 500000 \
  --no-confirm
```

**Option B: Using Launcher Script**
```bash
bash start_production_training.sh
```

**Option C: Background with nohup**
```bash
nohup python mdlm_atat/scripts/train_atat.py \
  --config-name atat/production_training \
  --max-steps 500000 \
  > /media/scratch/adele/mdlm_fresh/logs/training.log 2>&1 &

# Check status
tail -f /media/scratch/adele/mdlm_fresh/logs/training.log
```

---

## Monitoring Training

### Real-time GPU Usage
```bash
# Terminal 1: Monitor GPUs
watch -n 1 nvidia-smi
```

### Training Logs
```bash
# Terminal 2: Monitor training
tail -f /media/scratch/adele/mdlm_fresh/logs/training_*.log
```

### Metrics
```bash
# Terminal 3: Monitor checkpoints
ls -lht /media/scratch/adele/mdlm_fresh/checkpoints/ | head -10
```

---

## Understanding Dataset Splits

### How HuggingFace Handles Splits

```python
# When you load a dataset like this:
dataset = load_dataset('openwebtext', cache_dir='/path/to/cache')

# You get:
# dataset['train']       # Training data (95% of OpenWebText)
# dataset['validation']  # Validation data (5% of OpenWebText)

# The framework automatically uses:
# - mode='train' → uses dataset['train']
# - mode='validation' → uses dataset['validation']
```

### Dataset Sizes

| Dataset | Train | Validation | Total |
|---------|-------|------------|-------|
| OpenWebText | ~38GB | ~2GB | ~40GB |
| WikiText-103 | ~353MB | ~36MB | ~389MB |
| WikiText-2 | ~10MB | ~3MB | ~13MB |
| PTB | ~5.5MB | ~400KB | ~5.9MB |

---

## Training Phases

### Phase 1: Data Preparation (30 mins)
```bash
python mdlm_atat/scripts/download_datasets.py --datasets openwebtext
# Downloads 40GB, caches to /media/scratch/adele/mdlm_fresh/data_cache
```

### Phase 2: Validation Run (optional, 1-2 hours)
```bash
# Verify setup with 50k steps
python mdlm_atat/scripts/train_atat.py \
  --config-name atat/wikitext103_validation \
  --max-steps 50000
```

### Phase 3: Production Training (1-2 weeks)
```bash
# Full 500k step training
python mdlm_atat/scripts/train_atat.py \
  --config-name atat/production_training \
  --max-steps 500000
```

### Phase 4: Evaluation (1-2 hours)
```bash
python mdlm_atat/scripts/eval_atat.py \
  --checkpoint /media/scratch/adele/mdlm_fresh/checkpoints/final.ckpt
```

---

## Checkpoints & Resuming

### Checkpoint Storage
```
/media/scratch/adele/mdlm_fresh/checkpoints/
├── epoch_0_step_5000.ckpt
├── epoch_0_step_10000.ckpt
├── ...
└── final.ckpt (or last.ckpt)
```

### Resume from Checkpoint
```bash
# Training will automatically resume from latest checkpoint
# If interrupted, just run the same command again:

python mdlm_atat/scripts/train_atat.py \
  --config-name atat/production_training \
  --max-steps 500000
  
# It will detect latest checkpoint and continue
```

### Manual Resume
```bash
python mdlm_atat/scripts/train_atat.py \
  --config-name atat/production_training \
  --resume-from-checkpoint /media/scratch/adele/mdlm_fresh/checkpoints/epoch_0_step_250000.ckpt \
  --max-steps 500000
```

---

## Estimated Costs & Time

### Hardware
- **GPUs**: 6x RTX 4090 (24GB each)
- **Memory**: 144GB total GPU memory
- **Storage**: 50GB+ (40GB data + checkpoints)

### Timeline
- **Data Download**: ~30 minutes
- **Validation Run** (50k steps): ~1-2 hours
- **Production Training** (500k steps): ~1-2 weeks continuous

### Throughput
- **Batch Size**: 24 (4 per GPU × 6 GPUs)
- **Sequence Length**: 1024 tokens
- **Estimated Speed**: ~200-300 steps/hour
- **500k steps**: ~1,700-2,500 hours = ~70-104 days wall-clock at 1x speed
- **Expected**: 7-14 days with 6 GPUs running 24/7

---

## Before You Start

✅ **Checklist**:
- [ ] 40GB free space available
- [ ] All 6 GPUs available and healthy
- [ ] OpenWebText downloaded to cache dir
- [ ] `production_training.yaml` created
- [ ] Check `/media/scratch/adele/` has enough space

**Verify Setup**:
```bash
# Check space
df -h /media/scratch/adele/

# Check GPUs
nvidia-smi

# Check data
ls /media/scratch/adele/mdlm_fresh/data_cache/openwebtext/

# Dry run (0 steps, just loads everything)
python mdlm_atat/scripts/train_atat.py \
  --config-name atat/production_training \
  --max-steps 1
```

---

## Common Issues & Solutions

### Issue: "Out of Memory" (OOM)
**Solution**: Reduce batch_size in config (4 → 2 per GPU)

### Issue: "Dataset not found"
**Solution**: Download first: `python mdlm_atat/scripts/download_datasets.py --datasets openwebtext`

### Issue: "Training stuck/slow"
**Solution**: Check GPU utilization with `nvidia-smi`, verify data loading

### Issue: "Validation takes too long"
**Solution**: Reduce `val_check_interval` in config (e.g., every 5000 steps instead of 1000)

---

## Quick Commands

```bash
# Download data
python mdlm_atat/scripts/download_datasets.py --datasets openwebtext

# Start training
python mdlm_atat/scripts/train_atat.py --config-name atat/production_training --max-steps 500000

# Monitor GPU
watch -n 1 nvidia-smi

# Monitor logs
tail -f /media/scratch/adele/mdlm_fresh/logs/training_*.log

# List checkpoints
ls -lt /media/scratch/adele/mdlm_fresh/checkpoints/

# Evaluate
python mdlm_atat/scripts/eval_atat.py

# Create visualizations
python mdlm_atat/scripts/create_sampling_gif.py
```

---

## Summary

✅ **No manual dataset splitting needed** - HuggingFace handles it automatically
✅ **Production training ready** - Just create config and run
✅ **Auto-resume** - Training automatically continues from latest checkpoint
✅ **Fully monitored** - Checkpoints, logs, and metrics tracked continuously
✅ **Scalable** - Same setup works for other datasets (just change `data.train` and `data.valid`)

**Ready to start?**
```bash
# Step 1: Download
python mdlm_atat/scripts/download_datasets.py --datasets openwebtext

# Step 2: Create config (use template above)
cat > mdlm_atat/configs/atat/production_training.yaml << 'EOF'
[paste config template from above]
EOF

# Step 3: Train
python mdlm_atat/scripts/train_atat.py --config-name atat/production_training --max-steps 500000
```

🚀 **Let's go!**
