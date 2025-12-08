# Quick Validation Training Setup Guide

## ⚠️ CRITICAL: Diffusion-Based Training

**REMEMBER**: We are training a **DISCRETE MASKED DIFFUSION MODEL**, not a standard autoregressive language model.

### Key Diffusion Concepts

```
FORWARD PROCESS (Masking):
X_0 (clean text) → X_1 (50% masked) → X_2 (75% masked) → ... → X_T (all masked)

REVERSE PROCESS (Denoising - what we train):
X_T (all masked) ← BERT predicts tokens ← X_{T-1} ← ... ← X_0 (final output)

TRAINING OBJECTIVE:
For each timestep t, predict original tokens from noisy (masked) version
Loss = weighted_crossentropy(predictions, original_tokens)
Weight = importance_score (from importance estimator)
```

### Key Configuration Parameters (ALL CRITICAL)

```yaml
diffusion: absorbing_state        # Discrete masking (not Gaussian!)
parameterization: subs            # Substitute (mask) tokens
noise:
  type: loglinear               # Noise schedule
  sigma_min: 1e-4               # Min noise (mostly clean)
  sigma_max: 20                 # Max noise (mostly masked)

sampling:
  predictor: ddpm_cache         # DDPM denoising schedule
  steps: 128                    # Number of denoising steps
```

---

## Pre-Training Checklist

### ✅ Step 1: Verify Project Structure

```bash
# Check ATAT components exist
ls -la mdlm_atat/atat/
# Should contain:
# - importance_estimator.py      ✓
# - adaptive_masking.py           ✓
# - curriculum.py                 ✓
# - uncertainty_sampler.py        ✓
```

### ✅ Step 2: Verify Configs

```bash
# Check config exists
ls -la mdlm_atat/configs/atat/
# Should include:
# - wikitext103_validation.yaml   ✓ (just created)
# - base_config.yaml
# - tiny.yaml
```

### ✅ Step 3: Set Up Python Environment

```bash
# Configure Python environment
cd /home/adelechinda/home/projects/mdlm
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r mdlm_atat/requirements.txt
# Should include: pytorch-lightning, torch, transformers, datasets, hydra-core, wandb
```

### ✅ Step 4: Verify Datasets Available

```bash
# Check WikiText-103 will be downloaded
python -c "from datasets import load_dataset; ds = load_dataset('wikitext', 'wikitext-103-v1'); print('WikiText-103 OK')"

# Verify GPT-2 tokenizer
python -c "from transformers import AutoTokenizer; tok = AutoTokenizer.from_pretrained('gpt2'); print('GPT-2 tokenizer OK')"

# Check storage space
df -h /media/scratch/adele/
# Should show available space (>100GB recommended)
```

### ✅ Step 5: Verify GPU Setup

```bash
# Check GPUs
nvidia-smi

# Expected output:
# GPU 0: [GPU name] | Memory: 80 GB
# GPU 1: [GPU name] | Memory: 80 GB
# ...etc

# Check PyTorch can see GPUs
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"
```

### ✅ Step 6: Set Up Logging

```bash
# Create log directories
mkdir -p /media/scratch/adele/mdlm_fresh/logs
mkdir -p /media/scratch/adele/mdlm_fresh/outputs
mkdir -p /media/scratch/adele/mdlm_fresh/checkpoints

# Verify permissions
touch /media/scratch/adele/mdlm_fresh/logs/test.log
echo "✓ Log directory writable"
```

---

## Quick Validation Training (50k steps, ~24 hours)

### Command 1: Download WikiText-103 Dataset

```bash
# One-time download (if not cached)
cd /home/adelechinda/home/projects/mdlm

python -c "
from datasets import load_dataset
import os

cache_dir = '/media/scratch/adele/mdlm_data_cache'
os.makedirs(cache_dir, exist_ok=True)

print('Downloading WikiText-103...')
dataset = load_dataset(
    'wikitext',
    'wikitext-103-v1',
    cache_dir=cache_dir
)
print(f'Train samples: {len(dataset[\"train\"])}')
print(f'Val samples: {len(dataset[\"validation\"])}')
print('✓ WikiText-103 ready for training')
"
```

**Expected Output**:
```
Downloading WikiText-103...
Train samples: 36718
Val samples: 3760
✓ WikiText-103 ready for training
```

**Time**: ~5 minutes (first time only, cached after)

---

### Command 2: Quick Sanity Check (100 steps)

```bash
# Verify training setup works (5-10 minutes)
cd /home/adelechinda/home/projects/mdlm

python mdlm_atat/scripts/train_atat.py \
  --config-name atat/wikitext103_validation \
  --max-steps 100 \
  --log-dir logs/sanity_check_$(date +%Y%m%d_%H%M%S) \
  2>&1 | tee sanity_check.log
```

**Expected Output**:
```
GPU(s) available: 6
Memory per GPU: ~45GB
Batch size: 8 per GPU
Global batch: 64

Sanity check: Testing model initialization... ✓
Sanity check: Testing data loading... ✓
Sanity check: Testing forward pass... ✓

Step 0/100   | Loss: 5.234  | LR: 0.0001
Step 10/100  | Loss: 5.187  | LR: 0.0001
Step 20/100  | Loss: 5.043  | LR: 0.0001
...
Step 100/100 | Loss: 4.891  | LR: 0.0001

✓ Sanity check PASSED
```

**If Errors**:
- ❌ CUDA out of memory → Reduce batch_size in config
- ❌ Dataset not found → Run dataset download command
- ❌ Module not found → Check environment setup

---

### Command 3: Full Validation Training (50k steps)

```bash
# Full validation run (12-24 hours depending on GPU)
cd /home/adelechinda/home/projects/mdlm

python mdlm_atat/scripts/train_atat.py \
  --config-name atat/wikitext103_validation \
  --max-steps 50000 \
  --log-dir logs/wikitext_validation_$(date +%Y%m%d_%H%M%S) \
  --wandb-project mdlm-atat \
  --wandb-run-name "wikitext_validation_$(date +%Y%m%d)" \
  2>&1 | tee training.log
```

**Expected Timeline**:

```
Timeline (50k steps, 1-2 GPUs):
├─ Step 0-10k (Warmup phase):
│  ├─ Duration: ~2-3 hours
│  ├─ Loss: 5.2 → 3.1
│  ├─ Focus: Common tokens (curriculum stage 1)
│  └─ Importance Estimator: Initializing
│
├─ Step 10k-30k (Medium phase):
│  ├─ Duration: ~6-8 hours
│  ├─ Loss: 3.1 → 2.1
│  ├─ Focus: Content words (curriculum stage 2)
│  └─ Importance Estimator: Learning patterns
│
├─ Step 30k-50k (Hard phase):
│  ├─ Duration: ~4-6 hours
│  ├─ Loss: 2.1 → 1.8
│  ├─ Focus: Rare words (curriculum stage 3)
│  └─ Importance Estimator: Fine-tuning
│
└─ TOTAL: ~12-24 hours

Final Metrics Expected:
├─ Training Loss: ~1.8
├─ Validation Loss: ~2.1
├─ Perplexity: ~8.2
└─ Status: ✓ Ready for full OpenWebText training
```

---

## What to Monitor During Training

### Terminal Output

```bash
# Watch logs in real-time
tail -f logs/wikitext_validation_*/train.log

# Key metrics to watch:
# ✓ Loss decreasing (should not increase)
# ✓ LR following schedule (warming up then decaying)
# ✓ GPU memory stable
# ✓ Training speed: ~1000-2000 tokens/sec
```

### WandB Dashboard

```bash
# Open browser to:
https://wandb.ai/[your-entity]/mdlm-atat

# Track in real-time:
# - Training loss (should decrease smoothly)
# - Validation loss (should decrease)
# - Importance scores distribution (should converge)
# - Curriculum stage transitions (kinks at 1/3 and 2/3 of training)
# - GPU utilization (should be ~80%+)
```

### Key Checkpoints

```
Training Progress Indicators:

✓ Step 1000:
  - Loss should drop from 5.2 to ~4.5
  - No errors in logs
  - GPU running smoothly

✓ Step 10000:
  - Loss should reach ~3.1
  - Validation metrics computed
  - Curriculum stage 1 complete

✓ Step 30000:
  - Loss should reach ~2.1
  - Curriculum stage 2 complete
  - Importance estimator learning visible

✓ Step 50000:
  - Final loss ~1.8
  - All curriculum stages complete
  - Ready for production training
```

---

## Diffusion-Specific Monitoring

### Verify Diffusion is Working

```bash
# After 1000 steps, check logs for:
grep "noise_level\|masking_rate\|denoising" logs/wikitext_validation_*/train.log

# Expected patterns:
# Step 500: noise_level=18.5, masking_rate=0.95  (highly masked)
# Step 501: noise_level=15.2, masking_rate=0.78  (less masked)
# Step 502: noise_level=12.1, masking_rate=0.62  (even less)
# ...
# Step 1000: noise_level=0.8, masking_rate=0.05  (mostly clean)

# This shows forward diffusion process is working ✓
```

### Verify Importance Estimator is Learning

```bash
# After 5000 steps, check:
grep "importance_" logs/wikitext_validation_*/train.log | head -20

# Expected output:
# importance_mean: 0.48 (converging to 0.5)
# importance_std: 0.15 (learning structure)
# importance_min: 0.02 (some easy tokens)
# importance_max: 0.98 (some hard tokens)

# This shows importance estimator is learning ✓
```

---

## Troubleshooting

### Issue 1: CUDA Out of Memory

```bash
# Reduce batch size
# Edit: mdlm_atat/configs/atat/wikitext103_validation.yaml

loader:
  batch_size: 4        # Changed from 8
  global_batch_size: 32  # 4 * 8 GPUs
```

### Issue 2: Training Very Slow

```bash
# Check GPU utilization:
nvidia-smi -lms 1000  # Update every 1 second

# If < 50% utilized:
# 1. Increase batch_size
# 2. Increase num_workers
# 3. Check disk I/O (dataset loading bottleneck)
```

### Issue 3: Loss Not Decreasing

```bash
# Check diffusion is running:
# Verify in logs: "noise_level" and "masking_rate" changing

# If loss flat after 1000 steps:
# 1. Check learning rate (should be 1e-4)
# 2. Verify gradients flowing (check gradient logs)
# 3. Restart with different seed
```

### Issue 4: Importance Estimator Not Learning

```bash
# After 5000 steps, check:
# importance_std should be > 0.1 (not all same value)

# If importance_std ≈ 0 (not learning):
# 1. Check importance_loss_weight (should be 0.1)
# 2. Verify curriculum is enabled
# 3. Check gradients through importance estimator
```

---

## After Validation Training Complete

### Step 1: Verify Results

```bash
# Check final checkpoint saved
ls -lh logs/wikitext_validation_*/checkpoints/
# Should have: best.ckpt (~500MB)

# Verify final metrics
tail -50 logs/wikitext_validation_*/train.log | grep "val_loss\|perplexity"
# Should show: val_loss ≈ 2.1, perplexity ≈ 8.2
```

### Step 2: Collect Metrics

```bash
# Save final results
cat logs/wikitext_validation_*/train.log | \
  grep "Step.*Loss" > validation_results.txt

echo "Validation Training Complete!"
cat validation_results.txt | tail -5
```

### Step 3: Decision Point

```bash
If validation successful (✓ loss decreasing, no errors):
  → Proceed to full OpenWebText training
  → Use same config as starting point
  → Expected: 1-2 weeks on 8x A100

If validation shows issues:
  → Debug specific problem
  → Adjust config and retry
  → Don't proceed to full training
```

---

## Full Production Training (Coming Next)

After validation passes, will train on OpenWebText (40GB) with:

```yaml
training:
  max_steps: 500000          # ~1-2 weeks
  val_check_interval: 5000   # Validate frequently
  
data:
  train: openwebtext
  valid: openwebtext
```

---

## Quick Reference Commands

```bash
# Download datasets
python -c "from datasets import load_dataset; load_dataset('wikitext', 'wikitext-103-v1', cache_dir='/media/scratch/adele/mdlm_data_cache')"

# Quick sanity check (5-10 min)
python mdlm_atat/scripts/train_atat.py --config-name atat/wikitext103_validation --max-steps 100

# Full validation (12-24 hours)
python mdlm_atat/scripts/train_atat.py --config-name atat/wikitext103_validation --max-steps 50000

# Monitor training
tail -f logs/wikitext_validation_*/train.log

# Check best checkpoint
ls -lh logs/wikitext_validation_*/checkpoints/best.ckpt

# View final results
tail -100 logs/wikitext_validation_*/train.log | grep "loss\|perplexity"
```

---

## Critical Reminders

🔴 **DO NOT FORGET**:
- ✅ Discrete **masking** diffusion (not Gaussian noise)
- ✅ Diffusion forward/reverse process in place
- ✅ Importance estimator trained jointly with BERT
- ✅ Curriculum learning active (stages 1→2→3)
- ✅ Noise schedule configured correctly
- ✅ DDPM denoising predictor active

These are embedded in the config but VERIFY they're working!

---

*Document Version: 1.0*  
*Created: December 8, 2025*  
*Status: Ready to Train*
