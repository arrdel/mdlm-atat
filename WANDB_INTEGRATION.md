# 🎯 WandB Integration Guide

## ✅ WandB Already Integrated!

Your training script already has **complete WandB integration** for real-time online logging!

---

## 🚀 Quick Setup

### Step 1: Install WandB
```bash
pip install wandb
```

### Step 2: Login to WandB
```bash
wandb login
# This opens a browser to get your API key
# Paste the key into the terminal
```

### Step 3: Start Training with WandB Logging
```bash
python mdlm_atat/scripts/train_atat.py \
  --config-name atat/production_training \
  --max-steps 500000 \
  --wandb-project "mdlm-atat" \
  --wandb-run-name "owt_500k_production"
```

### Step 4: View Real-Time Logs
Open: **https://wandb.ai/your-username/mdlm-atat**

---

## 📊 What Gets Logged

### Training Metrics
- **Loss**: Training and validation loss at each step
- **Learning Rate**: LR schedule progress
- **Gradient Norm**: Gradient magnitude (for debugging)
- **Epoch**: Training progress
- **Step**: Absolute step counter

### Model Metrics
- **Validation Loss**: Evaluated every 1,000 steps
- **Perplexity**: Language modeling metrics
- **ATAT Metrics**: 
  - Importance scores
  - Masking ratios
  - Curriculum progression

### System Metrics
- **GPU Memory**: Memory usage per GPU
- **GPU Utilization**: GPU % in use
- **Training Speed**: Steps per second
- **Wall Clock Time**: Total training time

### Checkpoints
- **Best Model**: Auto-saves best checkpoint
- **Latest Checkpoint**: Most recent save
- **Checkpoint Info**: Step, loss, metrics

---

## 🔧 Configuration

### Production Config Already Includes WandB
Your `mdlm_atat/configs/atat/production_training.yaml` already has WandB:

```yaml
# WandB logging configuration (already in your config)
wandb:
  project: "mdlm-atat"
  entity: null  # Your username
  name: null    # Auto-generated timestamp
  mode: "online"  # or "offline"
  tags:
    - "production"
    - "atat"
    - "openwebtext"
  notes: "Full 500k step production training"
```

---

## 📈 Usage Examples

### Example 1: Basic Production Training with WandB
```bash
python mdlm_atat/scripts/train_atat.py \
  --config-name atat/production_training \
  --max-steps 500000 \
  --no-confirm
```

### Example 2: Training with Custom Run Name
```bash
python mdlm_atat/scripts/train_atat.py \
  --config-name atat/production_training \
  --max-steps 500000 \
  --wandb-run-name "experiment_v1_longer_warmup"
```

### Example 3: Offline Mode (No Internet)
```bash
python mdlm_atat/scripts/train_atat.py \
  --config-name atat/production_training \
  --max-steps 500000 \
  --wandb-offline
```

Logs will sync later when internet is available:
```bash
wandb sync ./wandb/offline-run-*
```

### Example 4: Custom Project and Entity
```bash
python mdlm_atat/scripts/train_atat.py \
  --config-name atat/production_training \
  --max-steps 500000 \
  --wandb-project "my-custom-project" \
  --wandb-run-name "custom_name"
```

---

## 🖥️ WandB Dashboard Features

Once you start training, visit: **https://wandb.ai**

### You'll See:

#### 1. **Real-time Metrics**
```
Loss Graph:
├─ Training Loss (smoothed curve)
├─ Validation Loss (every 1000 steps)
└─ Learning Rate (schedule progress)

Metrics:
├─ Steps completed
├─ Tokens processed
├─ Training speed (steps/sec)
└─ GPU utilization
```

#### 2. **System Monitoring**
```
Hardware:
├─ GPU Memory (MB)
├─ GPU Utilization (%)
├─ CPU Usage
└─ System Temperature
```

#### 3. **Configuration Tracking**
```
All hyperparameters logged:
├─ Model architecture
├─ Training settings
├─ Data configuration
├─ Learning rate schedule
└─ ATAT parameters
```

#### 4. **Artifact Management**
```
Checkpoints:
├─ Best model (by validation loss)
├─ Latest checkpoint
├─ Custom saved runs
└─ Download for local use
```

#### 5. **Comparison Tools**
```
Compare multiple runs:
├─ Different learning rates
├─ Different batch sizes
├─ Different models
└─ Different datasets
```

---

## 📊 Viewing Logs Online

### During Training
```
Dashboard Updates:
• Real-time metrics (refresh every 10-30 seconds)
• System resource monitoring
• Training progress bar
• Estimated time remaining
• Alert on anomalies
```

### Log Types

**1. Scalar Metrics** (Line Graphs)
```
training/loss           ← Training loss over steps
validation/loss         ← Validation loss
learning_rate           ← LR schedule
gpu_memory_mb           ← GPU memory usage
```

**2. Histograms**
```
gradients               ← Gradient distribution
weights                 ← Weight distribution
layer_output           ← Layer output statistics
```

**3. Tables**
```
config                  ← All hyperparameters
model_summary          ← Model architecture
```

**4. Media**
```
loss_curves            ← Saved plots
gpu_utilization        ← System metrics plots
```

---

## 🔍 Advanced Features

### 1. Custom Metrics
Add custom logging in your trainer:
```python
# In train_atat.py or main.py
wandb.log({
  "custom_metric": value,
  "step": current_step
})
```

### 2. Checkpoint Logging
```python
# Save checkpoints to WandB
wandb.save("/path/to/checkpoint.ckpt")
```

### 3. Alerts
Set up alerts in WandB dashboard:
- Loss spike detection
- Training stall detection
- GPU/Memory issues
- Training complete notification

### 4. Artifact Versioning
```python
# Log artifacts for version control
artifact = wandb.Artifact("model-v1", type="model")
artifact.add_file("/path/to/model.ckpt")
wandb.log_artifact(artifact)
```

---

## 💾 Offline Mode & Syncing

### If Internet Drops During Training
```bash
# Training automatically saves offline logs
# No need to restart training

# Later, when internet returns:
wandb sync ./wandb/offline-*
# This uploads all offline logs to WandB
```

### Check Offline Runs
```bash
ls -la ./wandb/offline-*/
```

---

## 🎯 Best Practices

### 1. Run Names
Use descriptive names:
```bash
--wandb-run-name "owt_500k_lr1e4_bs24_atat_v2"
# ✓ Good (describes experiment)

--wandb-run-name "run1"
# ✗ Avoid (not descriptive)
```

### 2. Tags
Add tags for filtering:
```yaml
wandb:
  tags:
    - "production"
    - "atat-enabled"
    - "openwebtext"
    - "v2"
```

### 3. Notes
Document the experiment:
```bash
# Add notes field to config
wandb:
  notes: "Testing longer curriculum warmup (5000 steps)"
```

### 4. Group Related Runs
```yaml
wandb:
  group: "hyperparameter-search"
  tags: ["lr_sweep"]
```

Then compare all runs in the group on WandB dashboard.

---

## 🚨 Troubleshooting

### Issue: "WandB not installed"
```bash
pip install wandb
```

### Issue: "Not logged in"
```bash
wandb login
# Paste your API key
```

### Issue: "Can't reach wandb.ai"
```bash
# Use offline mode automatically
--wandb-offline
# Logs sync when internet returns
```

### Issue: "Too many logs (network heavy)"
Reduce logging frequency in config:
```yaml
trainer:
  log_every_n_steps: 500  # Log every 500 steps instead of 100
```

### Issue: "Project not found"
```bash
# Create the project on WandB first:
# https://wandb.ai/your-username/create-project
# Or it will be created automatically on first run
```

---

## 📚 Useful Commands

### View All Runs
```bash
wandb runs mdlm-atat
```

### Download Run Data
```bash
wandb pull your-username/mdlm-atat/run-id
```

### Compare Runs
```bash
wandb compare your-username/mdlm-atat run1 run2 run3
```

### Create Report
On WandB dashboard:
- Select runs to compare
- Create custom report
- Share with team
- Embed in presentations

---

## 🎓 Full Training Command with WandB

```bash
#!/bin/bash
# Full production training with WandB

cd /home/adelechinda/home/projects/mdlm

# Step 1: Download dataset (one-time)
echo "Downloading dataset..."
python mdlm_atat/scripts/download_datasets.py --datasets openwebtext

# Step 2: Start training with WandB
echo "Starting training with WandB logging..."
python mdlm_atat/scripts/train_atat.py \
  --config-name atat/production_training \
  --max-steps 500000 \
  --wandb-project "mdlm-atat" \
  --wandb-run-name "production_owt_$(date +%Y%m%d_%H%M%S)" \
  --no-confirm

# Step 3: Sync offline logs (if training was interrupted)
echo "Syncing offline logs..."
wandb sync ./wandb/offline-*

echo "Training complete!"
echo "View results at: https://wandb.ai"
```

---

## 📊 Sample Metrics Dashboard

After 1000 steps, you'll see:

```
Training Loss:        0.8234
Validation Loss:      0.9123
Learning Rate:        1e-4
Steps/Second:         12.5
GPU Memory Used:      22.3GB / 24GB
GPU Utilization:      92%
Time Elapsed:         1m 30s
Estimated Time Left:  13d 5h 22m
```

All updating in real-time on your WandB dashboard!

---

## 🔗 Resources

- **WandB Home**: https://wandb.ai
- **WandB Docs**: https://docs.wandb.ai
- **PyTorch Lightning Integration**: https://docs.wandb.ai/guides/integrations/lightning
- **WandB API Reference**: https://docs.wandb.ai/ref/python

---

## ✅ Summary

✓ WandB fully integrated
✓ Real-time online logging
✓ System monitoring included
✓ Checkpoint tracking
✓ Offline mode available
✓ Easy to use commands
✓ Beautiful dashboard

**Start training with WandB:**
```bash
python mdlm_atat/scripts/train_atat.py \
  --config-name atat/production_training \
  --max-steps 500000 \
  --no-confirm
```

**View results at:** https://wandb.ai

🚀 **Let's monitor training in real-time!**
