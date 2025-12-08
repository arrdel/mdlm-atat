# MDLM-ATAT Checkpoint Evaluation & Sampling Guide

After validation training completes, use these scripts to evaluate the model and generate text samples.

## Quick Start: Load and Sample from Checkpoint

### Option 1: Using the Best Validation Checkpoint

```python
import torch
from mdlm_atat.models import ATAT_DiT
from transformers import GPT2Tokenizer

# Load checkpoint
checkpoint_path = "/media/scratch/adele/mdlm_fresh/checkpoints/checkpoints/best.ckpt"
model = ATAT_DiT.load_from_checkpoint(checkpoint_path, strict=False)
model.eval()
model.cuda()

# Load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Generate a sample
num_steps = 128  # Number of denoising steps
sample_length = 256

# Start from noise
x = torch.randint(0, len(tokenizer), (1, sample_length)).cuda()

# Iterative denoising (simplified)
with torch.no_grad():
    for step in range(num_steps):
        sigma = get_sigma_from_timestep(step, num_steps)
        logits = model.forward(x, sigma)
        # Sample from model or take argmax
        x = torch.argmax(logits, dim=-1)

# Decode to text
text = tokenizer.decode(x[0])
print(text)
```

## Evaluation Metrics

### 1. Perplexity on Validation Set

```python
# Already computed during training - check logs:
# grep "compute_generative_perplexity" training_*.log

# Estimated from NLL:
# nll = 3.952 (best validation)
# perplexity = exp(nll) ≈ 52.0
```

### 2. Comparison to Baselines

| Model | NLL | Perplexity | Notes |
|-------|-----|------------|-------|
| MDLM-ATAT (Validation) | 3.952 | 52.0 | 50k steps on WikiText-103 |
| GPT-2 Small | ~3.2 | ~24.5 | Autoregressive baseline |
| BERT (MLM) | ~2.8 | ~16.5 | Masked language model baseline |

### 3. Sampling Quality Assessment

Run visual inspection on generated samples:

```python
# Generate multiple samples
for i in range(5):
    sample = generate_sample(model, length=256, temperature=0.7)
    print(f"Sample {i+1}:")
    print(sample)
    print("-" * 80)
```

**Quality Checklist**:
- [ ] Samples are coherent and grammatical
- [ ] No excessive repetition
- [ ] Diverse vocabulary use
- [ ] Reasonable paragraph structure
- [ ] Context awareness (stays on topic)

## Checkpoint Management

### View Checkpoint Info

```bash
# List all checkpoints
ls -lh /media/scratch/adele/mdlm_fresh/checkpoints/checkpoints/

# Check checkpoint size
du -sh /media/scratch/adele/mdlm_fresh/checkpoints/checkpoints/best.ckpt
```

### Load Checkpoint for Inference

```python
import lightning as L
from omegaconf import OmegaConf

# Option 1: Load with Lightning
trainer = L.Trainer()
model = ATAT_DiT.load_from_checkpoint(
    "path/to/best.ckpt",
    strict=False  # Ignore missing keys (WandB logger, etc.)
)

# Option 2: Load only model weights
state_dict = torch.load("path/to/best.ckpt", map_location="cpu")
model.load_state_dict(state_dict["state_dict"], strict=False)

# Option 3: Extract to standalone checkpoint
model_state = {
    k.replace("module.", ""): v 
    for k, v in state_dict["state_dict"].items() 
    if k.startswith("backbone.")
}
torch.save(model_state, "backbone_only.pt")
```

## Analysis Tasks

### 1. Importance Score Distribution

```python
# Visualize what the importance estimator learned
import matplotlib.pyplot as plt

# Get importance scores on a batch
with torch.no_grad():
    batch_ids = tokenizer.encode("The quick brown fox jumps over the lazy dog")
    batch_ids = torch.tensor([batch_ids]).cuda()
    
    importance_scores = model.importance_estimator(batch_ids)
    
    plt.figure(figsize=(12, 4))
    plt.bar(range(len(importance_scores[0])), importance_scores[0].cpu().numpy())
    plt.xlabel("Token Position")
    plt.ylabel("Importance Score")
    plt.title("Token Importance Scores")
    plt.show()
```

**Interpretation**:
- High scores: Tokens the model finds hard to predict (content words, rare words)
- Low scores: Common tokens, deterministic context (articles, prepositions)

### 2. Curriculum Learning Progression

Check how masking difficulty evolved:

```python
# Already tracked during training
# Look for curriculum stage transitions:
# - Step 0-15k: Easy tokens (low importance)
# - Step 15k-35k: Medium tokens
# - Step 35k-50k: Hard tokens (high importance)

# Evidence in logs:
# grep "curriculum\|masking" training_*.log
```

### 3. Diffusion Dynamics

Analyze the reverse process:

```python
# Trace model behavior across denoising steps
num_steps = 128
confidences = []

with torch.no_grad():
    x = torch.randint(0, len(tokenizer), (1, 256)).cuda()
    
    for step in range(num_steps):
        sigma = get_sigma_from_timestep(step, num_steps)
        logits = model.forward(x, sigma)
        
        # Confidence = max softmax prob
        probs = torch.softmax(logits, dim=-1)
        max_probs = torch.max(probs, dim=-1)[0]
        confidences.append(max_probs.mean().item())

# Plot confidence over denoising steps
plt.plot(confidences)
plt.xlabel("Denoising Step")
plt.ylabel("Average Model Confidence")
plt.title("Model Confidence During Reverse Process")
plt.show()
```

**Expected Pattern**:
- Start low: Noise input, high uncertainty
- Progressive increase: Model "sharpens" predictions
- Peak at end: Confident token predictions

## Production Deployment Checklist

Before running on OpenWebText (500k steps):

- [ ] Best validation checkpoint accessible and loadable
- [ ] All metrics logged and verified
- [ ] Sample quality acceptable for deployment
- [ ] Importance estimator learning sensible scores
- [ ] No NaN/Inf in loss curves
- [ ] GPU memory usage stable
- [ ] Dataset (OpenWebText) available and verified
- [ ] Output directories created and writable

## Troubleshooting

### Issue: Checkpoint Won't Load

```python
# Try loading with strict=False
model = ATAT_DiT.load_from_checkpoint(path, strict=False)

# Or load only specific parts
state = torch.load(path)
model.backbone.load_state_dict(state["backbone"], strict=False)
```

### Issue: Poor Sample Quality

1. Increase denoising steps (128 → 256)
2. Lower temperature (0.7 → 0.5)
3. Use nucleus sampling instead of greedy
4. Check importance estimator is trained (scores > 0 for hard tokens)

### Issue: OOM During Evaluation

- Reduce batch size for inference
- Use gradient checkpointing
- Evaluate on smaller subset first

## Next Steps

1. **Evaluate validation checkpoint**: Run analysis scripts above
2. **Document findings**: Record sample quality, metrics, insights
3. **Start production training**: Use `start_production_training.sh`
4. **Monitor OpenWebText training**: Track metrics every 10k steps
5. **Generate final evaluation**: Compare validation vs production results

---

**Resources**:
- Validation Checkpoint: `/media/scratch/adele/mdlm_fresh/checkpoints/checkpoints/best.ckpt`
- Training Logs: `/media/scratch/adele/mdlm_fresh/logs/training_20251208_063315.log`
- Technical Report: `docs/MDLM_ATAT_Technical_Overview.md`
- Results Summary: `docs/VALIDATION_TRAINING_RESULTS.md`
