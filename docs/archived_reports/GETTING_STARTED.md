# Getting Started with MDLM-ATAT

This guide will help you get started with training and evaluating ATAT-enhanced masked diffusion language models.

## Prerequisites

- Python 3.9+
- CUDA 12.1 compatible GPU
- ~16GB GPU memory for tiny model
- ~32GB GPU memory for small model

## Installation

### 1. Clone and Setup Environment

```bash
# Navigate to project directory
cd /home/adelechinda/home/projects/mdlm

# Activate conda environment (already created)
conda activate mdlm-atat

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 2. Verify ATAT Installation

```bash
# Test imports
python -c "from mdlm_atat import ATATDiT, ImportanceEstimator"
echo "ATAT components imported successfully!"

# Run tests (optional)
cd mdlm_atat
pytest tests/ -v
```

## Quick Start: Training

### 1. Train Baseline MDLM (for comparison)

```bash
cd mdlm_atat/scripts
chmod +x train_owt_baseline.sh
bash train_owt_baseline.sh
```

This will:
- Download OpenWebText dataset automatically
- Train a tiny DiT model with standard uniform masking
- Log metrics to Weights & Biases
- Save checkpoints every 5000 steps

### 2. Train ATAT Model

```bash
chmod +x train_owt_atat.sh
bash train_owt_atat.sh
```

This will:
- Train the same tiny DiT but with ATAT enhancements
- Use importance estimation and adaptive masking
- Apply curriculum learning
- Log additional ATAT-specific metrics

### 3. Monitor Training

Training progress is logged to Weights & Biases. Key metrics to watch:

**Standard Metrics:**
- `loss/total`: Overall training loss
- `val/perplexity`: Validation perplexity (lower is better)

**ATAT-Specific Metrics:**
- `loss/importance`: Importance estimation loss
- `importance/mean`: Average importance score
- `importance/std`: Importance score diversity
- `curriculum/stage`: Current curriculum stage (easy/medium/hard)

## Configuration

### Model Size Options

Edit the script to use different model sizes:

```bash
# Tiny model (256 dim, 8 layers) - fast experiments
model=atat/tiny

# Small model (512 dim, 12 layers) - better performance
model=atat/small
```

### ATAT Hyperparameters

Key hyperparameters in `configs/atat/atat_config.yaml`:

```yaml
# Importance estimation
importance_estimator:
  hidden_dim: 256          # Importance network size
  num_layers: 2            # Number of transformer layers
  
# Adaptive masking
adaptive_masking:
  temperature: 1.0         # Lower = sharper importance weighting
  position_bias: false     # Enable position-aware masking
  
# Curriculum learning
curriculum:
  warmup_steps: 1000       # Curriculum warmup duration
  easy_fraction: 0.3       # Fraction of "easy" tokens early on
  
# Training
training:
  importance_loss_weight: 0.1  # Weight for importance loss
```

### Dataset Options

Available datasets (in `mdlm/configs/data/`):

```bash
# Small datasets (for quick experiments)
data=text8          # 100MB text
data=wikitext2      # ~4MB

# Medium datasets
data=wikitext103    # ~500MB
data=openwebtext    # ~40GB (recommended)

# Large datasets
data=lm1b           # ~1.5B tokens
```

## Evaluation

### Evaluate Trained Model

```bash
# Evaluate ATAT model
chmod +x eval_atat.sh
bash eval_atat.sh outputs/atat_owt_tiny/checkpoints/last.ckpt openwebtext
```

This will compute:
- Validation perplexity
- Importance score statistics
- Uncertainty metrics
- Sample quality

### Generate Samples

```python
import torch
from mdlm_atat.models import ATATDiT
from transformers import AutoTokenizer

# Load model
model = ATATDiT.load_from_checkpoint('path/to/checkpoint.ckpt')
model.eval()
model.cuda()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('gpt2')

# Generate
prompt = "The future of AI is"
input_ids = tokenizer.encode(prompt, return_tensors='pt').cuda()

# Sample with uncertainty guidance
with torch.no_grad():
    samples = model.sample(
        batch_size=1,
        length=100,
        prompt=input_ids,
        use_uncertainty_guidance=True,
    )
    
generated_text = tokenizer.decode(samples[0])
print(generated_text)
```

## Ablation Studies

Test individual components:

```bash
# Test importance estimation alone
bash ablation_importance_only.sh

# Test without curriculum
bash ablation_no_curriculum.sh
```

Compare results to understand each component's contribution.

## Troubleshooting

### Out of Memory

If you get OOM errors:

1. Reduce batch size in config:
```yaml
data:
  batch_size: 32  # Default: 64
```

2. Use gradient accumulation:
```yaml
trainer:
  accumulate_grad_batches: 2
```

3. Use smaller model:
```bash
model=atat/tiny
```

### Slow Training

Enable flash-attention for 2-3x speedup:

```bash
pip install flash-attn==2.8.3 --no-build-isolation
```

### Import Errors

Make sure to run from project root:

```bash
cd /home/adelechinda/home/projects/mdlm
python mdlm/main.py ...
```

## Expected Timeline

**Tiny Model on Single V100:**
- Training: ~24 hours for 100k steps
- Validation: ~5 minutes per checkpoint
- Sampling: ~1 second per 100 tokens

**Small Model on Single V100:**
- Training: ~48 hours for 100k steps
- More GPU memory required

## Next Steps

1. **Reproduce Baseline**: Train baseline MDLM first
2. **Train ATAT**: Train full ATAT model
3. **Compare**: Analyze perplexity improvement
4. **Ablations**: Test individual components
5. **Analyze**: Visualize importance scores
6. **Tune**: Adjust hyperparameters based on results

## Resources

- [MDLM Paper](https://arxiv.org/abs/2406.07524)
- [Original MDLM Code](https://github.com/kuleshov-group/mdlm)
- [PyTorch Lightning Docs](https://lightning.ai/docs/pytorch/stable/)
- [Hydra Configuration](https://hydra.cc/docs/intro/)

## Support

For issues or questions:
1. Check existing GitHub issues
2. Open a new issue with:
   - Error message
   - Config used
   - System info (GPU, PyTorch version)
   - Minimal reproduction steps

Happy experimenting! 🚀
