# MDLM Uniform Baseline

This directory contains the **MDLM Uniform Baseline** implementation - standard masked discrete diffusion with uniform random masking (no importance weighting or adaptive strategies).

## Purpose

This baseline serves as the **primary comparison** for ATAT's adaptive masking approach:

- **MDLM Uniform** (this): Standard diffusion with uniform random masking
- **ATAT** (our method): Diffusion with importance-weighted adaptive masking

If ATAT achieves lower perplexity than MDLM, it proves that adaptive content-aware masking is superior to uniform random masking.

## Architecture

- **Backbone**: DiT (Diffusion Transformer)
- **Hidden size**: 768
- **Layers**: 12
- **Parameters**: ~169M
- **Masking**: Uniform random (no importance weighting)
- **Training**: Standard MDLM without importance sampling or curriculum

## Key Configuration

The config (`mdlm_baseline_config_v2.yaml`) uses the standard MDLM structure with:

```yaml
training:
  importance_sampling: False  # Uniform random masking only
```

All other settings match ATAT for fair comparison:
- Batch size: 512 (global)
- Learning rate: 3e-4 with cosine decay
- Max steps: 500K
- Data: OpenWebText

## Usage

### Test Run (10 steps)

```bash
cd mdlm_atat/baselines/mdlm
python train_mdlm_baseline.py --max-steps 10 --num-gpus 1 --no-wandb
```

### Full Training (500K steps, ~55-65 hours)

```bash
cd mdlm_atat/baselines/mdlm
python train_mdlm_baseline.py --max-steps 500000 --num-gpus 2
```

### Monitor Training

```bash
# Check GPU usage
watch -n 60 nvidia-smi

# Monitor logs
tail -f /media/scratch/adele/mdlm_fresh/outputs/baselines/mdlm_uniform/logs/*.log

# WandB dashboard
# Project: mdlm-atat-baselines
# Run: mdlm_uniform_baseline
```

## Expected Results

Based on MDLM paper (Sahoo et al. 2024) and similar work:

| Metric | Expected Value | Comparison |
|--------|---------------|------------|
| Validation PPL | 40-43 | Baseline (no adaptive masking) |
| Training time | 55-65 hours | 2x RTX 4090 |
| Memory/GPU | ~16GB | Similar to ATAT |

## Comparison Timeline

1. **Phase 1B** (running now): ATAT with 4 masking strategies
   - Expected completion: Jan 16-19
   
2. **Phase 2 - Baselines**: Run after Phase 1B completes
   - AR Transformer: 50-60 hours → PPL 35-38 (gold standard)
   - MDLM Uniform: 55-65 hours → PPL 40-43 (this baseline)
   
3. **Results Analysis**:
   - If ATAT achieves PPL 38-40, it's:
     - Only 5-7% worse than AR (proves competitive)
     - 5-7% better than MDLM (proves adaptive masking works)

## Files

- `mdlm_baseline_config_v2.yaml`: Training configuration (follows MDLM structure)
- `train_mdlm_baseline.py`: Training script (wraps MDLM main.py)
- `README.md`: This file

## Implementation Notes

This baseline **reuses the existing MDLM codebase** from `mdlm/` with:
- Config pointing to uniform masking (`importance_sampling: False`)
- No modifications to model architecture
- Same data pipeline as ATAT (OpenWebText)
- Output directory: `/media/scratch/adele/mdlm_fresh/outputs/baselines/mdlm_uniform`

The key difference from ATAT is the **absence of importance estimation and adaptive masking strategies** - this uses standard uniform random masking throughout training.

## References

- MDLM paper: Sahoo et al., "Simple and Effective Masked Diffusion Language Models", 2024
- MDLM code: `mdlm/` directory in this repository
- ATAT code: `mdlm_atat/` directory in this repository
