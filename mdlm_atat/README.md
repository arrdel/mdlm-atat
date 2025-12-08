# MDLM-ATAT: Adaptive Time-Aware Token Masking for Masked Diffusion Language Models

This project extends the **Masked Diffusion Language Models (MDLM)** framework from [Sahoo et al., NeurIPS 2024](https://arxiv.org/abs/2406.07524) with a novel **Adaptive Time-Aware Token Masking (ATAT)** mechanism.

## 🎯 Key Innovation: ATAT

ATAT enhances masked diffusion language models by learning which tokens are more important/difficult and adapting the masking strategy accordingly. Unlike baseline MDLM which uses uniform masking probabilities, ATAT introduces:

1. **Importance Estimation**: A lightweight neural network that predicts token-level importance based on context and diffusion timestep
2. **Adaptive Masking**: Dynamically adjusts masking probabilities based on learned importance scores
3. **Curriculum Learning**: Progressively transitions from easy to hard tokens during training
4. **Uncertainty-Guided Sampling**: Uses model confidence to guide the denoising process at inference

## 📁 Project Structure

```
mdlm/                           # Original MDLM baseline code
├── diffusion.py                # Core diffusion implementations (SUBS, D3PM, SEDD)
├── main.py                     # Training entry point
├── models/                     # Model architectures (DiT, DiMamba, AR)
├── configs/                    # Hydra configurations
└── scripts/                    # Training/evaluation scripts

mdlm_atat/                      # ATAT extension (our contribution)
├── __init__.py
├── atat/                       # Core ATAT components
│   ├── importance_estimator.py # Token importance network
│   ├── adaptive_masking.py     # Adaptive masking scheduler
│   ├── curriculum.py           # Curriculum learning
│   └── uncertainty_sampler.py  # Uncertainty-guided sampling
├── models/                     # ATAT-enhanced models
│   └── atat_dit.py            # ATATDiT (DiT + ATAT)
├── configs/                    # ATAT configurations
│   └── atat/                  # Model configs & ablations
├── scripts/                    # Training & evaluation scripts
└── tests/                      # Unit tests
```

## 🚀 Quick Start

### 1. Environment Setup

The project requires Python 3.9+ and CUDA 12.1. Install dependencies:

```bash
# Create conda environment
conda create -n mdlm-atat python=3.9
conda activate mdlm-atat

# Install PyTorch with CUDA 12.1
pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install -r requirements.txt

# Install flash-attention (optional but recommended)
pip install flash-attn==2.8.3 --no-build-isolation
```

### 2. Training ATAT Model

Train ATAT-enhanced MDLM on OpenWebText:

```bash
cd mdlm_atat/scripts
bash train_owt_atat.sh
```

Train baseline MDLM for comparison:

```bash
bash train_owt_baseline.sh
```

### 3. Ablation Studies

Test individual ATAT components:

```bash
# Importance estimation only
bash ablation_importance_only.sh

# No curriculum learning
bash ablation_no_curriculum.sh
```

### 4. Evaluation

Evaluate trained model:

```bash
bash eval_atat.sh <checkpoint_path> <data_config>
```

Example:
```bash
bash eval_atat.sh outputs/atat_owt_tiny/checkpoints/last.ckpt openwebtext
```

## 🔧 Configuration

ATAT uses Hydra for configuration management. Key configs:

### Model Configuration (`configs/atat/tiny.yaml`)

```yaml
model:
  name: atat_dit
  dim: 256
  n_layers: 8
  importance_hidden_dim: 128
  masking_strategy: "importance"
  
use_importance: true
use_adaptive_masking: true
use_curriculum: true
```

### ATAT Hyperparameters (`configs/atat/atat_config.yaml`)

```yaml
importance_estimator:
  hidden_dim: 256
  num_layers: 2
  
adaptive_masking:
  temperature: 1.0
  position_bias: false
  
curriculum:
  warmup_steps: 1000
  easy_fraction: 0.3
  
training:
  importance_loss_weight: 0.1
```

## 📊 Expected Results

ATAT should improve over baseline MDLM on:

1. **Perplexity**: Better language modeling performance
2. **Sample Quality**: More coherent generated text
3. **Training Efficiency**: Faster convergence through curriculum
4. **Token-Level Analysis**: Interpretable importance scores

## 🧪 Experiments

### Main Comparison
- **Baseline MDLM**: Standard uniform masking
- **ATAT (Full)**: All components enabled
- **Expected Gain**: 5-10% perplexity improvement

### Ablations
1. **No Importance**: Tests baseline vs ATAT
2. **Importance Only**: Tests adaptive masking contribution
3. **No Curriculum**: Tests curriculum learning contribution

## 📝 Implementation Details

### ImportanceEstimator
- Transformer-based network (2 layers, 256 dim)
- Time-conditioned (uses diffusion timestep)
- Outputs importance score ∈ [0, 1] per token

### AdaptiveMaskingScheduler
- Converts importance to masking probability
- Temperature-controlled sharpness
- Optional position bias

### CurriculumScheduler
- 3-stage progression (easy → medium → hard)
- Linear warmup over first 1000 steps
- Dynamically adjusts token difficulty thresholds

### UncertaintyGuidedSampler
- Entropy-based uncertainty metric
- Denoise uncertain tokens first
- Adaptive number of sampling steps

## 🔬 Key Code Paths

### Training Loop Integration
```python
# In mdlm_atat/models/atat_dit.py
def training_step(x_0, t, step):
    # 1. Adaptive forward diffusion
    x_t, importance, _ = model.adaptive_forward_diffusion(x_0, t, step)
    
    # 2. Model prediction
    logits, pred_importance = model(x_t, t, return_importance=True)
    
    # 3. Compute losses
    main_loss = F.cross_entropy(logits, x_0)
    importance_loss = compute_importance_loss(pred_importance, x_0, x_t)
    
    return main_loss + λ * importance_loss
```

### Inference
```python
# Uncertainty-guided sampling
x_t, uncertainty = model.uncertainty_guided_sample_step(
    x_t, t, dt, mask_index
)
```

## 📚 Citation

If you use this code, please cite both the original MDLM paper and this extension:

```bibtex
@inproceedings{sahoo2024simple,
  title={Simple and Effective Masked Diffusion Language Models},
  author={Sahoo, Subham and others},
  booktitle={NeurIPS},
  year={2024}
}

@software{mdlm_atat,
  title={ATAT: Adaptive Time-Aware Token Masking for Masked Diffusion Language Models},
  author={Your Name},
  year={2024}
}
```

## 🤝 Contributing

This is a research project. Feel free to:
- Open issues for bugs or questions
- Submit PRs for improvements
- Share experimental results

## 📄 License

This project inherits the Apache 2.0 license from the original MDLM codebase.

## 🙏 Acknowledgments

Built on top of the excellent [MDLM](https://github.com/kuleshov-group/mdlm) codebase by Sahoo et al.

---

**Status**: 🚧 Active Research Project

For questions, please open an issue or contact [your email].
