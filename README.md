# MDLM-ATAT: Adaptive Time-Aware Token Masking

**Extended Masked Diffusion Language Models with Adaptive Token-Level Learning**

> 🔬 Research project extending [MDLM (Sahoo et al., NeurIPS 2024)](https://arxiv.org/abs/2406.07524) with adaptive token masking strategies.

---

## 🎯 Project Overview

This repository contains:

1. **MDLM Baseline** (`mdlm/`) - Original Masked Diffusion Language Models implementation
2. **ATAT Extension** (`mdlm_atat/`) - Our novel Adaptive Time-Aware Token Masking framework
3. **Research Documentation** (`docs/`) - Papers, reports, and guides

### Key Innovation: ATAT Framework

ATAT enhances masked diffusion models by learning token-level importance and adapting masking strategies accordingly:

- 🧠 **Importance Estimation**: Learns which tokens are difficult/important
- 🎯 **Adaptive Masking**: Dynamically adjusts masking based on token importance
- 📈 **Curriculum Learning**: Progressive easy → hard training
- ⚡ **Uncertainty Sampling**: Adaptive inference for 30% speedup

**Expected Results**: 10% perplexity improvement, 20% faster convergence, 30% inference speedup

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone <repo-url>
cd mdlm

# Create environment
conda env create -f requirements.yaml
conda activate mdlm

# Install ATAT package
pip install -e mdlm_atat/
```

### Run Your First ATAT Training

```bash
# Quick test with tiny model (10k steps, ~30 min)
python mdlm_atat/scripts/train_atat.py \
  --config-name atat/tiny \
  trainer.max_steps=10000

# Full tiny model training (100k steps, ~12 hours)
python mdlm_atat/scripts/train_atat.py --config-name atat/tiny

# Production small model (100k steps, ~3 days on 8x V100)
python mdlm_atat/scripts/train_atat.py --config-name atat/small
```

### Verify Dataset Access

```bash
# Check scratch storage
ls -lh /media/scratch/adele/

# Expected output:
# mdlm_data_cache/  (47GB)   - HuggingFace cache
# mdlm_fresh/       (321GB)  - Outputs & checkpoints
# datasets/         (50GB)   - Raw datasets
```

---

## 📁 Repository Structure

```
mdlm/
├── README.md                    # This file
├── requirements.yaml            # Conda environment
│
├── mdlm/                       # 🔵 Base MDLM (original)
│   ├── README.md               # Original MDLM docs
│   ├── main.py                 # Training entry point
│   ├── diffusion.py            # Diffusion processes
│   ├── models/                 # DiT, DiMamba, AR
│   ├── configs/                # Hydra configs
│   └── scripts/                # Training scripts
│
├── mdlm_atat/                  # 🟢 ATAT Extension (our work)
│   ├── README.md               # ATAT-specific docs
│   ├── atat/                   # Core ATAT components
│   │   ├── importance_estimator.py
│   │   ├── adaptive_masking.py
│   │   ├── curriculum.py
│   │   └── uncertainty_sampler.py
│   ├── models/                 # ATAT-DiT model
│   ├── configs/atat/           # ATAT configs
│   ├── scripts/                # Training & evaluation
│   ├── utils/                  # Visualization tools
│   └── tests/                  # Unit tests
│
└── docs/                       # 📚 Documentation
    ├── INDEX.md                # Documentation index
    ├── RESTRUCTURING_GUIDE.md  # **START HERE** - Project organization
    ├── DATA_PATHS.md           # Dataset & storage guide
    ├── research_proposal.tex   # CVPR paper draft
    └── reports/                # Technical reports
        ├── GETTING_STARTED.md
        ├── TECHNICAL_REPORT.md
        ├── EXECUTIVE_SUMMARY.md
        └── ...
```

---

## 📖 Documentation

**New to the project? Start here:**

1. 📘 [**RESTRUCTURING_GUIDE.md**](docs/RESTRUCTURING_GUIDE.md) - Project organization and next steps
2. 🚀 [**GETTING_STARTED.md**](docs/reports/GETTING_STARTED.md) - Setup and first experiment
3. 📊 [**DATA_PATHS.md**](docs/DATA_PATHS.md) - Dataset locations and storage
4. 🔍 [**TECHNICAL_REPORT.md**](docs/reports/TECHNICAL_REPORT.md) - Implementation details
5. 📄 [**research_proposal.tex**](docs/research_proposal.tex) - CVPR paper draft

**Full documentation index**: [docs/INDEX.md](docs/INDEX.md)

---

## 🎓 Research

### Paper

> **Adaptive Time-Aware Token Masking for Masked Diffusion Language Models**  
> Draft available: [docs/research_proposal.tex](docs/research_proposal.tex)  
> Target: CVPR 2026

### Key Results (Expected)

| Metric | Baseline MDLM | ATAT | Improvement |
|--------|---------------|------|-------------|
| Perplexity (OWT) | 25.0 | **22.5** | **10%** ↓ |
| Convergence Steps | 100k | **80k** | **20%** faster |
| Inference Time | 1.0x | **0.7x** | **30%** faster |

### Components

1. **ImportanceEstimator** (2M params)
   - 2-layer transformer
   - Predicts token difficulty
   - Time-conditioned embeddings

2. **AdaptiveMaskingScheduler**
   - Temperature-controlled sampling
   - Importance-weighted masking
   - Position bias support

3. **CurriculumScheduler**
   - 3-stage progression (easy → medium → hard)
   - Dynamic difficulty adjustment
   - Performance-based adaptation

4. **UncertaintyGuidedSampler**
   - Entropy-based prioritization
   - Adaptive step scheduling
   - Early termination support

---

## 🔬 Experiments

### Datasets

- **OpenWebText** (primary): 8.5M docs, ~8B tokens
- **WikiText-103**: 100M tokens
- **Text8**: 100M characters
- **LM1B**: 1B words

All stored in `/media/scratch/adele/` (418GB total)

### Model Sizes

| Size | Params | Layers | Dim | Training Time |
|------|--------|--------|-----|---------------|
| Tiny | 25M | 8 | 256 | 12 hours |
| Small | 125M | 12 | 768 | 3 days |
| Medium | 350M | 24 | 1024 | 7 days |

### Running Experiments

```bash
# Ablation studies
python mdlm_atat/scripts/run_ablation.py

# Evaluation
python mdlm_atat/scripts/eval_atat.py \
  --checkpoint /media/scratch/adele/mdlm_fresh/checkpoints/model.ckpt

# Create sampling GIF
python mdlm_atat/scripts/create_sampling_gif.py \
  --checkpoint <path> \
  --prompt "The future of AI is"
```

---

## 💻 Development

### Project Status (December 2024)

- ✅ **Codebase restructured** - Clean, organized, production-ready
- ✅ **ATAT implementation complete** - All 4 components working
- ✅ **Testing infrastructure** - Unit tests, ablations ready
- ✅ **Research proposal written** - CVPR paper draft complete
- ⏳ **Training runs pending** - Ready to collect results
- ⏳ **Experimental validation** - Awaiting full training

### Next Steps

1. **Immediate**: Test run ATAT-Tiny (10k steps)
2. **This Week**: Full ATAT-Tiny training (100k steps)
3. **This Month**: ATAT-Small on OpenWebText, collect results
4. **Next Month**: Finalize paper with experimental results

See [docs/RESTRUCTURING_GUIDE.md](docs/RESTRUCTURING_GUIDE.md) for detailed roadmap.

### Contributing

This is a research project. For development:

1. Work in `mdlm_atat/` (never modify `mdlm/` directly)
2. Test with tiny config first
3. Document results in `docs/reports/`
4. Update paper in `docs/research_proposal.tex`

---

## 🔗 Related Work

- **MDLM** (NeurIPS 2024): [Paper](https://arxiv.org/abs/2406.07524) | [Original Code](https://github.com/kuleshov-group/mdlm)
- **SEDD** (2023): [Score Entropy Discrete Diffusion](https://arxiv.org/abs/2310.16834)
- **D3PM** (2021): [Structured Denoising Diffusion](https://arxiv.org/abs/2107.03006)

---

## 📊 Storage & Data

### Data Locations

All large files stored in scratch drive:

```
/media/scratch/adele/
├── mdlm_data_cache/    # 47GB  - HuggingFace cache
├── mdlm_fresh/         # 321GB - Outputs & checkpoints
└── datasets/           # 50GB  - Raw datasets
```

**Important**: Never commit datasets or checkpoints to git! Use `.gitignore`.

See [docs/DATA_PATHS.md](docs/DATA_PATHS.md) for complete storage guide.

---

## 📝 Citation

If you use this work, please cite:

```bibtex
@article{chinda2024atat,
  title={Adaptive Time-Aware Token Masking for Masked Diffusion Language Models},
  author={Chinda, Adele},
  year={2024},
  note={Research in progress}
}

@inproceedings{sahoo2024mdlm,
  title={Simple and Effective Masked Diffusion Language Models},
  author={Sahoo, Subham Sekhar and Arriola, Marianne and Schiff, Yair and 
          Gokaslan, Aaron and Marroquin, Edgar and Chiu, Justin T and 
          Rush, Alexander and Kuleshov, Volodymyr},
  booktitle={NeurIPS},
  year={2024}
}
```

---

## 📄 License

See LICENSE file for details.

---

## 👤 Contact

**Maintainer**: Adele Chinda  
**Last Updated**: December 3, 2024  
**Status**: Active development - restructured and ready for training runs

---

## 🔥 Quick Command Reference

```bash
# Setup
conda env create -f requirements.yaml
conda activate mdlm
pip install -e mdlm_atat/

# Train ATAT
python mdlm_atat/scripts/train_atat.py --config-name atat/tiny

# Evaluate
python mdlm_atat/scripts/eval_atat.py --checkpoint <path>

# Run tests
pytest mdlm_atat/tests/

# Check storage
du -sh /media/scratch/adele/*

# View logs
tensorboard --logdir /media/scratch/adele/mdlm_fresh/outputs
```

**For more commands**: See [docs/reports/QUICK_REFERENCE.md](docs/reports/QUICK_REFERENCE.md)

---

**🎯 Ready to start training? See [docs/RESTRUCTURING_GUIDE.md](docs/RESTRUCTURING_GUIDE.md) for next steps!**
