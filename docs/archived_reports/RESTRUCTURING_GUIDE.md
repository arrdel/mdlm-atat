# MDLM-ATAT Project Restructuring (December 2024)

## Overview
This document outlines the restructured codebase for optimal organization and workflow efficiency.

## Directory Structure

```
mdlm/
├── README.md                       # Main project README
├── requirements.yaml               # Conda environment
├── .gitignore                      # Git ignore patterns
│
├── docs/                          # 📚 All documentation (NEW)
│   ├── reports/                   # Research reports from mdlm_atat/reports
│   │   ├── INDEX.md              # Report index
│   │   ├── EXECUTIVE_SUMMARY.md
│   │   ├── TECHNICAL_REPORT.md
│   │   ├── PROJECT_SUMMARY.md
│   │   ├── GETTING_STARTED.md
│   │   └── ...
│   └── research_proposal.tex      # CVPR paper draft
│
├── mdlm/                          # 🔵 Base MDLM implementation
│   ├── main.py                    # Training entry point
│   ├── diffusion.py               # Core diffusion (SUBS, D3PM, SEDD)
│   ├── dataloader.py              # Dataset loading
│   ├── noise_schedule.py          # Noise schedules
│   ├── utils.py                   # Utilities
│   ├── configs/                   # Hydra configs
│   │   ├── config.yaml           # Base config
│   │   ├── model/                # Model configs (small, medium, large)
│   │   ├── data/                 # Dataset configs
│   │   ├── noise/                # Noise schedules
│   │   └── lr_scheduler/         # Learning rate schedules
│   ├── models/                    # Model architectures
│   │   ├── dit.py                # Diffusion Transformer (DiT)
│   │   ├── dimamba.py            # DiMamba (Mamba-based)
│   │   ├── autoregressive.py     # AR baseline
│   │   └── ema.py                # EMA wrapper
│   └── scripts/                   # Training scripts
│       ├── train_owt_mdlm.sh     # OpenWebText MDLM
│       ├── train_lm1b_d3pm.sh    # LM1B D3PM
│       └── eval_owt_*.sh         # Evaluation scripts
│
├── mdlm_atat/                     # 🟢 ATAT Extension (our contribution)
│   ├── README.md                  # ATAT-specific README
│   ├── setup.py                   # Package setup
│   ├── requirements.txt           # Additional dependencies
│   ├── __init__.py               # Package initialization
│   │
│   ├── atat/                      # Core ATAT components
│   │   ├── __init__.py
│   │   ├── importance_estimator.py   # Token importance network
│   │   ├── adaptive_masking.py       # Adaptive masking scheduler
│   │   ├── curriculum.py             # Curriculum learning
│   │   └── uncertainty_sampler.py    # Uncertainty-guided sampling
│   │
│   ├── models/                    # ATAT-enhanced models
│   │   ├── __init__.py
│   │   └── atat_dit.py           # ATATDiT (DiT + ATAT)
│   │
│   ├── configs/                   # ATAT configurations
│   │   ├── atat/
│   │   │   ├── base_config.yaml  # Base ATAT config
│   │   │   ├── tiny.yaml         # Tiny model (25M params)
│   │   │   ├── small.yaml        # Small model (125M params)
│   │   │   └── ablations/        # Ablation configs
│   │   ├── model/
│   │   └── lr_scheduler/
│   │
│   ├── scripts/                   # ATAT training & evaluation
│   │   ├── train_atat.py         # Main training script
│   │   ├── eval_atat.py          # Evaluation script
│   │   ├── train_pipeline.py     # Full training pipeline
│   │   ├── run_ablation.py       # Ablation studies
│   │   ├── download_datasets.py  # Dataset downloader
│   │   ├── generate_tiny_dataset.py  # Tiny dataset for testing
│   │   ├── create_sampling_gif.py    # GIF visualization
│   │   └── run_full_pipeline.sh      # Full pipeline script
│   │
│   ├── utils/                     # Utilities
│   │   ├── __init__.py
│   │   ├── visualization.py      # Visualization tools
│   │   └── gif_visualization.py  # GIF generation
│   │
│   └── tests/                     # Unit tests
│       ├── conftest.py
│       ├── test_atat_components.py
│       └── test_atat_models.py
│
├── .archive/                      # 🗄️ Archived/deprecated files
│   └── (old test scripts, etc.)
│
└── /media/scratch/adele/          # 💾 Data storage (external)
    ├── mdlm_data_cache/          # Dataset cache (47GB)
    ├── mdlm_fresh/               # Fresh outputs (321GB)
    └── datasets/                 # Raw datasets (50GB)
```

## Key Improvements

### 1. Clean Separation
- **mdlm/**: Base MDLM implementation (untouched, stable)
- **mdlm_atat/**: ATAT extension (our innovation, active development)
- **docs/**: All documentation in one place

### 2. Removed Clutter
- ✅ Deleted all `__pycache__/` directories
- ✅ Removed `.pyc`, `.pyo` temporary files
- ✅ Consolidated reports into `docs/reports/`
- ✅ Added comprehensive `.gitignore`

### 3. Configuration Organization
- Shared configs in `mdlm/configs/` for base functionality
- ATAT-specific configs in `mdlm_atat/configs/atat/`
- All paths point to `/media/scratch/adele/` for data

### 4. Dataset Management
- **Location**: `/media/scratch/adele/`
  - `mdlm_data_cache/`: 47GB (HuggingFace cache)
  - `mdlm_fresh/`: 321GB (outputs, checkpoints)
  - `datasets/`: 50GB (raw datasets)
- All config files updated to point to correct paths
- Training outputs saved to scratch, not cluttering repo

## Quick Reference

### Training ATAT Models
```bash
# Tiny model (fast testing)
python mdlm_atat/scripts/train_atat.py --config-name atat/tiny

# Small model (production)
python mdlm_atat/scripts/train_atat.py --config-name atat/small

# With custom config
python mdlm_atat/scripts/train_atat.py \
  --config-name atat/small \
  data.train=openwebtext \
  trainer.max_steps=100000
```

### Evaluation
```bash
# Evaluate trained model
python mdlm_atat/scripts/eval_atat.py \
  --checkpoint /media/scratch/adele/mdlm_fresh/checkpoints/model.ckpt

# Run ablation studies
python mdlm_atat/scripts/run_ablation.py
```

### Dataset Setup
```bash
# Download datasets to scratch
python mdlm_atat/scripts/download_datasets.py \
  --output-dir /media/scratch/adele/datasets \
  --datasets openwebtext wikitext103

# Generate tiny dataset for testing
python mdlm_atat/scripts/generate_tiny_dataset.py
```

## What Was Removed/Archived

### Deleted
- All `__pycache__/` directories (27 total)
- Compiled Python files (`.pyc`, `.pyo`)
- Temporary editor files (`*~`, `.DS_Store`)

### Archived to `.archive/`
- Old test scripts (test_importance_estimator.py, quick_importance_test.py)
- Redundant/experimental code

### Consolidated
- 9 report files from `mdlm_atat/reports/` → `docs/reports/`
- Documentation scattered across README files → organized in `docs/`

## Next Steps for Development

### Immediate (Ready to Run)
1. **Start training runs**:
   ```bash
   python mdlm_atat/scripts/train_atat.py --config-name atat/tiny --max-steps 10000
   ```

2. **Validate installation**:
   ```bash
   python -c "import mdlm_atat; print('ATAT imported successfully')"
   ```

3. **Test dataset access**:
   ```bash
   ls -lh /media/scratch/adele/mdlm_data_cache/
   ```

### Short-term (Next Week)
1. Run ATAT-Tiny training (100k steps, ~12 hours)
2. Generate baseline MDLM results for comparison
3. Run ablation studies (no-importance, no-curriculum, etc.)
4. Create visualizations for paper

### Medium-term (Next Month)
1. Full ATAT-Small training on OpenWebText
2. Complete experimental results for research proposal
3. Generate figures and update `docs/research_proposal.tex`
4. Submit paper to CVPR 2026

## Dependencies

All dependencies managed via conda:
```bash
conda env create -f requirements.yaml
conda activate mdlm
```

Additional ATAT dependencies:
```bash
pip install -e mdlm_atat/
```

## Data Paths Configuration

All configs use scratch paths:
- **Data cache**: `/media/scratch/adele/mdlm_data_cache/`
- **Outputs**: `/media/scratch/adele/mdlm_fresh/outputs/`
- **Checkpoints**: `/media/scratch/adele/mdlm_fresh/checkpoints/`

Update these in config files if your scratch location changes.

## Development Workflow

1. **Make changes** in `mdlm_atat/` (never touch `mdlm/` unless fixing bugs)
2. **Test locally** with tiny dataset first
3. **Run on scratch** for full experiments
4. **Document results** in `docs/reports/`
5. **Update paper** in `docs/research_proposal.tex`

## Important Notes

- ⚠️ **Never commit large files**: Outputs, checkpoints, datasets stay in scratch
- ⚠️ **Use git LFS** for any binary files in repo if needed
- ⚠️ **Test first**: Always use tiny config before full runs
- ⚠️ **Monitor scratch space**: 418GB used of available capacity

---

**Last Updated**: December 3, 2024  
**Maintainer**: Adele Chinda  
**Status**: Restructured and ready for training runs
