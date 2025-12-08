# Dataset and Storage Configuration

## Storage Overview

All large files (datasets, checkpoints, outputs) are stored on the scratch drive to keep the repository clean and under version control size limits.

### Current Storage Usage

```bash
Location: /media/scratch/adele/

mdlm_data_cache/    47GB   # HuggingFace dataset cache
mdlm_fresh/        321GB   # Training outputs & checkpoints  
datasets/           50GB   # Raw datasets
-----------------------------------
Total:             418GB
```

## Data Locations

### 1. Dataset Cache (`mdlm_data_cache/`)
**Path**: `/media/scratch/adele/mdlm_data_cache/`  
**Size**: 47GB  
**Contents**: HuggingFace `datasets` library cache
- Pre-processed tokenized data
- Downloaded datasets from HuggingFace Hub
- Cached transformations

**Config Reference**:
```yaml
data:
  cache_dir: /media/scratch/adele/mdlm_data_cache
```

### 2. Training Outputs (`mdlm_fresh/`)
**Path**: `/media/scratch/adele/mdlm_fresh/`  
**Size**: 321GB  
**Contents**: 
- `outputs/`: Hydra output directories
- `checkpoints/`: Model checkpoints (`.ckpt` files)
- `logs/`: Training logs
- WandB runs

**Directory Structure**:
```
mdlm_fresh/
├── outputs/
│   ├── openwebtext/
│   ├── wikitext103/
│   └── <dataset>/<date>/<time>/
│       ├── .hydra/
│       ├── checkpoints/
│       └── logs/
├── data_cache/      # Alternative cache location
└── checkpoints/     # Shared checkpoint directory
```

**Config Reference**:
```yaml
hydra:
  run:
    dir: /media/scratch/adele/mdlm_fresh/outputs/${data.train}/${now:%Y.%m.%d}/${now:%H%M%S}
```

### 3. Raw Datasets (`datasets/`)
**Path**: `/media/scratch/adele/datasets/`  
**Size**: 50GB  
**Contents**: Original downloaded datasets
- OpenWebText raw files
- WikiText-103 raw files
- Text8 raw files
- Custom datasets

## Configuration Files

### ATAT Configs

All ATAT configs in `mdlm_atat/configs/atat/*.yaml` use scratch paths:

**Base Config** (`mdlm_atat/configs/atat/base_config.yaml`):
```yaml
data:
  cache_dir: /media/scratch/adele/mdlm_data_cache

hydra:
  run:
    dir: /media/scratch/adele/mdlm_fresh/outputs/${data.train}/${now:%Y.%m.%d}/${now:%H%M%S}
```

**Tiny Config** (`mdlm_atat/configs/atat/tiny.yaml`):
```yaml
data:
  cache_dir: /media/scratch/adele/mdlm_fresh/data_cache

hydra:
  run:
    dir: /media/scratch/adele/mdlm_fresh/outputs/${data.train}/${now:%Y.%m.%d}/${now:%H%M%S}
```

### Baseline MDLM Configs

Original MDLM configs in `mdlm/configs/` may have different paths. Update as needed:

```yaml
# mdlm/configs/config.yaml (update if needed)
defaults:
  - _self_
  - data: openwebtext-split

# Ensure cache points to scratch
data:
  cache_dir: /media/scratch/adele/mdlm_data_cache
```

## Dataset Downloads

### Available Datasets

The following datasets are configured and ready to use:

1. **OpenWebText** (Primary)
   - Config: `data=openwebtext-split` or `data=openwebtext-streaming`
   - Size: ~40GB
   - Docs: 8.5M documents
   - Tokens: ~8B tokens (GPT-2)

2. **WikiText-103**
   - Config: `data=wikitext103`
   - Size: ~500MB
   - Tokens: ~100M tokens

3. **Text8**
   - Config: `data=text8`
   - Size: ~100MB
   - Chars: 100M characters

4. **LM1B** (One Billion Word)
   - Config: `data=lm1b` or `data=lm1b-streaming`
   - Size: ~4GB
   - Words: 1B words

### Download Script

Use the download script to fetch datasets:

```bash
python mdlm_atat/scripts/download_datasets.py \
  --output-dir /media/scratch/adele/datasets \
  --datasets openwebtext wikitext103 text8
```

## Checkpoint Management

### Saving Checkpoints

Checkpoints are automatically saved during training:

```yaml
checkpointing:
  save_dir: /media/scratch/adele/mdlm_fresh/checkpoints/${data.train}
  save_top_k: 3
  monitor: val/loss
  mode: min
  every_n_train_steps: 5000
```

### Loading Checkpoints

To resume training or evaluate:

```bash
python mdlm_atat/scripts/train_atat.py \
  --config-name atat/small \
  checkpointing.resume_from=/media/scratch/adele/mdlm_fresh/checkpoints/epoch=10.ckpt
```

## Storage Best Practices

### DO ✅
- Store all datasets in `/media/scratch/adele/datasets/`
- Save checkpoints to `/media/scratch/adele/mdlm_fresh/checkpoints/`
- Use HuggingFace cache at `/media/scratch/adele/mdlm_data_cache/`
- Keep training outputs in `/media/scratch/adele/mdlm_fresh/outputs/`

### DON'T ❌
- Don't save checkpoints in the git repository
- Don't commit dataset files
- Don't store large outputs in home directory
- Don't use relative paths in configs

### Cleanup Commands

When scratch is getting full:

```bash
# Remove old training outputs (keep only recent)
find /media/scratch/adele/mdlm_fresh/outputs -type d -mtime +30 -exec rm -rf {} +

# Remove old checkpoints (keep only top-k)
# (Handled automatically by PyTorch Lightning)

# Clear HuggingFace cache (if corrupted)
rm -rf /media/scratch/adele/mdlm_data_cache/*
# Then re-download datasets
```

## Monitoring Storage

Check storage usage:

```bash
# Overall scratch usage
du -sh /media/scratch/adele/*

# Dataset cache size
du -sh /media/scratch/adele/mdlm_data_cache

# Training outputs size
du -sh /media/scratch/adele/mdlm_fresh

# Checkpoints size
du -sh /media/scratch/adele/mdlm_fresh/checkpoints

# Recent outputs
ls -lht /media/scratch/adele/mdlm_fresh/outputs/ | head -20
```

## Backup Strategy

### Critical Files (Backup to Cloud/External)
- Final trained checkpoints
- Best model weights
- Training logs with WandB links
- Configuration files (already in git)

### Temporary Files (Can Delete)
- Intermediate checkpoints
- Old training runs
- Cached datasets (can re-download)

## Path Update Checklist

If you need to change the scratch location:

1. ✅ Update `mdlm_atat/configs/atat/base_config.yaml`
2. ✅ Update `mdlm_atat/configs/atat/tiny.yaml`
3. ✅ Update `mdlm_atat/configs/atat/small.yaml`
4. ✅ Update `mdlm/configs/config.yaml` (if using baseline)
5. ✅ Update this file (`docs/DATA_PATHS.md`)
6. ✅ Update `docs/RESTRUCTURING_GUIDE.md`
7. ✅ Run test to verify: `python -c "from hydra import compose, initialize; ..."`

## Environment Variables

Set these for convenience:

```bash
# Add to ~/.bashrc or ~/.zshrc
export MDLM_SCRATCH="/media/scratch/adele"
export MDLM_DATA_CACHE="$MDLM_SCRATCH/mdlm_data_cache"
export MDLM_OUTPUTS="$MDLM_SCRATCH/mdlm_fresh/outputs"
export MDLM_CHECKPOINTS="$MDLM_SCRATCH/mdlm_fresh/checkpoints"

# HuggingFace cache
export HF_HOME="$MDLM_DATA_CACHE"
export TRANSFORMERS_CACHE="$MDLM_DATA_CACHE/transformers"
export HF_DATASETS_CACHE="$MDLM_DATA_CACHE/datasets"
```

Then use in code:
```python
import os
cache_dir = os.environ.get('MDLM_DATA_CACHE', '/media/scratch/adele/mdlm_data_cache')
```

---

**Last Updated**: December 3, 2024  
**Total Storage**: 418GB / ? available  
**Status**: Paths verified and configured
