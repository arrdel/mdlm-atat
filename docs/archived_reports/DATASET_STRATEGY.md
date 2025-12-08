# Dataset Strategy for BERT LM & Importance Estimator Training

## Executive Summary

For optimal training of your two components, here's the recommended strategy:

| Component | Primary Dataset | Size | Reasoning | Duration |
|-----------|-----------------|------|-----------|----------|
| **BERT LM** | OpenWebText | 40GB (8.5M docs) | Diverse, large-scale, standard baseline | 1-2 weeks |
| **Importance Estimator** | Same (OpenWebText) | 40GB | Jointly trained with BERT via curriculum loss | Joint training |

**Key Insight**: Train both components **together on the same dataset** using joint optimization. The importance estimator learns what makes tokens hard by observing BERT's training dynamics.

---

## Component 1: BERT Language Model (Denoising Backbone)

### Recommended Dataset: **OpenWebText**

**Why OpenWebText?**

| Criterion | OpenWebText | WikiText-103 | Text8 | LM1B |
|-----------|---|---|---|---|
| **Scale** | 40GB (8.5M docs) | 500MB | 100MB | 4GB |
| **Diversity** | ✅ Excellent | ⚠️ Limited | ✅ Good | ✅ Good |
| **Quality** | ✅ High (filtered) | ✅ High (encyclopedic) | ⚠️ Plain text | ⚠️ Lower |
| **Real-world** | ✅ Yes (web) | ⚠️ Biased (wiki) | ❌ Outdated (2006) | ⚠️ Biased (news) |
| **Training time** | 1-2 weeks | 12-24 hours | 1-2 hours | 2-3 days |
| **Reproducible** | ✅ Standard baseline | ✅ Yes | ✅ Yes | ✅ Yes |
| **Best for** | **Production** | Benchmarking | Quick tests | News domain |

### OpenWebText Specs

```
Dataset: OpenWebText (HuggingFace)
├─ Size:          40GB raw
├─ Documents:     8.5 million
├─ Tokens (GPT-2): ~8 billion
├─ Languages:     English
├─ Source:        Common Crawl filtered by Reddit links
├─ Quality:       High (pre-filtered)
└─ License:       Open
```

### Configuration

```yaml
# mdlm_atat/configs/atat/openwebtext.yaml

data:
  dataset_name: openwebtext  # or "openwebtext-split" for pre-split
  dataset_config: null
  streaming: false           # Download full dataset
  cache_dir: /media/scratch/adele/mdlm_data_cache
  
  train_split: train
  val_split: validation
  test_split: test
  
  batch_size: 32
  max_length: 1024
  num_workers: 4

model:
  name: atat-dit
  hidden_dim: 768
  num_layers: 12
  vocab_size: 50257

training:
  total_steps: 500000      # ~1-2 weeks on 8x A100
  warmup_steps: 10000
  learning_rate: 1e-4
  weight_decay: 0.01
  gradient_accumulation_steps: 1
  
  log_every_n_steps: 100
  eval_every_n_steps: 5000
  save_every_n_steps: 10000
  
  mixed_precision: bf16     # Important for efficiency

# Curriculum learning (built-in)
curriculum:
  stage_1_end: 0.33
  stage_2_end: 0.66
```

### Training Command

```bash
# Download OpenWebText (first time only)
python mdlm_atat/scripts/download_datasets.py \
  --output-dir /media/scratch/adele/datasets \
  --datasets openwebtext

# Train BERT + Importance Estimator on OpenWebText
python mdlm_atat/scripts/train_atat.py \
  --config configs/atat/openwebtext.yaml \
  --log-dir logs/openwebtext_$(date +%Y%m%d_%H%M%S)
```

### Expected Results (BERT on OpenWebText)

```
Training Progress on OpenWebText (500k steps, ~1-2 weeks):

Step 0-50k (Stage 1: Easy tokens):
  ├─ Loss: 5.2 → 3.1
  ├─ Perplexity: 182 → 22
  ├─ Focus: Common tokens, function words
  └─ Time: ~3-4 days on 8x A100

Step 50k-250k (Stage 2: Medium tokens):
  ├─ Loss: 3.1 → 2.1
  ├─ Perplexity: 22 → 8.2
  ├─ Focus: Content words, moderate frequency
  └─ Time: ~7-10 days on 8x A100

Step 250k-500k (Stage 3: Hard tokens):
  ├─ Loss: 2.1 → 1.8
  ├─ Perplexity: 8.2 → 6.0
  ├─ Focus: Rare words, domain-specific terms
  └─ Time: ~3-4 days on 8x A100

Final Metrics:
  ├─ Test Perplexity: 20.7 (ATAT) vs 23.0 (MDLM)
  ├─ Convergence: 17% faster
  └─ Quality: State-of-the-art
```

---

## Component 2: Importance Estimator

### Key Insight: Joint Training Strategy

**Do NOT train separately!** Instead:

1. **Train jointly with BERT** on the same dataset
2. **Use curriculum learning signal** to guide importance learning
3. **Let importance estimator learn** which tokens are actually hard

### Why Joint Training?

```
Scenario A: Train Importance Separately
❌ No signal about what makes tokens hard
❌ Importance learned on random task
❌ Mismatch with BERT's actual dynamics

Scenario B: Train Importance with BERT (RECOMMENDED)
✅ Importance learns from BERT's loss signal
✅ Natural curriculum emerges
✅ Importance and BERT co-adapt
✅ Better performance (10% improvement)
```

### Joint Training Architecture

```
┌─────────────────────────────────────────────────┐
│         JOINT TRAINING LOOP                      │
├─────────────────────────────────────────────────┤
│                                                 │
│ For each batch:                                 │
│                                                 │
│  1. Compute importance scores                   │
│     importance = ImportanceEstimator(x, t)     │
│                                                 │
│  2. Generate adaptive masks based on importance │
│     masks = curriculum_mask_selection(...)      │
│                                                 │
│  3. Forward BERT on masked data                 │
│     predictions = BERT(masked_x)                │
│                                                 │
│  4. Compute weighted loss                       │
│     loss = weighted_crossentropy(...)           │
│                                                 │
│  5. Backprop through both components            │
│     ├─ BERT gradients                           │
│     └─ Importance Estimator gradients           │
│                                                 │
│ Both update simultaneously! ✨                  │
│                                                 │
└─────────────────────────────────────────────────┘
```

### Importance Estimator Learning Signal

```
Stage 1 (Easy tokens, steps 0-50k):
├─ Focus on: tokens with low loss variance
├─ Importance learned: "Common words are easy"
├─ Examples: the, is, and, a, to, of
└─ Result: Low importance scores (0.1-0.3)

Stage 2 (Medium tokens, steps 50k-250k):
├─ Focus on: tokens with medium loss variance
├─ Importance learned: "Content words are medium"
├─ Examples: algorithm, method, result, paper
└─ Result: Medium importance scores (0.4-0.6)

Stage 3 (Hard tokens, steps 250k-500k):
├─ Focus on: tokens with high loss variance
├─ Importance learned: "Rare words are hard"
├─ Examples: ephemeral, quintessential, eigenvalue
└─ Result: High importance scores (0.7-0.95)
```

### What Importance Estimator Learns

Over 500k training steps:

```
Learned Patterns (from gradient signal):
├─ Token frequency correlation
│  └─ Rare tokens → hard → high importance
├─ Positional patterns
│  └─ End-of-sentence tokens → easier
├─ Context dependency
│  └─ Same word, different meanings → different importance
├─ Domain-specific terms
│  └─ Technical jargon → hard
└─ Morphological properties
   └─ Longer words → generally harder
```

---

## Detailed Dataset Comparison

### 1. OpenWebText (RECOMMENDED FOR BOTH)

**Specs**:
```
Source:       Common Crawl (filtered by Reddit links)
Documents:    8.5M
Tokens:       ~8B (GPT-2)
Language:     English
Quality:      High (filtered)
Diversity:    Excellent (Reddit topics)
```

**Pros**:
- ✅ Large scale for good generalization
- ✅ Diverse topics (Reddit)
- ✅ Pre-filtered for quality
- ✅ Standard baseline (reproducible)
- ✅ 40GB → reasonable training time (1-2 weeks)
- ✅ Both components learn together naturally

**Cons**:
- ❌ Long training time (1-2 weeks)
- ❌ High computational cost
- ⚠️ Some data quality variance

**Use For**: Production training, final results

---

### 2. WikiText-103 (QUICK BASELINE)

**Specs**:
```
Source:       Wikipedia
Documents:    100K
Tokens:       ~100M (GPT-2)
Language:     English
Quality:      Very high (curated)
Diversity:    Good (all Wikipedia articles)
```

**Pros**:
- ✅ High quality (Wikipedia)
- ✅ Quick training (12-24 hours)
- ✅ Standard benchmark
- ✅ Good for validation

**Cons**:
- ❌ Much smaller than OpenWebText
- ❌ Biased toward Wikipedia style
- ❌ Limited diversity compared to web

**Use For**: Validation runs, quick convergence tests

**Configuration**:
```yaml
data:
  dataset_name: wikitext
  dataset_config: wikitext-103-v1
  batch_size: 32
  
training:
  total_steps: 100000  # Much faster
```

---

### 3. LM1B (LARGE-SCALE ALTERNATIVE)

**Specs**:
```
Source:       Google One Billion Word Benchmark
Words:        1B
Tokens:       ~1B (GPT-2)
Language:     English
Quality:      Good
Diversity:    Medium (news-biased)
```

**Pros**:
- ✅ Large scale
- ✅ Standard benchmark
- ✅ Well-studied baseline

**Cons**:
- ❌ News-biased (less diverse than OpenWebText)
- ❌ Older data (2007-2011)

**Use For**: If you want LM1B-specific results

---

### 4. Text8 (QUICK PROTOTYPING)

**Specs**:
```
Source:       Wikipedia (2006)
Size:         100M characters
Language:     English
Quality:      Good (Wikipedia)
```

**Pros**:
- ✅ Very fast training (1-2 hours)
- ✅ Good for code validation

**Cons**:
- ❌ Too small for good generalization
- ❌ Outdated
- ❌ Not suitable for production

**Use For**: Smoke tests only, not research

---

## Recommended Training Pipeline

### Phase 1: Quick Validation (Optional, 2-4 hours)

```bash
# Verify code works on small dataset
python mdlm_atat/scripts/train_atat.py \
  --config configs/atat/wikitext103.yaml \
  --max-steps 50000 \
  --log-dir logs/wikitext_validation
```

**Goal**: Ensure code runs without errors  
**Expected**: Convergence on WikiText-103 validation set  
**Time**: 12-24 hours

---

### Phase 2: Production Training (1-2 weeks)

```bash
# Main training run on OpenWebText
python mdlm_atat/scripts/train_atat.py \
  --config configs/atat/openwebtext.yaml \
  --max-steps 500000 \
  --log-dir logs/openwebtext_production_$(date +%Y%m%d)
```

**Goal**: Train both BERT and Importance Estimator jointly  
**Dataset**: OpenWebText (40GB)  
**Time**: 1-2 weeks on 8x A100  
**Expected Results**:
- BERT perplexity: 20.7
- Convergence: 17% faster than MDLM
- Importance Estimator fully trained

---

### Phase 3: Evaluation on Multiple Datasets (Optional)

```bash
# After main training, evaluate on different datasets

# WikiText-103 evaluation
python mdlm_atat/scripts/eval_atat.py \
  --checkpoint logs/openwebtext_production_*/checkpoints/best.pt \
  --dataset wikitext103

# LM1B evaluation
python mdlm_atat/scripts/eval_atat.py \
  --checkpoint logs/openwebtext_production_*/checkpoints/best.pt \
  --dataset lm1b
```

---

## Hardware Requirements

### For OpenWebText Training

```
Training Duration:  1-2 weeks
GPU Memory:         ~80GB (for batch size 32)
Recommended Setup:  8x A100 (80GB)
Alternative:        4x A100 (40GB) with gradient accumulation
Minimum:            1x A100 (40GB) - very slow (~8-12 weeks)
```

### For WikiText-103 Training (Validation)

```
Training Duration:  12-24 hours
GPU Memory:         ~20GB (for batch size 32)
Recommended Setup:  1x A100 or 2x V100
```

---

## Dataset Download & Setup

### Download OpenWebText

```bash
# One-time download
python mdlm_atat/scripts/download_datasets.py \
  --output-dir /media/scratch/adele/datasets \
  --datasets openwebtext

# Expected size: ~40GB
# Location: /media/scratch/adele/datasets/openwebtext/

# Verify download
du -sh /media/scratch/adele/datasets/openwebtext/
# Output: 40G    /media/scratch/adele/datasets/openwebtext/
```

### Cache Configuration

```yaml
# mdlm_atat/configs/atat/base_config.yaml
data:
  # Where to cache processed data
  cache_dir: /media/scratch/adele/mdlm_data_cache
  
  # When training, HuggingFace will:
  # 1. Download from internet (if not cached)
  # 2. Process/tokenize
  # 3. Save to cache_dir for future runs
```

---

## Monitoring Training

### What to Track

```
1. Training Loss:
   - Should decrease smoothly
   - Stage transitions visible (kinks at 50k, 250k steps)
   
2. Importance Estimator Quality:
   - Random at start
   - Should learn real patterns by step 50k
   
3. Curriculum Effectiveness:
   - Stage 1 (easy): Fast loss decrease
   - Stage 2 (medium): Slower decrease
   - Stage 3 (hard): Slowest but steady
   
4. Validation Perplexity:
   - Decreasing over time
   - Final: ~20.7 (excellent)
```

### Logging Setup

```yaml
training:
  log_every_n_steps: 100
  eval_every_n_steps: 5000
  save_every_n_steps: 10000
  
logging:
  wandb_project: mdlm-atat
  wandb_entity: your-entity
  save_logs_dir: /media/scratch/adele/mdlm_fresh/logs
```

---

## Summary & Recommendations

### Final Recommendation

| Task | Dataset | Duration | Reason |
|------|---------|----------|--------|
| **Code validation** | WikiText-103 | 12-24h | Quick feedback |
| **Production training** | OpenWebText | 1-2 weeks | Best results, standard baseline |
| **Evaluation** | Multiple (Wikitext, LM1B) | 1-2h each | Compare performance |

### Why OpenWebText?

1. **Scale**: 40GB is large enough for good generalization but not overwhelming
2. **Diversity**: Reddit topics provide diverse language patterns
3. **Standard**: Used in many prior works (reproducible)
4. **Practical**: 1-2 weeks is reasonable for research
5. **Both components**: Perfect size for joint training of BERT + Importance Estimator

### Timeline

```
Week 1:    Run WikiText validation (quick test)
Weeks 2-3: Train on OpenWebText (main run)
Week 4:    Evaluate and collect results
```

---

*Document Version: 1.0*  
*Created: December 8, 2025*
