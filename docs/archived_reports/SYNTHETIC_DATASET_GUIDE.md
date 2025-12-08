# Synthetic Dataset Guide

## Overview

The **synthetic dataset** is a small, lightweight test dataset created for **quick ATAT model prototyping and debugging** without needing large real datasets.

### Purpose

| Use Case | Why Synthetic | Alternative |
|----------|---|---|
| **Quick testing** (5-10 min) | Fast generation, instant use | Download real dataset (hours) |
| **Debugging code** | Works offline, reproducible | Need internet + storage |
| **Prototyping** | Small memory footprint | Run out of memory on real data |
| **CI/CD pipelines** | Stable, deterministic | Flaky, bandwidth-limited |
| **Learning** | Easy to inspect and modify | Complex real data |

---

## What It Contains

### Dataset Composition

```
Synthetic Tiny Dataset
├── Training set:   1,000 samples
├── Validation set:   100 samples
├── Test set:          50 samples
└── Total:          1,150 samples
```

Each sample:
- **Format**: GPT-2 tokenized text
- **Length**: Variable (50-200 tokens)
- **Content**: Templated text about AI/ML (deterministic, repeatable)

### Example Generated Text

```
The quick brown fox jumps over the lazy dog. 42.
In a world of constant change, adaptability is key. 89.
Machine learning models learn patterns from data. 156.
Natural language processing enables computers to understand text. 203.
...
```

The templates are:
- "The quick brown fox jumps over the lazy dog."
- "In a world of constant change, adaptability is key."
- "Machine learning models learn patterns from data."
- "Natural language processing enables computers to understand text."
- "Deep learning has revolutionized artificial intelligence."
- "Transformers have become the dominant architecture for NLP."
- "Diffusion models generate high-quality samples iteratively."
- "Language models can generate coherent and contextual text."
- "Training neural networks requires large amounts of data."
- "The future of AI holds many exciting possibilities."

### Data Format

**Tokenization**: GPT-2 tokenizer
```python
# Each sample is tokenized and padded to max_length=1024
input_ids: [BOS_ID, tok1, tok2, ..., PAD_ID, PAD_ID, ...]
attention_mask: [1, 1, 1, ..., 0, 0, ...]
```

**Storage Format**: HuggingFace `datasets` library format
- Binary format optimized for fast loading
- Supports streaming and efficient caching
- Compatible with PyTorch DataLoader

---

## How to Generate It

### Generate Fresh Synthetic Dataset

```bash
# From project root
python mdlm_atat/scripts/generate_tiny_dataset.py \
    --cache-dir /media/scratch/adele/mdlm_fresh/data_cache \
    --train-samples 1000 \
    --val-samples 100 \
    --test-samples 50 \
    --max-length 1024
```

### Output Files

```
/media/scratch/adele/mdlm_fresh/data_cache/
├── synthetic_tiny_train_bs1024_wrapped.dat/      (1,000 samples)
├── synthetic_tiny_validation_bs1024_wrapped.dat/ (100 samples)
└── synthetic_tiny_test_bs1024_wrapped.dat/       (50 samples)
```

Each is a HuggingFace Dataset saved to disk.

### Command Line Arguments

| Argument | Default | Purpose |
|----------|---------|---------|
| `--cache-dir` | `/media/scratch/adele/mdlm_fresh/data_cache` | Where to save datasets |
| `--train-samples` | 1000 | Number of training samples |
| `--val-samples` | 100 | Number of validation samples |
| `--test-samples` | 50 | Number of test samples |
| `--max-length` | 1024 | Sequence length (padding/truncation) |

---

## How to Use It in Training

### Configuration File Example

Create or use `mdlm_atat/configs/atat/tiny.yaml`:

```yaml
# Training configuration for tiny synthetic dataset

model:
  name: atat-dit
  hidden_dim: 256
  num_layers: 6
  vocab_size: 50257

data:
  # Point to synthetic dataset
  train_path: /media/scratch/adele/mdlm_fresh/data_cache/synthetic_tiny_train_bs1024_wrapped.dat
  val_path: /media/scratch/adele/mdlm_fresh/data_cache/synthetic_tiny_validation_bs1024_wrapped.dat
  test_path: /media/scratch/adele/mdlm_fresh/data_cache/synthetic_tiny_test_bs1024_wrapped.dat
  
  batch_size: 4        # Small batch for quick iteration
  num_workers: 0       # No multi-processing needed for synthetic data
  max_length: 1024

training:
  total_steps: 1000    # Quick training (1000 steps = ~5 min)
  warmup_steps: 100
  learning_rate: 1e-4
  gradient_accumulation_steps: 1
  
  # Logging
  log_every_n_steps: 10
  eval_every_n_steps: 100
  save_every_n_steps: 500

curriculum:
  stage_1_end: 0.33
  stage_2_end: 0.66
  importance_ranges:
    easy: [0.0, 0.4]
    medium: [0.3, 0.7]
    hard: [0.6, 1.0]
```

### Training Script

```bash
# Quick test (1000 steps, ~5 minutes)
python mdlm_atat/scripts/train_atat.py \
    --config configs/atat/tiny.yaml \
    --max-steps 1000 \
    --log-dir logs/tiny_test_$(date +%Y%m%d_%H%M%S)

# Medium test (10k steps, ~1 hour)
python mdlm_atat/scripts/train_atat.py \
    --config configs/atat/tiny.yaml \
    --max-steps 10000 \
    --log-dir logs/tiny_medium_$(date +%Y%m%d_%H%M%S)
```

### Python Usage

```python
from datasets import load_from_disk
import torch
from torch.utils.data import DataLoader

# Load synthetic dataset
train_dataset = load_from_disk(
    '/media/scratch/adele/mdlm_fresh/data_cache/synthetic_tiny_train_bs1024_wrapped.dat'
)

# Create DataLoader
dataloader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=0
)

# Training loop
for batch in dataloader:
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    
    # Forward pass
    outputs = model(input_ids, attention_mask)
    loss = outputs.loss
    
    # Backward
    loss.backward()
    optimizer.step()
```

---

## Why Use It?

### Advantages

✅ **Speed**
- Generate in seconds
- Train on ~1000 steps in 5 minutes
- Complete feedback loop in < 10 minutes

✅ **Reproducibility**
- Same data every time (deterministic)
- No download randomness
- Easy to debug

✅ **Memory Efficiency**
- 1,150 samples fit in ~500MB
- No OOM on laptops or small GPUs
- Fast iteration cycles

✅ **Offline Development**
- No internet required
- No bandwidth bottleneck
- Works anywhere

✅ **Testing Infrastructure**
- Validate code changes quickly
- Test edge cases easily
- Smoke-test before real runs

### Trade-offs

❌ **Limited Diversity**
- Only 10 templates
- No real linguistic variety
- Cannot assess generalization

❌ **Unrealistic Patterns**
- Repetitive structure
- Easy tokens dominate
- Different from real language

❌ **Small Scale**
- Only 1,150 samples
- Doesn't test scalability
- Different dynamics than large datasets

**Use Case**: Synthetic datasets are for **development/debugging only**, not for final evaluation or research results.

---

## Typical Workflow

### Phase 1: Development & Debugging (1-2 hours)

```bash
# 1. Generate synthetic dataset
python mdlm_atat/scripts/generate_tiny_dataset.py

# 2. Quick smoke test (5 min)
python mdlm_atat/scripts/train_atat.py \
    --config configs/atat/tiny.yaml \
    --max-steps 100

# 3. Fix any errors found

# 4. Run slightly longer test (1 hour)
python mdlm_atat/scripts/train_atat.py \
    --config configs/atat/tiny.yaml \
    --max-steps 10000
```

### Phase 2: Real Training (on real dataset)

```bash
# Once code is debugged, move to real data

# Download real dataset
python mdlm_atat/scripts/download_datasets.py --dataset openwebtext

# Run on real data (multiple hours/days)
python mdlm_atat/scripts/train_atat.py \
    --config configs/atat/large.yaml \
    --max-steps 100000
```

---

## Monitoring Training

### Metrics to Watch

When training on synthetic data, look for:

| Metric | Good Value | Bad Value | Interpretation |
|--------|-----------|----------|-----------------|
| **Training loss** | Decreasing | Increasing | Model learning? |
| **Validation loss** | Decreasing | Flat | Generalization? |
| **Perplexity** | Decreasing | High | Language modeling? |
| **GPU memory** | Stable | Growing | Memory leak? |
| **Training speed** | Stable | Slowing down | Performance issue? |

### Example Output

```
Step 100/1000 | Loss: 4.523 | Perplexity: 92.3 | Speed: 1200 tok/s
Step 200/1000 | Loss: 4.201 | Perplexity: 66.8 | Speed: 1205 tok/s
Step 300/1000 | Loss: 3.987 | Perplexity: 54.2 | Speed: 1200 tok/s
Step 400/1000 | Loss: 3.745 | Perplexity: 42.1 | Speed: 1203 tok/s
...
Step 1000/1000 | Loss: 2.134 | Perplexity: 8.4 | Speed: 1201 tok/s

✅ Training complete! Synthetic dataset test passed.
```

---

## Common Questions

### Q: Can I use synthetic data for final results?

**A**: No. Synthetic data is too simple and doesn't capture real language patterns. Use it only for development/debugging. Always report results on real datasets (LM1B, OpenWebText, etc.).

### Q: How long should I train on synthetic data?

**A**: Typically 1000-10000 steps (5 min - 1 hour):
- 100 steps: Quick smoke test (< 1 min)
- 1000 steps: Verify convergence behavior (5 min)
- 10000 steps: Full debugging run (1 hour)

### Q: Why does loss plateau on synthetic data?

**A**: Synthetic data has limited diversity, so the model memorizes everything quickly. This is normal and expected.

### Q: Can I modify the synthetic data?

**A**: Yes! Edit `generate_tiny_dataset.py`:
- Add more templates
- Change text length
- Modify sample counts
- Adjust tokenization

### Q: Is synthetic data deterministic?

**A**: Yes, same samples are generated every time. To get different samples, modify the random seed or templates in the script.

---

## Implementation Details

### Generation Algorithm

```python
def generate_synthetic_text_samples(num_samples, min_length=50, max_length=200):
    templates = [
        "The quick brown fox jumps over the lazy dog. ",
        "In a world of constant change, adaptability is key. ",
        # ... 8 more templates
    ]
    
    samples = []
    for i in range(num_samples):
        # Randomly select 1-20 sentences
        num_sentences = random.randint(min_length // 10, max_length // 10)
        
        text_parts = []
        for j in range(num_sentences):
            # Random template
            template = templates[random.randint(len(templates))]
            # Add variation with numbers
            text = template.replace(".", f" {i*j + random.randint(100)}.")
            text_parts.append(text)
        
        samples.append(" ".join(text_parts))
    
    return samples
```

### Tokenization

```python
def tokenize_and_save_dataset(samples, tokenizer, output_path, max_length=1024):
    all_tokens = []
    for text in samples:
        tokens = tokenizer.encode(text, add_special_tokens=True)
        
        # Pad or truncate to max_length
        if len(tokens) < max_length:
            tokens = tokens + [PAD_ID] * (max_length - len(tokens))
        else:
            tokens = tokens[:max_length]
        
        all_tokens.append(tokens)
    
    # Save as HuggingFace Dataset
    dataset = Dataset.from_dict({'input_ids': all_tokens})
    dataset.save_to_disk(output_path)
```

---

## Next Steps

1. ✅ **Generate synthetic dataset**:
   ```bash
   python mdlm_atat/scripts/generate_tiny_dataset.py
   ```

2. ✅ **Run quick test**:
   ```bash
   python mdlm_atat/scripts/train_atat.py --config configs/atat/tiny.yaml --max-steps 100
   ```

3. ✅ **Verify no errors**:
   Check logs for runtime issues, memory problems, etc.

4. ✅ **Debug if needed**:
   Fix any code issues found

5. ✅ **Move to real data**:
   Once working, use real datasets for final training

---

*Document Version: 1.0*  
*Created: December 8, 2025*
