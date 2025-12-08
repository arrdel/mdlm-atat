# GIF Visualization for ATAT Diffusion Sampling

This module creates animated GIF visualizations of the diffusion sampling process, similar to those in the MDLM paper. Watch tokens appear progressively as they are denoised!

## Features

🎬 **Two Visualization Styles**:
1. **Detailed View**: Shows tokens + importance heatmap + uncertainty heatmap
2. **Compact View**: Clean, paper-style visualization focusing on text only

🎨 **Highlights**:
- Progressive token unmasking
- Color-coded revealed vs masked tokens
- Just-revealed tokens highlighted in green
- Optional importance and uncertainty overlays
- Timestep information
- Customizable colors, sizes, and frame rates

## Quick Start

### 1. Simple Demo

Create a GIF from any sampling run:

```bash
cd mdlm_atat/scripts
python create_sampling_gif.py --output my_sampling.gif --steps 50 --style compact
```

### 2. With Trained Model

```bash
python create_sampling_gif.py \
    --checkpoint path/to/your/model.ckpt \
    --output sampling.gif \
    --steps 100 \
    --style detailed \
    --prompt "The future of AI"
```

### 3. From Python Code

```python
from transformers import AutoTokenizer
from mdlm_atat.utils.gif_visualization import CompactDiffusionGIF

tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Assume you have a trajectory from sampling
# trajectory = [step_0_tokens, step_1_tokens, ..., step_N_tokens]

visualizer = CompactDiffusionGIF(tokenizer, fps=5)
visualizer.create_compact_gif(
    trajectory=trajectory,
    save_path="output.gif"
)
```

## Visualization Styles

### Compact Style (Recommended for Papers)

Clean, minimal visualization similar to MDLM paper:

```python
from mdlm_atat.utils.gif_visualization import CompactDiffusionGIF

visualizer = CompactDiffusionGIF(
    tokenizer=tokenizer,
    width=800,
    height=200,
    fps=5,
    bg_color=(255, 255, 255),  # White
    highlight_color=(76, 175, 80),  # Green
)

visualizer.create_compact_gif(
    trajectory=trajectory,
    save_path="compact_sampling.gif",
    show_step_numbers=True,
    chars_per_line=80
)
```

**Features**:
- ✅ Clean text display
- ✅ Masked tokens shown as `[?]`
- ✅ Just-revealed tokens highlighted in green
- ✅ Wraps text automatically
- ✅ Shows step numbers

**Output**: ~100-200 KB GIF

### Detailed Style (For Analysis)

Comprehensive view with importance and uncertainty:

```python
from mdlm_atat.utils.gif_visualization import DiffusionGIFVisualizer

visualizer = DiffusionGIFVisualizer(
    tokenizer=tokenizer,
    figsize=(16, 6),
    fps=5,
    font_size=14,
    show_importance=True,
    show_uncertainty=True
)

visualizer.create_sampling_gif(
    trajectory=trajectory,
    importance_trajectory=importance_scores,  # Optional
    uncertainty_trajectory=uncertainty_scores,
    timesteps=timesteps,
    save_path="detailed_sampling.gif"
)
```

**Features**:
- ✅ Token display with color coding
- ✅ Importance heatmap (red=important, green=easy)
- ✅ Uncertainty heatmap (yellow=certain, red=uncertain)
- ✅ Timestep information
- ✅ Synchronized views

**Output**: ~500KB - 2MB GIF

## Advanced Usage

### Side-by-Side Comparison

Compare baseline vs ATAT sampling:

```python
from mdlm_atat.utils.gif_visualization import create_side_by_side_comparison_gif

create_side_by_side_comparison_gif(
    baseline_trajectory=baseline_samples,
    atat_trajectory=atat_samples,
    tokenizer=tokenizer,
    save_path="comparison.gif",
    fps=5
)
```

### Custom Sampling with Visualization

```python
import torch
from mdlm_atat.models.atat_dit import ATATDiT
from mdlm_atat.utils.gif_visualization import DiffusionGIFVisualizer

model = ATATDiT(...)  # Your model
tokenizer = ...  # Your tokenizer

# Storage for trajectory
trajectory = []
importance_trajectory = []
uncertainty_trajectory = []

# Sampling loop
x_t = torch.full((1, length), mask_idx, dtype=torch.long)

for step in range(num_steps):
    t = torch.tensor([1.0 - step/num_steps])
    
    # Forward pass
    logits, importance = model(x_t, t, return_importance=True)
    
    # Compute uncertainty
    probs = torch.softmax(logits, dim=-1)
    uncertainty = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
    
    # Store for visualization
    trajectory.append(x_t[0].clone())
    importance_trajectory.append(importance[0].clone())
    uncertainty_trajectory.append(uncertainty[0].clone())
    
    # ... denoising step ...

# Create GIF
visualizer = DiffusionGIFVisualizer(tokenizer)
visualizer.create_sampling_gif(
    trajectory=trajectory,
    importance_trajectory=importance_trajectory,
    uncertainty_trajectory=uncertainty_trajectory,
    save_path="my_sampling.gif"
)
```

## Customization

### Colors

```python
visualizer = CompactDiffusionGIF(
    tokenizer=tokenizer,
    bg_color=(240, 240, 240),      # Light gray background
    text_color=(0, 0, 0),           # Black text
    masked_color=(150, 150, 150),   # Gray for masks
    highlight_color=(255, 87, 34),  # Orange for new tokens
)
```

### Frame Rate

```python
# Slower animation (2 fps)
visualizer = CompactDiffusionGIF(tokenizer, fps=2)

# Faster animation (10 fps)
visualizer = CompactDiffusionGIF(tokenizer, fps=10)
```

### Image Size

```python
# Larger visualization
visualizer = CompactDiffusionGIF(
    tokenizer,
    width=1200,
    height=300
)

# For presentations
visualizer = DiffusionGIFVisualizer(
    tokenizer,
    figsize=(20, 8),  # Width, height in inches
    font_size=16
)
```

## Examples

### Example 1: Quick Demo

```bash
# Create a quick demo GIF (no trained model needed)
python create_sampling_gif.py \
    --output demo.gif \
    --steps 30 \
    --length 30 \
    --style compact \
    --fps 5
```

### Example 2: With Prompt

```bash
# Generate with a prompt
python create_sampling_gif.py \
    --prompt "The capital of France is" \
    --output prompted_sampling.gif \
    --steps 50 \
    --style compact
```

### Example 3: Detailed Analysis

```bash
# Full detailed view with all metrics
python create_sampling_gif.py \
    --checkpoint models/atat_trained.ckpt \
    --output analysis.gif \
    --steps 100 \
    --style detailed \
    --fps 3
```

### Example 4: Long Sequence

```bash
# Generate longer sequence
python create_sampling_gif.py \
    --length 100 \
    --steps 100 \
    --output long_sequence.gif \
    --style compact
```

## Output Examples

### Compact Style
```
Step 1/50: [?] [?] [?] [?] [?] [?] [?] [?]
Step 10/50: The [?] [?] sat [?] [?] [?] mat
Step 25/50: The cat [?] sat on [?] [?] mat
Step 50/50: The cat quickly sat on the soft mat
            ^^^^^
          (highlighted - just revealed)
```

### Detailed Style
```
┌─────────────────────────────────────────┐
│  Step 25/50 (t=0.500)                  │
│  The cat [?] sat on [?] the mat       │
├─────────────────────────────────────────┤
│  Importance: ████░░░░████░░░░████      │
│             (high) (low) (medium)      │
├─────────────────────────────────────────┤
│  Uncertainty: ░░░░░░░░████████░░░░     │
│              (certain) (uncertain)     │
└─────────────────────────────────────────┘
```

## Performance Tips

1. **Limit sequence length**: For GIFs, 30-50 tokens is usually enough
2. **Adjust FPS**: 3-5 fps is good for analysis, 10 fps for quick previews
3. **Use compact style for papers**: Smaller file size, cleaner look
4. **Reduce steps for faster generation**: 30-50 steps often sufficient for demos

## Integration with W&B

Log GIFs to Weights & Biases:

```python
import wandb

# Create GIF
gif_path = visualizer.create_compact_gif(trajectory, save_path="temp.gif")

# Log to W&B
wandb.log({
    "sampling_visualization": wandb.Video(gif_path, fps=5, format="gif")
})
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'PIL'"
```bash
pip install Pillow
```

### "Font not found" warning
```bash
# Install DejaVu fonts (Ubuntu/Debian)
sudo apt-get install fonts-dejavu

# Or specify custom font in code
visualizer = CompactDiffusionGIF(tokenizer)
visualizer.font = ImageFont.truetype("/path/to/font.ttf", 20)
```

### GIF too large
```python
# Reduce size:
# 1. Use compact style
# 2. Lower resolution
visualizer = CompactDiffusionGIF(tokenizer, width=600, height=150)

# 3. Reduce FPS
visualizer = CompactDiffusionGIF(tokenizer, fps=3)

# 4. Limit sequence length
trajectory = trajectory[:30]  # Only first 30 steps
```

### Colors not showing correctly
```python
# Ensure RGB tuples, not hex
bg_color = (255, 255, 255)  # ✓ Correct
bg_color = "#FFFFFF"         # ✗ Wrong
```

## API Reference

### CompactDiffusionGIF

```python
CompactDiffusionGIF(
    tokenizer,              # HuggingFace tokenizer
    width=800,             # Image width in pixels
    height=200,            # Image height per line
    fps=5,                 # Frames per second
    bg_color=(255,255,255), # Background RGB
    text_color=(0,0,0),    # Text RGB
    masked_color=(200,200,200), # Masked token RGB
    highlight_color=(76,175,80) # Highlight RGB
)
```

### DiffusionGIFVisualizer

```python
DiffusionGIFVisualizer(
    tokenizer,                  # HuggingFace tokenizer
    figsize=(16, 6),           # Matplotlib figure size
    fps=5,                     # Frames per second
    font_size=14,              # Base font size
    show_importance=True,      # Show importance heatmap
    show_uncertainty=True      # Show uncertainty heatmap
)
```

## Citation

If you use these visualizations in your work:

```bibtex
@software{atat_viz,
  title={ATAT GIF Visualization Tools},
  author={Your Name},
  year={2025},
  note={Visualization tools for masked diffusion language models}
}
```

## Contributing

To add new visualization styles:

1. Extend `CompactDiffusionGIF` or `DiffusionGIFVisualizer`
2. Add new coloring schemes
3. Implement custom frame generation
4. Update this README

## License

Same as parent MDLM-ATAT project (Apache 2.0)

---

**Questions?** Check the main project README or open an issue.

**Examples**: See `examples/` directory for sample outputs (coming soon!)
