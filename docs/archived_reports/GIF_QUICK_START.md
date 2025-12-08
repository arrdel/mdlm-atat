# Quick Reference: Creating Diffusion Sampling GIFs

## TL;DR - Just Show Me How to Create a GIF!

### Simplest Way (No Model Needed - Demo Only)

```bash
cd /home/adelechinda/home/projects/mdlm/mdlm_atat/scripts
python create_sampling_gif.py --output demo.gif --style compact
```

This creates a GIF showing random sampling (for demo purposes).

### With Your Trained Model

```bash
python create_sampling_gif.py \
    --checkpoint /path/to/your/model.ckpt \
    --output my_sampling.gif \
    --steps 50 \
    --style compact
```

### With a Prompt

```bash
python create_sampling_gif.py \
    --prompt "The future of artificial intelligence" \
    --output prompted.gif \
    --steps 50
```

---

## Command Line Options

| Option | Description | Default | Example |
|--------|-------------|---------|---------|
| `--checkpoint` | Path to trained model | None (uses random model) | `model.ckpt` |
| `--output` | Output GIF filename | `diffusion_sampling.gif` | `my_viz.gif` |
| `--style` | `compact` or `detailed` | `compact` | `compact` |
| `--prompt` | Starting text prompt | Empty | `"The cat"` |
| `--length` | Sequence length | 50 | 30 |
| `--steps` | Number of denoising steps | 50 | 100 |
| `--fps` | Frames per second | 5 | 10 |
| `--device` | `cuda` or `cpu` | Auto-detect | `cuda` |

---

## Visual Styles Compared

### Compact Style (Recommended)
- ✅ Clean, paper-ready visualization
- ✅ Small file size (~100 KB)
- ✅ Focuses on text generation
- ✅ Shows masked tokens as `[?]`
- ✅ Highlights newly revealed tokens in green
- ✅ Perfect for presentations and papers

**Best for**: Publications, presentations, sharing

### Detailed Style
- ✅ Shows importance heatmap
- ✅ Shows uncertainty heatmap
- ✅ Color-coded tokens
- ✅ Timestep information
- ⚠️ Larger file size (~500 KB - 2 MB)
- ⚠️ More complex visually

**Best for**: Analysis, debugging, understanding model behavior

---

## Common Use Cases

### 1. Demo for Supervisor

```bash
# Quick 30-step visualization
python create_sampling_gif.py \
    --output supervisor_demo.gif \
    --steps 30 \
    --length 40 \
    --style compact \
    --fps 5
```

**Result**: Clean GIF showing text appearing token by token

### 2. Compare Different Steps

```bash
# Fast sampling (fewer steps)
python create_sampling_gif.py --steps 20 --output fast_sampling.gif

# Slow sampling (more steps)
python create_sampling_gif.py --steps 100 --output slow_sampling.gif
```

### 3. Show Importance Learning

```bash
# Use detailed style with your trained model
python create_sampling_gif.py \
    --checkpoint trained_atat.ckpt \
    --style detailed \
    --output importance_viz.gif
```

**Result**: Shows which tokens model considers important

### 4. Generate Multiple Examples

```bash
# Create several GIFs with different prompts
for prompt in "The future" "Science and" "Once upon a time"; do
    python create_sampling_gif.py \
        --prompt "$prompt" \
        --output "sample_${prompt// /_}.gif"
done
```

---

## From Python Code

### Minimal Example

```python
from transformers import AutoTokenizer
from mdlm_atat.utils.gif_visualization import CompactDiffusionGIF

# Setup
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Your sampling trajectory (list of token tensors)
trajectory = [...]  # From your sampling loop

# Create GIF
viz = CompactDiffusionGIF(tokenizer, fps=5)
viz.create_compact_gif(trajectory, save_path="output.gif")
```

### Full Example with Model

```python
import torch
from transformers import AutoTokenizer
from mdlm_atat.models.atat_dit import ATATDiT
from mdlm_atat.utils.gif_visualization import DiffusionGIFVisualizer

# Load model and tokenizer
model = ATATDiT(...)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model.eval()

# Sample and track trajectory
trajectory = []
importance_traj = []
uncertainty_traj = []

mask_idx = tokenizer.mask_token_id
x_t = torch.full((1, 50), mask_idx)

for step in range(50):
    t = torch.tensor([1.0 - step/50])
    
    # Store state
    trajectory.append(x_t[0].clone())
    
    # Forward
    logits, importance = model(x_t, t, return_importance=True)
    
    # Compute uncertainty
    probs = torch.softmax(logits, dim=-1)
    uncertainty = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
    
    # Store
    importance_traj.append(importance[0].clone())
    uncertainty_traj.append(uncertainty[0].clone())
    
    # Denoise (your logic here)
    # ...

# Create GIF
viz = DiffusionGIFVisualizer(tokenizer, show_importance=True)
viz.create_sampling_gif(
    trajectory=trajectory,
    importance_trajectory=importance_traj,
    uncertainty_trajectory=uncertainty_traj,
    save_path="sampling.gif"
)
```

---

## Tips & Tricks

### Make GIF Smaller
```bash
# Use compact style (saves ~80% space)
--style compact

# Reduce FPS (fewer frames)
--fps 3

# Fewer steps
--steps 30

# Shorter sequence
--length 30
```

### Make GIF Higher Quality
```python
# From Python, increase resolution
viz = CompactDiffusionGIF(
    tokenizer,
    width=1200,  # Bigger
    height=300,
    fps=10  # Smoother
)
```

### For Presentations
```bash
# Large, slow GIF for visibility
python create_sampling_gif.py \
    --output presentation.gif \
    --steps 50 \
    --fps 3 \
    --length 40
```

Then insert into PowerPoint/Google Slides.

### For Paper
```bash
# Clean, compact GIF
python create_sampling_gif.py \
    --output paper_figure.gif \
    --style compact \
    --steps 50 \
    --length 50 \
    --fps 5
```

### For Social Media
```bash
# Eye-catching, fast
python create_sampling_gif.py \
    --prompt "AI can now" \
    --output social.gif \
    --steps 40 \
    --fps 8
```

---

## Troubleshooting

### "CUDA out of memory"
```bash
# Use CPU instead
python create_sampling_gif.py --device cpu --output demo.gif
```

### "ModuleNotFoundError: No module named 'PIL'"
```bash
pip install Pillow
```

### GIF not animating
- Check FPS is > 0
- Ensure you have multiple frames (steps > 1)
- Try opening in different viewer (browser vs image viewer)

### Colors look wrong
- Check your terminal/image viewer supports colors
- Try opening GIF in web browser

### Script hangs
- Reduce `--steps` and `--length` for faster generation
- Use `--device cpu` if GPU issues
- Check you're not running out of memory

---

## File Locations

- **Script**: `/home/adelechinda/home/projects/mdlm/mdlm_atat/scripts/create_sampling_gif.py`
- **Module**: `/home/adelechinda/home/projects/mdlm/mdlm_atat/utils/gif_visualization.py`
- **Full Docs**: `/home/adelechinda/home/projects/mdlm/mdlm_atat/utils/GIF_VISUALIZATION_README.md`

---

## Quick Commands Cheat Sheet

```bash
# Most common use cases
cd /home/adelechinda/home/projects/mdlm/mdlm_atat/scripts

# 1. Quick demo
python create_sampling_gif.py --output demo.gif

# 2. With model
python create_sampling_gif.py --checkpoint model.ckpt --output viz.gif

# 3. With prompt
python create_sampling_gif.py --prompt "Text here" --output prompted.gif

# 4. Detailed view
python create_sampling_gif.py --style detailed --output analysis.gif

# 5. Long sequence
python create_sampling_gif.py --length 100 --steps 100 --output long.gif

# 6. Fast animation
python create_sampling_gif.py --fps 10 --output fast.gif

# 7. For paper (high quality)
python create_sampling_gif.py --steps 100 --fps 5 --length 50 --style compact --output paper_quality.gif
```

---

## What the GIF Shows

### Compact Style Shows:
1. **Step counter**: "Step X/Y"
2. **Masked tokens**: Displayed as `[?]`
3. **Revealed tokens**: Show actual text
4. **New reveals**: Highlighted in **green**
5. **Text wrapping**: Auto-wraps long sequences

### Detailed Style Shows:
1. **Token display**: With color-coded boxes
2. **Importance heatmap**: Red (important) → Green (easy)
3. **Uncertainty heatmap**: Yellow (certain) → Red (uncertain)
4. **Timestep**: Current diffusion time `t`
5. **Synchronized views**: All aligned by token position

---

## Example Outputs

After running:
```bash
python create_sampling_gif.py --output demo.gif --steps 30
```

You'll get:
- **demo.gif**: Animated GIF file
- **Console output**: Shows progress and final text
- **File size**: ~100-500 KB depending on style

Open `demo.gif` in:
- Web browser
- Image viewer
- Insert into presentation
- Upload to W&B
- Share on social media

---

**That's it!** You now have everything you need to create amazing diffusion sampling visualizations just like in the MDLM paper! 🎬

For more details, see `GIF_VISUALIZATION_README.md`.
