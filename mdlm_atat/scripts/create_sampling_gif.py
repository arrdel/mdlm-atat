#!/usr/bin/env python
"""
Demo script for creating diffusion sampling GIF visualizations

Usage:
    python create_sampling_gif.py --checkpoint path/to/model.ckpt --output sampling.gif
"""

import sys
sys.path.insert(0, '/home/adelechinda/home/projects/mdlm')

import torch
import argparse
from transformers import AutoTokenizer

from mdlm_atat.models.atat_dit import ATATDiT
from trash.gif_visualization import (
    DiffusionGIFVisualizer,
    CompactDiffusionGIF,
    create_sample_visualization
)


def sample_with_trajectory_tracking(
    model,
    tokenizer,
    prompt: str = "",
    length: int = 50,
    num_steps: int = 50,
    device: str = "cuda",
):
    """
    Sample from model while tracking the full trajectory.
    
    Returns:
        trajectory: List of token tensors at each step
        importance_trajectory: List of importance scores
        uncertainty_trajectory: List of uncertainty scores
        timesteps: List of timestep values
    """
    model.eval()
    model.to(device)
    
    # Storage
    trajectory = []
    importance_trajectory = []
    uncertainty_trajectory = []
    timesteps = []
    
    # Initialize
    mask_index = tokenizer.mask_token_id if hasattr(tokenizer, 'mask_token_id') else tokenizer.vocab_size
    
    if prompt:
        # Encode prompt
        prompt_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        x_t = torch.full((1, length), mask_index, dtype=torch.long, device=device)
        x_t[0, :len(prompt_ids[0])] = prompt_ids[0]
    else:
        x_t = torch.full((1, length), mask_index, dtype=torch.long, device=device)
    
    # Sampling loop
    print(f"Sampling {num_steps} steps...")
    with torch.no_grad():
        for step in range(num_steps + 1):
            t_value = max(0.0, 1.0 - step / num_steps)
            t = torch.tensor([t_value], device=device)
            timesteps.append(t_value)
            
            # Store current state
            trajectory.append(x_t[0].cpu().clone())
            
            if step == num_steps:
                break
            
            # Get predictions
            try:
                logits, importance = model(x_t, t, return_importance=True)
            except:
                # If model doesn't support return_importance
                logits = model(x_t, t)
                importance = None
            
            # Compute uncertainty (entropy)
            probs = torch.softmax(logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
            
            # Store
            if importance is not None:
                importance_trajectory.append(importance[0].cpu().clone())
            uncertainty_trajectory.append(entropy[0].cpu().clone())
            
            # Denoise step - uncertainty-guided
            is_masked = (x_t[0] == mask_index)
            num_masked = is_masked.sum().item()
            
            if num_masked > 0:
                # Calculate how many to denoise this step
                remaining_steps = num_steps - step
                k = max(1, num_masked // remaining_steps)
                
                # Get masked positions
                masked_positions = torch.where(is_masked)[0]
                
                # Select top-k uncertain positions
                uncertainties = entropy[0][masked_positions]
                top_k_indices = torch.topk(
                    uncertainties, 
                    min(k, len(masked_positions))
                )[1]
                positions_to_denoise = masked_positions[top_k_indices]
                
                # Sample tokens for selected positions
                for pos in positions_to_denoise:
                    probs_pos = torch.softmax(logits[0, pos], dim=-1)
                    sampled_token = torch.multinomial(probs_pos, 1)
                    x_t[0, pos] = sampled_token
            
            # Progress
            if (step + 1) % 10 == 0:
                print(f"  Step {step + 1}/{num_steps} - Masked: {is_masked.sum().item()}/{length}")
    
    print("✓ Sampling complete!")
    
    return (
        trajectory,
        importance_trajectory if importance_trajectory else None,
        uncertainty_trajectory,
        timesteps
    )


def main():
    parser = argparse.ArgumentParser(description="Create diffusion sampling GIF")
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to model checkpoint (optional, will use untrained model if not provided)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="diffusion_sampling.gif",
        help="Output GIF path"
    )
    parser.add_argument(
        "--style",
        type=str,
        choices=["detailed", "compact"],
        default="compact",
        help="Visualization style"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="Optional prompt to condition on"
    )
    parser.add_argument(
        "--length",
        type=int,
        default=50,
        help="Sequence length"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Number of denoising steps"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=5,
        help="Frames per second"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Diffusion Sampling GIF Creator")
    print("=" * 60)
    
    # Load tokenizer
    print("\n1. Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if not hasattr(tokenizer, 'mask_token_id'):
        tokenizer.mask_token_id = tokenizer.vocab_size
    print(f"   ✓ Tokenizer loaded (vocab size: {tokenizer.vocab_size})")
    
    # Load or create model
    print("\n2. Loading model...")
    if args.checkpoint:
        # Load from checkpoint
        print(f"   Loading from checkpoint: {args.checkpoint}")
        # This would load your trained model
        # model = ATATDiT.load_from_checkpoint(args.checkpoint)
        print("   Note: Checkpoint loading not implemented in demo")
        print("   Using random initialized model for demonstration")
        
        # For demo, create a mock model
        class MockConfig:
            class model:
                length = args.length
                dim = 256
                n_layers = 4
                n_heads = 4
                @staticmethod
                def get(key, default=None):
                    return default
        
        model = ATATDiT(MockConfig(), tokenizer.vocab_size)
    else:
        # Create demo model
        print("   Creating demo model (random weights)...")
        class MockConfig:
            class model:
                length = args.length
                dim = 256
                n_layers = 4
                n_heads = 4
                @staticmethod
                def get(key, default=None):
                    return default
        
        model = ATATDiT(MockConfig(), tokenizer.vocab_size)
    
    print(f"   ✓ Model loaded")
    
    # Sample with trajectory tracking
    print(f"\n3. Generating sample (length={args.length}, steps={args.steps})...")
    if args.prompt:
        print(f"   Prompt: '{args.prompt}'")
    
    trajectory, importance_traj, uncertainty_traj, timesteps = sample_with_trajectory_tracking(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        length=args.length,
        num_steps=args.steps,
        device=args.device
    )
    
    # Create visualization
    print(f"\n4. Creating {args.style} GIF visualization...")
    
    if args.style == "detailed":
        visualizer = DiffusionGIFVisualizer(
            tokenizer=tokenizer,
            fps=args.fps,
            show_importance=importance_traj is not None,
            show_uncertainty=True
        )
        visualizer.create_sampling_gif(
            trajectory=trajectory,
            importance_trajectory=importance_traj,
            uncertainty_trajectory=uncertainty_traj,
            timesteps=timesteps,
            save_path=args.output
        )
    else:  # compact
        visualizer = CompactDiffusionGIF(
            tokenizer=tokenizer,
            fps=args.fps
        )
        visualizer.create_compact_gif(
            trajectory=trajectory,
            save_path=args.output
        )
    
    # Show final text
    print(f"\n5. Final generated text:")
    final_tokens = trajectory[-1].numpy()
    final_text = tokenizer.decode(final_tokens, skip_special_tokens=True)
    print(f"   '{final_text}'")
    
    print(f"\n" + "=" * 60)
    print(f"✓ Done! GIF saved to: {args.output}")
    print(f"  - Frames: {len(trajectory)}")
    print(f"  - FPS: {args.fps}")
    print(f"  - Style: {args.style}")
    print("=" * 60)


if __name__ == "__main__":
    main()
