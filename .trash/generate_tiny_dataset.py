#!/usr/bin/env python3
"""
Generate a tiny synthetic dataset for quick ATAT training testing.

This creates a small dataset with:
- Training set: 1000 samples
- Validation set: 100 samples
- Test set: 50 samples

Each sample is GPT-2 tokenized text of reasonable length for sequence modeling.
"""

import os
import argparse
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer
import datasets
from datasets import Dataset


def generate_synthetic_text_samples(num_samples, min_length=50, max_length=200):
    """Generate synthetic text samples with some structure."""
    
    templates = [
        "The quick brown fox jumps over the lazy dog. ",
        "In a world of constant change, adaptability is key. ",
        "Machine learning models learn patterns from data. ",
        "Natural language processing enables computers to understand text. ",
        "Deep learning has revolutionized artificial intelligence. ",
        "Transformers have become the dominant architecture for NLP. ",
        "Diffusion models generate high-quality samples iteratively. ",
        "Language models can generate coherent and contextual text. ",
        "Training neural networks requires large amounts of data. ",
        "The future of AI holds many exciting possibilities. ",
    ]
    
    samples = []
    for i in range(num_samples):
        # Generate a sample by repeating templates with variations
        num_sentences = np.random.randint(min_length // 10, max_length // 10)
        text_parts = []
        
        for j in range(num_sentences):
            template = templates[np.random.randint(len(templates))]
            # Add some variation with numbers
            text_parts.append(template.replace(".", f" {i * j + np.random.randint(100)}."))
        
        samples.append(" ".join(text_parts))
    
    return samples


def tokenize_and_save_dataset(samples, tokenizer, output_path, max_length=1024):
    """Tokenize samples and save in HuggingFace Dataset format."""
    
    print(f"Tokenizing {len(samples)} samples...")
    
    # Tokenize all samples
    all_tokens = []
    all_attention_masks = []
    
    for i, text in enumerate(samples):
        if i % 100 == 0:
            print(f"  Tokenized {i}/{len(samples)} samples")
        
        tokens = tokenizer.encode(text, add_special_tokens=True)
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = [1] * len(tokens)
        
        # Pad or truncate to max_length
        if len(tokens) < max_length:
            padding_length = max_length - len(tokens)
            tokens = tokens + [tokenizer.eos_token_id] * padding_length
            attention_mask = attention_mask + [0] * padding_length
        else:
            tokens = tokens[:max_length]
            attention_mask = attention_mask[:max_length]
        
        all_tokens.append(tokens)
        all_attention_masks.append(attention_mask)
    
    # Create HuggingFace Dataset
    print(f"Creating HuggingFace Dataset...")
    dataset_dict = {
        'input_ids': all_tokens,
        'attention_mask': all_attention_masks
    }
    hf_dataset = Dataset.from_dict(dataset_dict)
    
    # Convert to torch format
    hf_dataset = hf_dataset.with_format('torch')
    
    print(f"Saving dataset to {output_path}")
    print(f"  Samples: {len(hf_dataset)}")
    print(f"  Features: {hf_dataset.features}")
    
    # Save in HuggingFace format
    hf_dataset.save_to_disk(output_path)
    
    return hf_dataset


def main():
    parser = argparse.ArgumentParser(description='Generate tiny synthetic dataset for ATAT testing')
    parser.add_argument('--cache-dir', type=str, 
                        default='/media/scratch/adele/mdlm_fresh/data_cache',
                        help='Directory to save the synthetic datasets')
    parser.add_argument('--train-samples', type=int, default=1000,
                        help='Number of training samples')
    parser.add_argument('--val-samples', type=int, default=100,
                        help='Number of validation samples')
    parser.add_argument('--test-samples', type=int, default=50,
                        help='Number of test samples')
    parser.add_argument('--max-length', type=int, default=1024,
                        help='Maximum sequence length')
    
    args = parser.parse_args()
    
    # Create cache directory if needed
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Generating Tiny Synthetic Dataset for ATAT Testing")
    print("="*60)
    
    # Load tokenizer
    print("\nLoading GPT-2 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    
    # Generate and save training data
    print(f"\n[1/3] Generating {args.train_samples} training samples...")
    train_samples = generate_synthetic_text_samples(args.train_samples)
    train_path = cache_dir / 'synthetic_tiny_train_bs1024_wrapped.dat'
    train_data = tokenize_and_save_dataset(train_samples, tokenizer, train_path, args.max_length)
    
    # Generate and save validation data
    print(f"\n[2/3] Generating {args.val_samples} validation samples...")
    val_samples = generate_synthetic_text_samples(args.val_samples)
    val_path = cache_dir / 'synthetic_tiny_validation_bs1024_wrapped.dat'
    val_data = tokenize_and_save_dataset(val_samples, tokenizer, val_path, args.max_length)
    
    # Generate and save test data
    print(f"\n[3/3] Generating {args.test_samples} test samples...")
    test_samples = generate_synthetic_text_samples(args.test_samples)
    test_path = cache_dir / 'synthetic_tiny_test_bs1024_wrapped.dat'
    test_data = tokenize_and_save_dataset(test_samples, tokenizer, test_path, args.max_length)
    
    print("\n" + "="*60)
    print("Dataset Generation Complete!")
    print("="*60)
    print(f"\nTrain: {train_path}")
    print(f"  Samples: {len(train_data)}")
    print(f"\nValidation: {val_path}")
    print(f"  Samples: {len(val_data)}")
    print(f"\nTest: {test_path}")
    print(f"  Samples: {len(test_data)}")


if __name__ == '__main__':
    main()
