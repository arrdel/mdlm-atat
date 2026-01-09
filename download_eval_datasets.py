#!/usr/bin/env python3
"""
Download and Prepare Evaluation Datasets for ATAT

Downloads multiple datasets for comprehensive baseline and ATAT evaluation:
- PTB (Penn Treebank): News text, ~1M tokens
- WikiText-103: Wikipedia articles, ~103M tokens  
- LM1B: Google's 1B word dataset
- Lambada: Cloze prediction task
- AG News: News classification corpus

Usage:
    python download_eval_datasets.py [OPTIONS]

Options:
    --cache-dir PATH        Download destination [default: /media/scratch/adele/mdlm_fresh/data_cache]
    --datasets NAMES        Comma-separated dataset names [default: all]
    --skip-existing         Skip if already downloaded
    --verify                Verify downloads
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Optional
import hashlib
import urllib.request
import tarfile
import gzip
import shutil

# Color codes for output
class Colors:
    BLUE = '\033[0;34m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    RED = '\033[0;31m'
    NC = '\033[0m'

def print_header(message: str):
    """Print formatted header."""
    print(f"\n{Colors.BLUE}{'='*70}{Colors.NC}")
    print(f"{Colors.BLUE}{message.center(70)}{Colors.NC}")
    print(f"{Colors.BLUE}{'='*70}{Colors.NC}\n")

def download_file(url: str, destination: Path, chunk_size: int = 8192) -> bool:
    """Download file with progress bar."""
    try:
        print(f"  Downloading from: {url}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        
        with urllib.request.urlopen(url) as response:
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(destination, 'wb') as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"  Progress: {percent:.1f}% ({downloaded / 1024 / 1024:.1f}MB / {total_size / 1024 / 1024:.1f}MB)", end='\r')
        
        print(f"  {Colors.GREEN}✓{Colors.NC} Downloaded successfully")
        return True
    except Exception as e:
        print(f"  {Colors.RED}✗{Colors.NC} Download failed: {e}")
        return False

def download_ptb(cache_dir: Path) -> bool:
    """Download Penn Treebank dataset."""
    print_header("Downloading Penn Treebank (PTB)")
    
    ptb_dir = cache_dir / "ptb"
    ptb_dir.mkdir(parents=True, exist_ok=True)
    
    # PTB is available through NLTK
    print(f"  Dataset: Penn Treebank")
    print(f"  Size: ~1M tokens")
    print(f"  Location: {ptb_dir}")
    print()
    
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        
        # Download PTB via nltk
        print(f"  Downloading PTB via NLTK...")
        try:
            # Try using datasets library instead (more reliable)
            from datasets import load_dataset
            dataset = load_dataset("ptb_text_only", cache_dir=str(cache_dir))
            
            # Save to directory
            for split in ['train', 'validation', 'test']:
                if split in dataset:
                    output_file = ptb_dir / f"{split}.txt"
                    with open(output_file, 'w') as f:
                        for item in dataset[split]:
                            f.write(item['sentence'] + '\n')
                    print(f"  {Colors.GREEN}✓{Colors.NC} {split}: {output_file}")
            
            return True
        except Exception as e:
            print(f"  {Colors.YELLOW}⚠{Colors.NC} Could not download PTB: {e}")
            return False
            
    except ImportError:
        print(f"  {Colors.RED}✗{Colors.NC} NLTK not installed. Installing...")
        os.system("pip install nltk datasets -q")
        return download_ptb(cache_dir)

def download_wikitext103(cache_dir: Path) -> bool:
    """Download WikiText-103 dataset."""
    print_header("Downloading WikiText-103")
    
    wt_dir = cache_dir / "wikitext103"
    wt_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"  Dataset: WikiText-103")
    print(f"  Size: ~103M tokens")
    print(f"  Location: {wt_dir}")
    print()
    
    try:
        from datasets import load_dataset
        
        print(f"  Downloading WikiText-103...")
        dataset = load_dataset("wikitext", "wikitext-103-v1", cache_dir=str(cache_dir))
        
        # Save splits
        for split in ['train', 'validation', 'test']:
            if split in dataset:
                output_file = wt_dir / f"{split}.txt"
                with open(output_file, 'w') as f:
                    for item in dataset[split]:
                        if item['text'].strip():
                            f.write(item['text'] + '\n')
                
                file_size = output_file.stat().st_size / 1024 / 1024
                print(f"  {Colors.GREEN}✓{Colors.NC} {split}: {output_file} ({file_size:.1f}MB)")
        
        return True
    except Exception as e:
        print(f"  {Colors.RED}✗{Colors.NC} Error: {e}")
        return False

def download_lambada(cache_dir: Path) -> bool:
    """Download LAMBADA dataset."""
    print_header("Downloading LAMBADA")
    
    lambada_dir = cache_dir / "lambada"
    lambada_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"  Dataset: LAMBADA (Cloze prediction)")
    print(f"  Size: ~10M tokens")
    print(f"  Location: {lambada_dir}")
    print()
    
    try:
        from datasets import load_dataset
        
        print(f"  Downloading LAMBADA...")
        dataset = load_dataset("lambada", cache_dir=str(cache_dir))
        
        # Save splits (lambada has train and test)
        for split in ['train', 'test']:
            if split in dataset:
                output_file = lambada_dir / f"{split}.txt"
                with open(output_file, 'w') as f:
                    for item in dataset[split]:
                        f.write(item['text'] + '\n')
                
                file_size = output_file.stat().st_size / 1024 / 1024
                print(f"  {Colors.GREEN}✓{Colors.NC} {split}: {output_file} ({file_size:.1f}MB)")
        
        return True
    except Exception as e:
        print(f"  {Colors.RED}✗{Colors.NC} Error: {e}")
        return False

def download_ag_news(cache_dir: Path) -> bool:
    """Download AG News dataset."""
    print_header("Downloading AG News")
    
    ag_dir = cache_dir / "ag_news"
    ag_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"  Dataset: AG News (News classification corpus)")
    print(f"  Size: ~100K news articles")
    print(f"  Location: {ag_dir}")
    print()
    
    try:
        from datasets import load_dataset
        
        print(f"  Downloading AG News...")
        dataset = load_dataset("ag_news", cache_dir=str(cache_dir))
        
        # Save splits - AG News uses train/test
        for split in ['train', 'test']:
            if split in dataset:
                output_file = ag_dir / f"{split}.txt"
                with open(output_file, 'w') as f:
                    for item in dataset[split]:
                        # Each item has 'text' and 'label'
                        text = item['text']
                        label = item['label']
                        f.write(f"[{label}] {text}\n")
                
                file_size = output_file.stat().st_size / 1024 / 1024
                print(f"  {Colors.GREEN}✓{Colors.NC} {split}: {output_file} ({file_size:.1f}MB)")
        
        return True
    except Exception as e:
        print(f"  {Colors.RED}✗{Colors.NC} Error: {e}")
        return False

def download_lm1b(cache_dir: Path) -> bool:
    """Download LM1B dataset (requires manual download)."""
    print_header("LM1B Dataset - Manual Download Required")
    
    lm1b_dir = cache_dir / "lm1b"
    lm1b_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"  {Colors.YELLOW}⚠ LM1B requires manual download{Colors.NC}")
    print(f"  Dataset: Google's 1 Billion Word Language Model Benchmark")
    print(f"  Size: ~1 Billion tokens")
    print(f"  Location: {lm1b_dir}")
    print()
    print(f"  Download instructions:")
    print(f"  1. Visit: http://www.statmt.org/lm-benchmark/")
    print(f"  2. Download the .tar files")
    print(f"  3. Extract to: {lm1b_dir}")
    print()
    print(f"  Alternative (via gsutil):")
    print(f"    gsutil -m cp -r gs://lm1b/* {lm1b_dir}/")
    print()
    
    return False

def verify_dataset(dataset_dir: Path) -> bool:
    """Verify dataset files exist."""
    if not dataset_dir.exists():
        return False
    files = list(dataset_dir.glob("*.txt"))
    return len(files) > 0

def main():
    parser = argparse.ArgumentParser(
        description="Download evaluation datasets for ATAT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="/media/scratch/adele/mdlm_fresh/data_cache",
        help="Cache directory for downloads"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="ptb,wikitext103,lambada,ag_news,lm1b",
        help="Comma-separated dataset names to download"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip if already downloaded"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify downloads after completion"
    )
    
    args = parser.parse_args()
    
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    datasets_to_download = [d.strip().lower() for d in args.datasets.split(",")]
    
    print_header("ATAT Evaluation Datasets Download")
    print(f"  Cache directory: {Colors.GREEN}{cache_dir}{Colors.NC}")
    print(f"  Datasets: {Colors.GREEN}{', '.join(datasets_to_download)}{Colors.NC}")
    print()
    
    results = {}
    
    # Download each dataset
    dataset_funcs = {
        "ptb": download_ptb,
        "wikitext103": download_wikitext103,
        "wikitext-103": download_wikitext103,
        "lambada": download_lambada,
        "ag_news": download_ag_news,
        "lm1b": download_lm1b,
    }
    
    for dataset_name in datasets_to_download:
        if dataset_name in dataset_funcs:
            func = dataset_funcs[dataset_name]
            
            # Skip if requested and exists
            if args.skip_existing:
                dataset_dir = cache_dir / dataset_name.replace("-", "")
                if verify_dataset(dataset_dir):
                    print(f"{Colors.YELLOW}Skipping {dataset_name} (already exists){Colors.NC}\n")
                    results[dataset_name] = True
                    continue
            
            success = func(cache_dir)
            results[dataset_name] = success
        else:
            print(f"{Colors.RED}Unknown dataset: {dataset_name}{Colors.NC}")
            results[dataset_name] = False
    
    # Summary
    print_header("Download Summary")
    
    for dataset_name, success in results.items():
        status = f"{Colors.GREEN}✓ Success{Colors.NC}" if success else f"{Colors.RED}✗ Failed{Colors.NC}"
        print(f"  {dataset_name:20} {status}")
    
    print()
    successful = sum(1 for v in results.values() if v)
    total = len(results)
    
    if successful == total:
        print(f"{Colors.GREEN}All datasets downloaded successfully!{Colors.NC}\n")
        return 0
    else:
        print(f"{Colors.YELLOW}{successful}/{total} datasets downloaded{Colors.NC}\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
