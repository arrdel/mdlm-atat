#!/usr/bin/env python3
"""
Phase 1A Implementation Verification

Checks that all components are properly configured for Phase 1A ablation study:
1. Config files exist for all 4 variants
2. ImportanceEstimator supports all 4 modes
3. Training script can load configs
4. Directory structure is ready
"""

import sys
from pathlib import Path
import yaml

# Configuration
PROJECT_ROOT = Path("/home/adelechinda/home/projects/mdlm/mdlm_atat")
VARIANTS = ["full", "frequency_only", "learned_only", "uniform"]
EXPECTED_CONFIGS = [
    "phase1a_base.yaml",
    "phase1a_full.yaml",
    "phase1a_frequency_only.yaml",
    "phase1a_learned_only.yaml",
    "phase1a_uniform.yaml",
]
EXPECTED_MODES = ["full", "frequency_only", "learned_only", "none"]

def check_config_files():
    """Check that all config files exist."""
    print("\n" + "="*80)
    print("Checking config files...")
    print("="*80)
    
    config_dir = PROJECT_ROOT / "configs" / "atat"
    all_exist = True
    
    for config_file in EXPECTED_CONFIGS:
        config_path = config_dir / config_file
        if config_path.exists():
            print(f"✓ {config_file}")
        else:
            print(f"✗ {config_file} NOT FOUND")
            all_exist = False
    
    return all_exist

def check_config_content():
    """Check that configs have correct importance_mode settings."""
    print("\n" + "="*80)
    print("Checking config content...")
    print("="*80)
    
    config_dir = PROJECT_ROOT / "configs" / "atat"
    mode_map = {
        "phase1a_full.yaml": "full",
        "phase1a_frequency_only.yaml": "frequency_only",
        "phase1a_learned_only.yaml": "learned_only",
        "phase1a_uniform.yaml": "none",
    }
    
    all_correct = True
    
    for config_file, expected_mode in mode_map.items():
        config_path = config_dir / config_file
        if not config_path.exists():
            print(f"✗ {config_file} NOT FOUND")
            all_correct = False
            continue
        
        with open(config_path) as f:
            config = yaml.safe_load(f)
            
        # Check importance_mode in config
        try:
            actual_mode = config.get("model", {}).get("importance_mode")
            if actual_mode == expected_mode:
                print(f"✓ {config_file}: importance_mode = '{actual_mode}'")
            else:
                print(f"✗ {config_file}: importance_mode = '{actual_mode}' (expected '{expected_mode}')")
                all_correct = False
        except Exception as e:
            print(f"✗ {config_file}: Error parsing - {e}")
            all_correct = False
    
    return all_correct

def check_importance_estimator():
    """Check that ImportanceEstimator supports all modes."""
    print("\n" + "="*80)
    print("Checking ImportanceEstimator implementation...")
    print("="*80)
    
    try:
        # Try to import the module
        sys.path.insert(0, str(PROJECT_ROOT))
        from atat.importance_estimator import ImportanceEstimator
        
        # Check that __init__ accepts mode parameter
        import inspect
        sig = inspect.signature(ImportanceEstimator.__init__)
        if 'mode' in sig.parameters:
            print("✓ ImportanceEstimator.__init__ accepts 'mode' parameter")
        else:
            print("✗ ImportanceEstimator.__init__ missing 'mode' parameter")
            return False
        
        # Try to instantiate with each mode
        all_modes_work = True
        for mode in EXPECTED_MODES:
            try:
                model = ImportanceEstimator(
                    vocab_size=50257,  # GPT-2 vocab size
                    hidden_dim=256,
                    num_layers=2,
                    mode=mode,
                )
                print(f"✓ Mode '{mode}' initialized successfully")
            except Exception as e:
                print(f"✗ Mode '{mode}' failed: {e}")
                all_modes_work = False
        
        return all_modes_work
        
    except ImportError as e:
        print(f"✗ Could not import ImportanceEstimator: {e}")
        return False

def check_scripts():
    """Check that Phase 1A scripts exist."""
    print("\n" + "="*80)
    print("Checking Phase 1A scripts...")
    print("="*80)
    
    scripts = [
        "run_phase1a.sh",
        "eval_phase1a.py",
    ]
    
    scripts_dir = PROJECT_ROOT / "scripts"
    all_exist = True
    
    for script in scripts:
        script_path = scripts_dir / script
        if script_path.exists():
            print(f"✓ {script}")
        else:
            print(f"✗ {script} NOT FOUND")
            all_exist = False
    
    return all_exist

def check_output_directories():
    """Check that output directories are ready."""
    print("\n" + "="*80)
    print("Checking output directories...")
    print("="*80)
    
    output_dirs = [
        "/media/scratch/adele/mdlm_fresh/outputs/phase1a_ablations",
        "/media/scratch/adele/mdlm_fresh/logs/phase1a_ablations",
    ]
    
    # Just check if they can be created
    from pathlib import Path
    all_ready = True
    
    for dir_path in output_dirs:
        path = Path(dir_path)
        try:
            path.mkdir(parents=True, exist_ok=True)
            print(f"✓ {dir_path} (ready)")
        except Exception as e:
            print(f"✗ {dir_path}: {e}")
            all_ready = False
    
    return all_ready

def print_summary(checks: dict):
    """Print final summary."""
    print("\n" + "="*80)
    print("PHASE 1A IMPLEMENTATION VERIFICATION SUMMARY")
    print("="*80 + "\n")
    
    all_passed = all(checks.values())
    
    for check_name, passed in checks.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {check_name}")
    
    print("\n" + "="*80)
    if all_passed:
        print("✓ All checks passed! Phase 1A is ready to run.")
        print("\nNext step: Start training with:")
        print("  bash /home/adelechinda/home/projects/mdlm/mdlm_atat/scripts/run_phase1a.sh")
    else:
        print("✗ Some checks failed. Please fix the issues above.")
    print("="*80 + "\n")
    
    return all_passed

def main():
    """Run all verification checks."""
    print("\n" + "="*80)
    print("PHASE 1A IMPLEMENTATION VERIFICATION")
    print("="*80)
    
    checks = {
        "Config files exist": check_config_files(),
        "Config content correct": check_config_content(),
        "ImportanceEstimator supports all modes": check_importance_estimator(),
        "Phase 1A scripts exist": check_scripts(),
        "Output directories ready": check_output_directories(),
    }
    
    all_passed = print_summary(checks)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
