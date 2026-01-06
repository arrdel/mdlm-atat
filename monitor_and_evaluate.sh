#!/bin/bash
# Monitor training progress and run evaluation when complete

echo "========================================================================"
echo "Training Progress Monitor"
echo "========================================================================"
echo ""

while true; do
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Checking training status..."
    
    # Check AR status
    ar_running=$(ps aux | grep -c "train_ar_simple.py.*--max-steps 10000")
    ar_status="❌ Not running"
    if [ $ar_running -gt 1 ]; then
        ar_status="🔄 Running"
        ar_progress=$(tail -1 /media/scratch/adele/mdlm_fresh/outputs/ar_latest.log 2>/dev/null | grep -oP '\d+/10000' || echo "N/A")
        ar_status="🔄 Running ($ar_progress steps)"
    fi
    echo "  AR Transformer: $ar_status"
    
    # Check MDLM status
    mdlm_running=$(ps aux | grep -c "mdlm.main")
    mdlm_status="❌ Not running"
    if [ $mdlm_running -gt 1 ]; then
        mdlm_status="🔄 Running"
        mdlm_progress=$(tail -1 /media/scratch/adele/mdlm_fresh/outputs/mdlm_latest.log 2>/dev/null | grep -oP '\d+/1105174' || echo "N/A")
        mdlm_status="🔄 Running ($mdlm_progress steps)"
    fi
    echo "  MDLM Baseline: $mdlm_status"
    
    echo ""
    
    # Check if both are done
    if [ $ar_running -le 1 ] && [ $mdlm_running -le 1 ]; then
        echo "✓ Both trainings complete!"
        echo ""
        echo "Running evaluation..."
        echo "========================================================================"
        
        cd /home/adelechinda/home/projects/mdlm
        source ~/miniconda3/etc/profile.d/conda.sh
        conda activate mdlm-atat
        python evaluate_all_baselines.py
        
        break
    fi
    
    sleep 30
done
