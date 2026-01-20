#!/usr/bin/env python3
"""
Full Calibration Training Pipeline.

Runs all steps to train Platt calibrators:
1. Combine PRED_SNAPSHOT logs and fetch market results
2. Export training CSVs for each TTE bucket
3. Train calibrators and save to platt_multi_tte.json
4. Analyze calibration quality

Usage:
    python train_calibration.py          # Run all steps
    python train_calibration.py --skip-combine  # Skip step 1 if data already combined
"""

import argparse
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))


def main():
    parser = argparse.ArgumentParser(description="Train Platt calibration pipeline")
    parser.add_argument("--skip-combine", action="store_true", help="Skip combining pred snapshots (use existing)")
    args = parser.parse_args()
    
    print("=" * 70)
    print("PLATT CALIBRATION TRAINING PIPELINE")
    print("=" * 70)
    
    # Step 1: Combine pred snapshots
    if not args.skip_combine:
        print("\n")
        from combine_pred_snapshots import main as combine_main
        result = combine_main()
        if result is None:
            print("[ERROR] Step 1 failed")
            sys.exit(1)
    else:
        print("\n[SKIP] Step 1: Using existing pred_snapshot_combined.csv")
    
    # Step 2: Export training CSVs
    print("\n")
    from export_platt_training_data import main as export_main
    if not export_main():
        print("[ERROR] Step 2 failed")
        sys.exit(1)
    
    # Step 3: Train calibrators
    print("\n" + "=" * 70)
    print("STEP 3: TRAIN PLATT CALIBRATORS")
    print("=" * 70)
    
    from prob_calibration_platt import MultiTTECalibrator
    
    csv_dir = Path(__file__).parent / "calibration_training"
    model_names = ["xgb80", "xgb180", "lstm"]
    output_files: list[Path] = []
    for model_name in model_names:
        output_file = Path(__file__).parent / f"platt_multi_tte_{model_name}.json"
        multi_cal = MultiTTECalibrator.train_from_csvs(csv_dir, model_name=model_name)
        multi_cal.save(output_file)
        output_files.append(output_file)
        print(f"\nSaved to: {output_file}")
    
    # Step 4: Analyze
    print("\n")
    from analyze_calibration import main as analyze_main
    for model_name in model_names:
        print("\n" + "=" * 70)
        print(f"ANALYZE CALIBRATION: {model_name}")
        print("=" * 70)
        analyze_main(model_name=model_name)
    
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    for out in output_files:
        print(f"Calibrator saved to: {out}")
    print("\nUsage in your bot:")
    print("  from tradebot.models.prob_calibration_platt import MultiTTECalibrator")
    print("  cal = MultiTTECalibrator.load('path/to/platt_multi_tte_xgb80.json')")
    print("  p_calibrated = cal.predict(p_raw=0.65, tte=450)")


if __name__ == "__main__":
    main()
