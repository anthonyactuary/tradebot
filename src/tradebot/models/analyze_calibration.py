#!/usr/bin/env python3
"""Analyze Platt calibration quality with backtesting metrics.

Computes for each TTE bucket:
- Brier score (raw vs calibrated)
- Accuracy at 0.5 threshold
- Expected Calibration Error (ECE)
- Reliability diagram data

Usage:
    python analyze_calibration.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

# Paths
MODELS_DIR = Path(__file__).parent
CAL_FILE = MODELS_DIR / "platt_multi_tte.json"
CSV_DIR = MODELS_DIR / "calibration_training"


def calibrate_single(p: float, a: float, b: float) -> float:
    """Apply Platt calibration to single probability."""
    import math
    p = max(1e-6, min(1 - 1e-6, p))
    z = math.log(p / (1 - p))
    z_cal = a * z + b
    return 1.0 / (1.0 + math.exp(-z_cal)) if z_cal >= 0 else math.exp(z_cal) / (1.0 + math.exp(z_cal))


def compute_ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """Compute Expected Calibration Error."""
    edges = np.linspace(0, 1, bins + 1)
    ece_sum = 0
    for i in range(bins):
        if i < bins - 1:
            mask = (p >= edges[i]) & (p < edges[i + 1])
        else:
            mask = (p >= edges[i]) & (p <= edges[i + 1])
        if mask.sum() > 0:
            ece_sum += mask.sum() * abs(p[mask].mean() - y[mask].mean())
    return ece_sum / len(p)


def reliability_table(p_raw: np.ndarray, p_cal: np.ndarray, y: np.ndarray, bins: int = 5) -> None:
    """Print reliability comparison table."""
    print(f"\n  {'Bin':<12} {'Count':>6} {'Avg_Raw':>8} {'Avg_Cal':>8} {'Actual':>8} {'Err_Raw':>8} {'Err_Cal':>8}")
    print(f"  {'-'*12} {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    
    edges = np.linspace(0, 1, bins + 1)
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p_raw >= lo) & (p_raw < hi) if i < bins - 1 else (p_raw >= lo) & (p_raw <= hi)
        count = mask.sum()
        if count > 0:
            avg_raw = p_raw[mask].mean()
            avg_cal = p_cal[mask].mean()
            actual = y[mask].mean()
            err_raw = abs(avg_raw - actual)
            err_cal = abs(avg_cal - actual)
            print(f"  [{lo:.1f}-{hi:.1f})    {count:>6} {avg_raw:>8.3f} {avg_cal:>8.3f} {actual:>8.3f} {err_raw:>8.3f} {err_cal:>8.3f}")


def main():
    # Load calibrator
    with open(CAL_FILE) as f:
        cal_data = json.load(f)
    calibrators = {int(k): (v["a"], v["b"]) for k, v in cal_data["calibrators"].items()}
    
    print("=" * 90)
    print("PLATT CALIBRATION BACKTEST ANALYSIS")
    print("=" * 90)
    print(f"{'TTE':>6} {'N':>5} {'Brier_Raw':>10} {'Brier_Cal':>10} {'Improv':>8} {'Acc_Raw':>8} {'Acc_Cal':>8} {'ECE_Raw':>8} {'ECE_Cal':>8}")
    print("-" * 90)
    
    all_results = []
    all_p_raw, all_p_cal, all_y = [], [], []
    
    # Sort by TTE descending
    csv_files = sorted(
        CSV_DIR.glob("platt_training_data_*_tte.csv"),
        key=lambda x: -int(x.stem.split("_")[3])
    )
    
    for f in csv_files:
        parts = f.stem.replace("platt_training_data_", "").replace("_tte", "").split("_")
        tte_max = int(parts[0])
        
        df = pd.read_csv(f)
        p_raw = df["p_yes"].values
        y = df["y"].values
        a, b = calibrators[tte_max]
        
        p_cal = np.array([calibrate_single(p, a, b) for p in p_raw])
        
        all_p_raw.extend(p_raw)
        all_p_cal.extend(p_cal)
        all_y.extend(y)
        
        # Brier scores
        brier_raw = np.mean((p_raw - y) ** 2)
        brier_cal = np.mean((p_cal - y) ** 2)
        improv = (brier_raw - brier_cal) / brier_raw * 100 if brier_raw > 0 else 0
        
        # Accuracy at 0.5 threshold
        acc_raw = np.mean((p_raw >= 0.5) == y)
        acc_cal = np.mean((p_cal >= 0.5) == y)
        
        # ECE
        ece_raw = compute_ece(p_raw, y)
        ece_cal = compute_ece(p_cal, y)
        
        all_results.append({
            "tte": tte_max, "n": len(df),
            "brier_raw": brier_raw, "brier_cal": brier_cal, "improv": improv,
            "acc_raw": acc_raw, "acc_cal": acc_cal,
            "ece_raw": ece_raw, "ece_cal": ece_cal,
        })
        
        print(f"{tte_max:>4}s {len(df):>6} {brier_raw:>10.4f} {brier_cal:>10.4f} {improv:>7.1f}% {acc_raw:>7.1%} {acc_cal:>7.1%} {ece_raw:>8.4f} {ece_cal:>8.4f}")
    
    print("-" * 90)
    
    # Averages
    total_n = sum(r["n"] for r in all_results)
    avg = lambda k: np.mean([r[k] for r in all_results])
    
    print(f"{'AVG':>6} {total_n:>5} {avg('brier_raw'):>10.4f} {avg('brier_cal'):>10.4f} {avg('improv'):>7.1f}% {avg('acc_raw'):>7.1%} {avg('acc_cal'):>7.1%} {avg('ece_raw'):>8.4f} {avg('ece_cal'):>8.4f}")
    
    # Overall stats on pooled data
    all_p_raw = np.array(all_p_raw)
    all_p_cal = np.array(all_p_cal)
    all_y = np.array(all_y)
    
    print("\n" + "=" * 90)
    print("OVERALL SUMMARY (POOLED)")
    print("=" * 90)
    
    brier_raw_all = np.mean((all_p_raw - all_y) ** 2)
    brier_cal_all = np.mean((all_p_cal - all_y) ** 2)
    
    print(f"Total samples: {len(all_y):,}")
    print(f"Base rate (actual YES): {all_y.mean():.1%}")
    print()
    print(f"Brier Score:  {brier_raw_all:.4f} (raw) → {brier_cal_all:.4f} (cal) = {(brier_raw_all-brier_cal_all)/brier_raw_all*100:.1f}% improvement")
    print(f"Accuracy:     {np.mean((all_p_raw >= 0.5) == all_y):.1%} (raw) → {np.mean((all_p_cal >= 0.5) == all_y):.1%} (cal)")
    print(f"ECE:          {compute_ece(all_p_raw, all_y):.4f} (raw) → {compute_ece(all_p_cal, all_y):.4f} (cal)")
    
    # Reliability table
    print("\n" + "-" * 90)
    print("RELIABILITY TABLE (Raw vs Calibrated)")
    print("-" * 90)
    reliability_table(all_p_raw, all_p_cal, all_y, bins=5)
    
    print("\n" + "=" * 90)
    print("⚠️  NOTE: This is IN-SAMPLE analysis (training data).")
    print("    For true out-of-sample backtest, use cross-validation or held-out data.")
    print("=" * 90)


if __name__ == "__main__":
    main()
