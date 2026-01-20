#!/usr/bin/env python3
"""
Export Platt Training Data.

Exports multiple CSV files for Platt scaling calibration at different TTE ranges.
Creates files in 20-second increments from 720s down to 60s.

Usage:
    python export_platt_training_data.py
"""

from pathlib import Path

import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================
INPUT_FILE = Path(r"c:\Users\slump\Tradebot\tradebot\runs\pred_snapshot_combined.csv")
OUTPUT_DIR = Path(__file__).parent / "calibration_training"

# Models to export: (name, probability column)
MODEL_PROBS = {
    "xgb80": "p_yes",
    "xgb180": "p_yes_xgb_180",
    "lstm": "p_yes_lstm",
}

TTE_MAX = 720
TTE_MIN = 60
TTE_WINDOW = 20


def generate_tte_ranges(tte_max: int, tte_min: int, window: int) -> list[tuple[int, int]]:
    """Generate list of (max, min) TTE ranges in descending order."""
    ranges = []
    current_max = tte_max
    while current_max > tte_min:
        ranges.append((current_max, current_max - window))
        current_max -= window
    return ranges


def export_tte_range(df: pd.DataFrame, tte_max: int, tte_min: int, output_dir: Path) -> dict:
    """Export training data for a single TTE range (all models)."""
    mask = (df["tte_s"] > tte_min) & (df["tte_s"] <= tte_max)
    training_df = df[mask].copy()
    
    if len(training_df) == 0:
        return {"tte_range": f"{tte_max}-{tte_min}", "tickers": 0, "status": "EMPTY"}
    
    # One row per ticker
    training_data = training_df.groupby("ticker").first().reset_index()
    training_data["y"] = (training_data["market_result"] == "yes").astype(int)

    results = {
        "tte_range": f"{tte_max}-{tte_min}",
        "tickers": int(len(training_data)),
        "yes_count": int(training_data["y"].sum()),
        "status": "OK",
        "models": {},
    }

    for model_name, prob_col in MODEL_PROBS.items():
        if prob_col not in training_data.columns:
            results["models"][model_name] = "MISSING_COLUMN"
            continue

        model_df = training_data[["ticker", prob_col, "y"]].copy()
        model_df = model_df.rename(columns={prob_col: "p_yes"})

        output_file = output_dir / f"platt_training_data_{model_name}_{tte_max}_{tte_min}_tte.csv"
        model_df.to_csv(output_file, index=False)
        results["models"][model_name] = "OK"

    return results


def main():
    print("=" * 70)
    print("STEP 2: EXPORT PLATT TRAINING DATA")
    print("=" * 70)
    
    if not INPUT_FILE.exists():
        print(f"[ERROR] Input file not found: {INPUT_FILE}")
        return False
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE, low_memory=False)
    print(f"Loaded {len(df):,} rows, {df['ticker'].nunique():,} unique tickers")
    print(f"Models: {', '.join(MODEL_PROBS.keys())}")
    
    tte_ranges = generate_tte_ranges(TTE_MAX, TTE_MIN, TTE_WINDOW)
    print(f"\nExporting {len(tte_ranges)} TTE ranges ({TTE_MAX}s → {TTE_MIN}s)\n")
    
    results = []
    for tte_max, tte_min in tte_ranges:
        stats = export_tte_range(df, tte_max, tte_min, OUTPUT_DIR)
        results.append(stats)
        status = f"{stats['tickers']:>5} tickers" if stats["status"] == "OK" else "EMPTY"
        models_ok = ", ".join(
            f"{k}:{v}" for k, v in (stats.get("models") or {}).items()
        )
        print(f"  {stats['tte_range']:>8}s  →  {status} | {models_ok}")
    
    successful = [r for r in results if r["status"] == "OK"]
    print(f"\nFiles created: {len(successful)} / {len(tte_ranges)}")
    print(f"Output: {OUTPUT_DIR}")
    return True


if __name__ == "__main__":
    main()
