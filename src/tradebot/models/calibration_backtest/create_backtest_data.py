#!/usr/bin/env python3
"""
Calibration Backtest Dataset Generator.

Creates a dataset with one row per (ticker, TTE bucket) containing:
- p_raw: Raw model probability
- p_cal: Calibrated probability  
- market_p_yes: Market yes price at prediction time
- market_p_no: Market no price at prediction time
- y_true: Actual outcome (1=YES, 0=NO)

This enables EV edge analysis by comparing calibrated probability to market prices.

Usage:
    python calibration_backtest.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================
INPUT_FILE = Path(r"c:\Users\slump\Tradebot\tradebot\runs\pred_snapshot_combined.csv")
CALIBRATOR_FILE = Path(__file__).parent.parent / "platt_multi_tte.json"
OUTPUT_FILE = Path(__file__).parent / "calibration_backtest.csv"

TTE_MAX = 720
TTE_MIN = 60
TTE_WINDOW = 20


def load_calibrators(path: Path) -> dict[int, tuple[float, float]]:
    """Load calibrator parameters from JSON."""
    with open(path) as f:
        data = json.load(f)
    return {int(k): (v["a"], v["b"]) for k, v in data["calibrators"].items()}


def calibrate(p: float, a: float, b: float) -> float:
    """Apply Platt calibration to single probability."""
    import math
    p = max(1e-6, min(1 - 1e-6, p))
    z = math.log(p / (1 - p))
    z_cal = a * z + b
    return 1.0 / (1.0 + math.exp(-z_cal)) if z_cal >= 0 else math.exp(z_cal) / (1.0 + math.exp(z_cal))


def find_tte_bucket(tte: float) -> int | None:
    """Find the TTE bucket max for a given TTE value."""
    # Buckets cover (max-20, max] ranges
    # e.g., TTE=450 belongs to bucket 460 which covers (440, 460]
    if tte < TTE_MIN or tte > TTE_MAX:
        return None
    
    # Find smallest bucket where bucket >= tte
    bucket = TTE_MIN + TTE_WINDOW
    while bucket < tte:
        bucket += TTE_WINDOW
    return min(bucket, TTE_MAX)


def main():
    print("=" * 70)
    print("CALIBRATION BACKTEST DATASET GENERATOR")
    print("=" * 70)
    
    # Load data
    print(f"\nLoading {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE, low_memory=False)
    print(f"Loaded {len(df):,} rows")
    
    # Filter to rows with market results and required columns
    required_cols = ["ticker", "tte_s", "p_yes", "market_p_yes", "market_p_no", "market_result"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"[ERROR] Missing columns: {missing}")
        return
    
    # Filter to settled markets only
    df = df[df["market_result"].isin(["yes", "no"])].copy()
    print(f"Rows with settled results: {len(df):,}")
    
    # Filter to valid TTE range
    df = df[(df["tte_s"] >= TTE_MIN) & (df["tte_s"] <= TTE_MAX)].copy()
    print(f"Rows in TTE range [{TTE_MIN}, {TTE_MAX}]: {len(df):,}")
    
    # Load calibrators
    print(f"\nLoading calibrators from {CALIBRATOR_FILE}...")
    calibrators = load_calibrators(CALIBRATOR_FILE)
    print(f"Loaded {len(calibrators)} TTE bucket calibrators")
    
    # Assign TTE buckets
    df["tte_bucket"] = df["tte_s"].apply(find_tte_bucket)
    df = df[df["tte_bucket"].notna()].copy()
    df["tte_bucket"] = df["tte_bucket"].astype(int)
    
    # For each (ticker, tte_bucket), take ONE row (first occurrence)
    # This ensures we get the market prices at that specific TTE
    print("\nGrouping by (ticker, tte_bucket)...")
    grouped = df.groupby(["ticker", "tte_bucket"]).first().reset_index()
    print(f"Unique (ticker, tte_bucket) combinations: {len(grouped):,}")
    
    # Build backtest dataset
    print("\nBuilding backtest dataset...")
    records = []
    
    for _, row in grouped.iterrows():
        ticker = row["ticker"]
        tte_bucket = row["tte_bucket"]
        p_raw = row["p_yes"]
        market_p_yes = row["market_p_yes"]
        market_p_no = row["market_p_no"]
        y_true = 1 if row["market_result"] == "yes" else 0
        
        # Get calibrator for this TTE bucket
        if tte_bucket not in calibrators:
            continue
        
        a, b = calibrators[tte_bucket]
        p_cal = calibrate(p_raw, a, b)
        
        records.append({
            "ticker": ticker,
            "tte_bucket": tte_bucket,
            "p_raw": round(p_raw, 6),
            "p_cal": round(p_cal, 6),
            "market_p_yes": round(market_p_yes, 4) if pd.notna(market_p_yes) else None,
            "market_p_no": round(market_p_no, 4) if pd.notna(market_p_no) else None,
            "y_true": y_true,
        })
    
    backtest_df = pd.DataFrame(records)
    
    # Summary
    print("\n" + "=" * 70)
    print("BACKTEST DATASET SUMMARY")
    print("=" * 70)
    print(f"Total rows: {len(backtest_df):,}")
    print(f"Unique tickers: {backtest_df['ticker'].nunique()}")
    print(f"TTE buckets: {sorted(backtest_df['tte_bucket'].unique())}")
    print(f"Rows per TTE bucket: ~{len(backtest_df) // backtest_df['tte_bucket'].nunique():.0f}")
    
    # Quick EV preview
    print("\n" + "-" * 70)
    print("EV EDGE PREVIEW (buy YES when p_cal > market_p_yes)")
    print("-" * 70)
    
    valid = backtest_df[backtest_df["market_p_yes"].notna()].copy()
    if len(valid) > 0:
        valid["edge"] = valid["p_cal"] - valid["market_p_yes"]
        valid["would_buy_yes"] = valid["p_cal"] > valid["market_p_yes"]
        valid["would_buy_no"] = (1 - valid["p_cal"]) > valid["market_p_no"]
        
        # EV calculation for YES trades
        yes_trades = valid[valid["would_buy_yes"]]
        if len(yes_trades) > 0:
            # Profit = +1 if win, -cost if lose (simplified to breakeven analysis)
            yes_win_rate = yes_trades["y_true"].mean()
            avg_edge = yes_trades["edge"].mean()
            print(f"YES trades: {len(yes_trades)} opportunities")
            print(f"  Win rate: {yes_win_rate:.1%}")
            print(f"  Avg edge (p_cal - market): {avg_edge:.1%}")
        
        # EV calculation for NO trades
        no_trades = valid[valid["would_buy_no"]]
        if len(no_trades) > 0:
            no_win_rate = 1 - no_trades["y_true"].mean()  # NO wins when y_true=0
            avg_edge_no = (1 - no_trades["p_cal"]) - no_trades["market_p_no"]
            print(f"NO trades: {len(no_trades)} opportunities")
            print(f"  Win rate: {no_win_rate:.1%}")
            print(f"  Avg edge (p_no_cal - market): {avg_edge_no.mean():.1%}")
    
    # Save
    backtest_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved to: {OUTPUT_FILE}")
    
    # Show sample
    print("\n" + "-" * 70)
    print("SAMPLE ROWS")
    print("-" * 70)
    print(backtest_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
