#!/usr/bin/env python3
"""
Combine PRED_SNAPSHOT events from multiple CSV files and enrich with market results.

Usage:
    python combine_pred_snapshots.py
"""

import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

# ============================================================
# CONFIGURATION - Add/remove files here
# ============================================================
INPUT_FILES = [
    r"c:\Users\slump\Tradebot\tradebot\runs\btc15m_live_v2_20260118_073256.csv",
    r"c:\Users\slump\Tradebot\tradebot\runs\btc15m_live_v2_20260118_162556.csv",
    r"c:\Users\slump\Tradebot\tradebot\runs\btc15m_live_v2_20260118_225714.csv",
    r"c:\Users\slump\Tradebot\tradebot\runs\btc15m_live_v2_20260119_042107.csv",
    r"c:\Users\slump\Tradebot\tradebot\runs\btc15m_live_v2_20260118_023821.csv",
    r"c:\Users\slump\Tradebot\tradebot\runs\btc15m_live_v2_20260118_011843.csv",
    r"c:\Users\slump\Tradebot\tradebot\runs\btc15m_live_v2_20260119_045709.csv",
    r"c:\Users\slump\Tradebot\tradebot\runs\btc15m_live_v2_20260119_153247.csv",
    r"c:\Users\slump\Tradebot\tradebot\runs\btc15m_live_v2_20260119_195642.csv",
    r"c:\Users\slump\Tradebot\tradebot\runs\btc15m_live_v2_20260120_002945.csv",
    r"c:\Users\slump\Tradebot\tradebot\runs\btc15m_live_v2_20260120_053948.csv",
    # Add more files below:
]

OUTPUT_FILE = Path(r"c:\Users\slump\Tradebot\tradebot\runs\pred_snapshot_combined.csv")
KALSHI_API_BASE = "https://api.elections.kalshi.com/trade-api/v2"


def load_and_filter_pred_snapshots(file_path: str) -> pd.DataFrame:
    """Load a CSV file and filter to only PRED_SNAPSHOT events."""
    print(f"Loading: {file_path}")
    df = pd.read_csv(file_path, low_memory=False)
    filtered = df[df["event"] == "PRED_SNAPSHOT"].copy()
    print(f"  -> {len(filtered)} PRED_SNAPSHOT rows (of {len(df)} total)")
    return filtered


def get_market_result(ticker: str, cache: dict) -> Optional[str]:
    """Fetch market result from Kalshi API."""
    if ticker in cache:
        return cache[ticker]
    
    try:
        url = f"{KALSHI_API_BASE}/markets/{ticker}"
        resp = requests.get(url, timeout=10)
        
        if resp.status_code == 200:
            data = resp.json()
            market = data.get("market", {})
            result = market.get("result")
            status = market.get("status")
            
            if status in ("settled", "finalized") and result in ("yes", "no"):
                cache[ticker] = result
                return result
        elif resp.status_code == 404:
            print(f"  [WARN] Market not found: {ticker}")
        
        cache[ticker] = None
        return None
            
    except Exception as e:
        print(f"  [ERROR] Failed to fetch {ticker}: {e}")
        cache[ticker] = None
        return None


def main():
    print("=" * 60)
    print("STEP 1: COMBINE PRED_SNAPSHOTS")
    print("=" * 60)
    
    dfs = []
    for file_path in INPUT_FILES:
        if not Path(file_path).exists():
            print(f"[WARN] File not found, skipping: {file_path}")
            continue
        dfs.append(load_and_filter_pred_snapshots(file_path))
    
    if not dfs:
        print("[ERROR] No valid files found!")
        return None
    
    combined = pd.concat(dfs, ignore_index=True)
    print(f"\nCombined: {len(combined)} total PRED_SNAPSHOT rows")
    
    tickers = combined["ticker"].dropna().unique()
    print(f"Unique tickers: {len(tickers)}")
    
    print("\nFetching market results from Kalshi API...")
    result_cache: dict = {}
    
    for i, ticker in enumerate(tickers):
        get_market_result(ticker, result_cache)
        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{len(tickers)} tickers...")
        time.sleep(0.1)
    
    combined["market_result"] = combined["ticker"].map(result_cache)
    
    settled = combined["market_result"].notna().sum()
    print(f"\nResults: {settled} rows with results, {(combined['market_result'] == 'yes').sum()} YES, {(combined['market_result'] == 'no').sum()} NO")
    
    combined.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved to: {OUTPUT_FILE}")
    return OUTPUT_FILE


if __name__ == "__main__":
    main()
