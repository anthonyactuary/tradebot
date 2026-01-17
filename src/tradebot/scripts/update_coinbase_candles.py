"""Fetch the latest Coinbase 1m candles and merge with existing data.

Usage:
    python -m tradebot.scripts.update_coinbase_candles

This script:
1. Reads the existing btc_1m_coinbase_80d.csv
2. Downloads the last 2 weeks of 1m candles from Coinbase
3. Merges, deduplicates by timestamp, and sorts
4. Writes to a NEW file (does not modify the original)
"""

from __future__ import annotations

import asyncio
import datetime as dt
import os
import sys
import time
from pathlib import Path

import httpx
import pandas as pd

# Allow running from repo root or src directory.
_THIS_FILE = Path(__file__).resolve()
_SRC_ROOT = _THIS_FILE.parents[2]  # .../src
if (_SRC_ROOT / "tradebot").is_dir():
    src_root_str = str(_SRC_ROOT)
    if src_root_str not in sys.path:
        sys.path.insert(0, src_root_str)

COINBASE_CANDLES_URL = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
EXISTING_CSV = DATA_DIR / "btc_1m_coinbase_80d.csv"


async def download_recent_candles(*, days: int = 14) -> list[dict[str, float]]:
    """Download the last `days` worth of 1m candles from Coinbase."""

    granularity = 60
    candles_needed = int(days * 24 * 60)
    max_per_request = 300

    end_ts = int(time.time())
    rows: list[dict[str, float]] = []
    requests_made = 0
    backoff = 0.5

    async with httpx.AsyncClient(timeout=30.0) as client:
        headers = {"Accept": "application/json"}

        while len(rows) < candles_needed:
            start_ts = end_ts - max_per_request * granularity
            params = {"granularity": granularity, "start": start_ts, "end": end_ts}

            try:
                resp = await client.get(COINBASE_CANDLES_URL, params=params, headers=headers)
                if resp.status_code == 429:
                    print(f"  Rate limited, backing off {backoff:.1f}s...")
                    await asyncio.sleep(backoff)
                    backoff = min(10.0, backoff * 1.5)
                    continue
                resp.raise_for_status()
            except Exception as e:
                print(f"  Request error: {e}, retrying...")
                await asyncio.sleep(backoff)
                backoff = min(10.0, backoff * 1.5)
                continue

            raw = resp.json()
            if not raw:
                break

            requests_made += 1
            backoff = 0.5

            # Coinbase format: [time, low, high, open, close, volume]
            for c in raw:
                rows.append(
                    {
                        "timestamp": float(c[0]),
                        "open": float(c[3]),
                        "high": float(c[2]),
                        "low": float(c[1]),
                        "close": float(c[4]),
                        "volume": float(c[5]),
                    }
                )

            # Next page goes earlier
            batch_min_ts = int(min(float(c[0]) for c in raw))
            end_ts = batch_min_ts

            if requests_made % 10 == 0:
                oldest = int(min(r["timestamp"] for r in rows))
                newest = int(max(r["timestamp"] for r in rows))
                covered_days = (newest - oldest) / 86400.0
                print(
                    f"  Progress: requests={requests_made} candles={len(rows)} "
                    f"covered_days~={covered_days:.1f}",
                    flush=True,
                )

            # Be polite to the API
            await asyncio.sleep(0.15)

    return rows


def main() -> None:
    print(f"Reading existing data from: {EXISTING_CSV}")
    if not EXISTING_CSV.exists():
        print(f"ERROR: {EXISTING_CSV} not found")
        sys.exit(1)

    df_existing = pd.read_csv(EXISTING_CSV)
    print(f"  Existing rows: {len(df_existing)}")
    print(f"  Existing range: {dt.datetime.fromtimestamp(df_existing['timestamp'].min())} to {dt.datetime.fromtimestamp(df_existing['timestamp'].max())}")

    print("\nDownloading last 14 days of Coinbase 1m candles...")
    new_rows = asyncio.run(download_recent_candles(days=14))
    print(f"  Downloaded {len(new_rows)} candles")

    if not new_rows:
        print("No new data downloaded, exiting.")
        sys.exit(1)

    df_new = pd.DataFrame(new_rows)
    print(f"  New data range: {dt.datetime.fromtimestamp(df_new['timestamp'].min())} to {dt.datetime.fromtimestamp(df_new['timestamp'].max())}")

    # Merge and deduplicate
    print("\nMerging and deduplicating...")
    df_merged = pd.concat([df_existing, df_new], ignore_index=True)
    df_merged = df_merged.drop_duplicates(subset=["timestamp"], keep="last")
    df_merged = df_merged.sort_values("timestamp").reset_index(drop=True)

    print(f"  Merged rows: {len(df_merged)} (was {len(df_existing)} + {len(new_rows)} before dedupe)")
    print(f"  Merged range: {dt.datetime.fromtimestamp(df_merged['timestamp'].min())} to {dt.datetime.fromtimestamp(df_merged['timestamp'].max())}")

    # Calculate new day coverage
    days_covered = (df_merged["timestamp"].max() - df_merged["timestamp"].min()) / 86400.0
    print(f"  Total coverage: ~{days_covered:.1f} days")

    # Write to new file
    out_name = f"btc_1m_coinbase_{int(days_covered)}d.csv"
    out_path = DATA_DIR / out_name
    df_merged.to_csv(out_path, index=False)
    print(f"\nWrote: {out_path}")
    print("(Original file btc_1m_coinbase_80d.csv unchanged)")


if __name__ == "__main__":
    main()
