"""
Snapshot Logger for recording live market data for replay.

Records snapshots of spot prices and Kalshi orderbooks to a JSONL file
for later replay simulation.

Usage:
    python -m tradebot.arbitrage.snapshot_logger --out data/day1.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from tradebot.config import Settings
from tradebot.kalshi.client import KalshiClient
from tradebot.arbitrage.spot import SyntheticSpot, now_ms
from tradebot.tools.kalshi_market_poll import (
    fetch_coinbase_best_bid_ask,
    fetch_kraken_best_bid_ask,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------
# Configuration
# ---------------------------

@dataclass(frozen=True)
class LoggerConfig:
    """Configuration for snapshot logging."""
    poll_interval_s: float = 0.5
    series_ticker: str = "KXBTC15M"
    min_seconds_to_expiry: int = 60
    max_seconds_to_expiry: int = 900


CONFIG = LoggerConfig()


# ---------------------------
# Snapshot Recording
# ---------------------------

async def fetch_spot_prices() -> tuple[Optional[float], Optional[float]]:
    """Fetch Coinbase and Kraken mid prices in parallel."""
    cb_result, kr_result = await asyncio.gather(
        fetch_coinbase_best_bid_ask(),
        fetch_kraken_best_bid_ask(),
    )
    # Each returns (bid, ask, mid)
    return cb_result[2], kr_result[2]


async def discover_markets(kalshi: KalshiClient) -> List[Dict[str, Any]]:
    """Discover active KXBTC15M markets."""
    now_ts = int(time.time())
    min_close = now_ts + CONFIG.min_seconds_to_expiry
    max_close = now_ts + CONFIG.max_seconds_to_expiry
    
    try:
        resp = await kalshi.get_markets_page(
            limit=50,
            series_ticker=CONFIG.series_ticker,
            mve_filter="exclude",
        )
        
        markets = resp.get("markets", [])
        result = []
        
        for m in markets:
            status = m.get("status", "")
            if status not in {"active", "initialized", "open"}:
                continue
            
            ticker = m.get("ticker", "")
            if not ticker:
                continue
            
            # Parse close time
            close_time_str = m.get("close_time", "")
            if not close_time_str:
                continue
            try:
                from datetime import datetime
                close_dt = datetime.fromisoformat(close_time_str.replace("Z", "+00:00"))
                close_ts_ms = int(close_dt.timestamp() * 1000)
            except Exception:
                continue
            
            # Check if within trading window
            close_ts_s = close_ts_ms // 1000
            if close_ts_s < min_close or close_ts_s > max_close:
                continue
            
            # Parse strike
            strike = m.get("floor_strike") or m.get("custom_strike")
            if strike is None:
                continue
            try:
                strike_f = float(strike)
            except Exception:
                continue
            
            result.append({
                "ticker": ticker,
                "strike": strike_f,
                "expiry_ts_ms": close_ts_ms,
            })
        
        return result
    except Exception as e:
        log.warning("MARKET_DISCOVERY_ERROR error=%s", e)
        return []


async def fetch_orderbook(kalshi: KalshiClient, ticker: str) -> Optional[Dict[str, int]]:
    """Fetch BBO for a market using the market endpoint."""
    try:
        resp = await kalshi.get_market(ticker)
        mkt = resp.get("market", {})
        
        # BBO is provided directly in cents
        yes_bid = mkt.get("yes_bid")
        yes_ask = mkt.get("yes_ask")
        no_bid = mkt.get("no_bid")
        no_ask = mkt.get("no_ask")
        
        return {
            "yes_bid_cents": int(yes_bid) if yes_bid is not None else 1,
            "yes_ask_cents": int(yes_ask) if yes_ask is not None else 99,
            "no_bid_cents": int(no_bid) if no_bid is not None else 1,
            "no_ask_cents": int(no_ask) if no_ask is not None else 99,
        }
    except Exception as e:
        log.warning("ORDERBOOK_ERROR %s error=%s", ticker, e)
        return None


async def record_snapshot(
    kalshi: KalshiClient,
    spot: SyntheticSpot,
    filepath: str,
) -> None:
    """
    Record a single snapshot to the JSONL file.
    
    Fetches current spot prices, discovers active markets, and
    fetches orderbooks for each market.
    """
    ts_ms = now_ms()
    
    # Fetch spot prices
    cb_mid, kr_mid = await fetch_spot_prices()
    if cb_mid is not None:
        spot.update_coinbase_mid(cb_mid, ts_ms)
    if kr_mid is not None:
        spot.update_kraken_mid(kr_mid, ts_ms)
    
    # Discover markets
    markets = await discover_markets(kalshi)
    
    # Fetch orderbooks for each market
    market_snapshots = []
    for m in markets:
        book = await fetch_orderbook(kalshi, m["ticker"])
        if book is None:
            continue
        
        market_snapshots.append({
            "ticker": m["ticker"],
            "strike": m["strike"],
            "expiry_ts_ms": m["expiry_ts_ms"],
            **book,
        })
    
    # Build snapshot
    snapshot = {
        "timestamp_ms": ts_ms,
        "coinbase_mid": cb_mid,
        "kraken_mid": kr_mid,
        "markets": market_snapshots,
    }
    
    # Append to file
    with open(filepath, "a") as f:
        f.write(json.dumps(snapshot) + "\n")
    
    log.debug(
        "SNAPSHOT ts=%d cb=%.2f kr=%.2f markets=%d",
        ts_ms,
        cb_mid or 0,
        kr_mid or 0,
        len(market_snapshots),
    )


async def run_logger(filepath: str, duration_s: Optional[int] = None) -> None:
    """Run the snapshot logger loop."""
    log.info("Starting snapshot logger -> %s", filepath)
    log.info("Poll interval: %.2fs", CONFIG.poll_interval_s)
    if duration_s:
        log.info("Duration: %ds", duration_s)
    
    # Ensure output directory exists
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize components
    spot = SyntheticSpot()
    settings = Settings.load()
    kalshi = KalshiClient.from_settings(settings)
    
    start_ts = time.time()
    snapshot_count = 0
    
    try:
        while True:
            if duration_s and (time.time() - start_ts) >= duration_s:
                break
            
            await record_snapshot(kalshi, spot, filepath)
            snapshot_count += 1
            
            if snapshot_count % 100 == 0:
                log.info("Recorded %d snapshots", snapshot_count)
            
            await asyncio.sleep(CONFIG.poll_interval_s)
    
    except KeyboardInterrupt:
        log.info("Stopped by user")
    
    finally:
        await kalshi.aclose()
        log.info("Total snapshots recorded: %d", snapshot_count)


# ---------------------------
# CLI
# ---------------------------

async def main() -> None:
    parser = argparse.ArgumentParser(description="Record market snapshots for replay")
    parser.add_argument("--out", required=True, help="Output JSONL file path")
    parser.add_argument("--duration", type=int, help="Recording duration in seconds (default: unlimited)")
    parser.add_argument("--interval", type=float, default=0.5, help="Poll interval in seconds")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Update config if interval provided
    global CONFIG
    if args.interval != 0.5:
        CONFIG = LoggerConfig(poll_interval_s=args.interval)
    
    await run_logger(args.out, args.duration)


if __name__ == "__main__":
    asyncio.run(main())
