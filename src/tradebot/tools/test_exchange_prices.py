"""Test script to compare Coinbase vs Kraken BTC prices.

This helps assess how well Coinbase alone approximates CF Benchmarks BRTI,
which is a composite of multiple exchanges including both Coinbase and Kraken.
"""

import asyncio
import time
import httpx


COINBASE_BOOK_URL = "https://api.exchange.coinbase.com/products/BTC-USD/book"
KRAKEN_TICKER_URL = "https://api.kraken.com/0/public/Ticker"


async def fetch_coinbase_mid() -> tuple[float | None, float | None, float | None]:
    """Fetch Coinbase bid, ask, mid."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(COINBASE_BOOK_URL, params={"level": 1})
            resp.raise_for_status()
            data = resp.json() or {}

        bids = data.get("bids") or []
        asks = data.get("asks") or []

        best_bid = float(bids[0][0]) if bids else None
        best_ask = float(asks[0][0]) if asks else None

        mid = None
        if best_bid is not None and best_ask is not None:
            mid = (best_bid + best_ask) / 2.0

        return (best_bid, best_ask, mid)
    except Exception as e:
        print(f"Coinbase error: {e}")
        return (None, None, None)


async def fetch_kraken_mid() -> tuple[float | None, float | None, float | None]:
    """Fetch Kraken bid, ask, mid."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(KRAKEN_TICKER_URL, params={"pair": "XBTUSD"})
            resp.raise_for_status()
            data = resp.json()

        if data.get("error"):
            print(f"Kraken API error: {data['error']}")
            return (None, None, None)

        result = data.get("result", {}).get("XXBTZUSD", {})
        
        # Kraken format: "a" = [ask_price, whole_lot_volume, lot_volume]
        #                "b" = [bid_price, whole_lot_volume, lot_volume]
        ask_data = result.get("a", [])
        bid_data = result.get("b", [])

        best_ask = float(ask_data[0]) if ask_data else None
        best_bid = float(bid_data[0]) if bid_data else None

        mid = None
        if best_bid is not None and best_ask is not None:
            mid = (best_bid + best_ask) / 2.0

        return (best_bid, best_ask, mid)
    except Exception as e:
        print(f"Kraken error: {e}")
        return (None, None, None)


async def compare_once() -> dict:
    """Fetch from both exchanges and compare."""
    cb_bid, cb_ask, cb_mid = await fetch_coinbase_mid()
    kr_bid, kr_ask, kr_mid = await fetch_kraken_mid()

    result = {
        "coinbase_bid": cb_bid,
        "coinbase_ask": cb_ask,
        "coinbase_mid": cb_mid,
        "kraken_bid": kr_bid,
        "kraken_ask": kr_ask,
        "kraken_mid": kr_mid,
        "composite_mid": None,
        "diff_usd": None,
        "diff_pct": None,
    }

    if cb_mid is not None and kr_mid is not None:
        result["composite_mid"] = (cb_mid + kr_mid) / 2.0
        result["diff_usd"] = cb_mid - kr_mid
        result["diff_pct"] = (cb_mid - kr_mid) / cb_mid * 100.0

    return result


async def run_comparison(duration_seconds: int = 60, interval_seconds: float = 2.0):
    """Run comparison for a duration, collecting stats."""
    print("=" * 80)
    print("COINBASE vs KRAKEN BTC PRICE COMPARISON")
    print("=" * 80)
    print()

    samples = []
    start = time.time()

    while (time.time() - start) < duration_seconds:
        data = await compare_once()
        samples.append(data)

        if data["coinbase_mid"] and data["kraken_mid"]:
            print(
                f"CB: ${data['coinbase_mid']:,.2f}  |  "
                f"KR: ${data['kraken_mid']:,.2f}  |  "
                f"Composite: ${data['composite_mid']:,.2f}  |  "
                f"Diff: ${data['diff_usd']:+.2f} ({data['diff_pct']:+.4f}%)"
            )
        else:
            print(f"Missing data: CB={data['coinbase_mid']} KR={data['kraken_mid']}")

        await asyncio.sleep(interval_seconds)

    # Summary stats
    valid = [s for s in samples if s["diff_usd"] is not None]
    if valid:
        diffs = [s["diff_usd"] for s in valid]
        pcts = [s["diff_pct"] for s in valid]

        print()
        print("=" * 80)
        print("SUMMARY STATISTICS")
        print("=" * 80)
        print(f"Samples collected: {len(valid)}")
        print()
        print(f"Absolute difference (CB - KR):")
        print(f"  Mean:   ${sum(diffs) / len(diffs):+.2f}")
        print(f"  Min:    ${min(diffs):+.2f}")
        print(f"  Max:    ${max(diffs):+.2f}")
        print(f"  StdDev: ${(sum((d - sum(diffs)/len(diffs))**2 for d in diffs) / len(diffs)) ** 0.5:.2f}")
        print()
        print(f"Percentage difference:")
        print(f"  Mean:   {sum(pcts) / len(pcts):+.4f}%")
        print(f"  Min:    {min(pcts):+.4f}%")
        print(f"  Max:    {max(pcts):+.4f}%")
        print()
        
        # Check typical spread
        cb_spreads = [s["coinbase_ask"] - s["coinbase_bid"] for s in valid if s["coinbase_ask"] and s["coinbase_bid"]]
        kr_spreads = [s["kraken_ask"] - s["kraken_bid"] for s in valid if s["kraken_ask"] and s["kraken_bid"]]
        
        if cb_spreads:
            print(f"Coinbase avg spread: ${sum(cb_spreads) / len(cb_spreads):.2f}")
        if kr_spreads:
            print(f"Kraken avg spread:   ${sum(kr_spreads) / len(kr_spreads):.2f}")
        
        print()
        print("RECOMMENDATION:")
        avg_diff = abs(sum(diffs) / len(diffs))
        if avg_diff < 10:
            print("  Coinbase alone is very close to Kraken (<$10 avg diff).")
            print("  Adding Kraken would provide marginal improvement.")
        elif avg_diff < 50:
            print("  Moderate difference ($10-50). Adding Kraken composite")
            print("  would improve accuracy, especially near expiry.")
        else:
            print("  Significant divergence (>$50). Strongly recommend")
            print("  using a composite of both exchanges.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=60, help="Duration in seconds")
    parser.add_argument("--interval", type=float, default=2.0, help="Interval between samples")
    args = parser.parse_args()

    asyncio.run(run_comparison(duration_seconds=args.duration, interval_seconds=args.interval))
