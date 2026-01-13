"""Live inference: Kalshi + Coinbase -> 15m direction probability.

This module takes per-poll Kalshi market data (ticker, strike/price_to_beat,
seconds_to_expiry, orderbook best prices) plus Coinbase BTC spot + recent 1m candles,
constructs the same feature vector used in training, and returns:

  p_up = P(BTC close at end of 15m window > window start)

Notes / mapping
- Training label uses K = window start close; in live Kalshi usage we typically map
  K to the market's strike/"price_to_beat".
- S_t uses Coinbase spot price at poll time.
- time_remaining uses Kalshi seconds_to_expiry / 60.

The feature engineering matches `tradebot.tools.btc15m_direction_model`.

Usage (prints JSON per market per poll):
    python -m tradebot.tools.btc15m_live_inference \
    --model-dir src/tradebot/data/btc15m_model_coinbase_80d_purged30 --live
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import math
from dataclasses import asdict
from dataclasses import dataclass
from typing import Any

import httpx

from tradebot.config import Settings
from tradebot.kalshi.client import KalshiClient
from tradebot.tools.kalshi_fees import fees_from_snapshot, kalshi_expected_fee_usd
from tradebot.tools.kalshi_market_poll import MarketSnapshot, poll_once


COINBASE_CANDLES_URL = "https://api.exchange.coinbase.com/products/BTC-USD/candles"


@dataclass(frozen=True)
class InferenceResult:
    poll_utc_iso: str
    ticker: str

    price_to_beat: float | None
    btc_spot_usd: float | None
    seconds_to_expiry: int | None

    contracts: int | None
    taker_fee_yes_usd: float | None
    maker_fee_yes_usd: float | None
    taker_fee_no_usd: float | None
    maker_fee_no_usd: float | None

    features: dict[str, float] | None
    p_up: float | None
    error: str | None = None


def _fees_from_snapshot(*, snap: MarketSnapshot, contracts: int) -> dict[str, float | None]:
    # Back-compat alias (older code imported the underscored helper).
    return fees_from_snapshot(snap=snap, contracts=int(contracts))


def load_model(model_dir: str) -> tuple[Any, list[str]]:
    """Public wrapper around the model loader."""
    return _load_model(model_dir)


def _require_numpy_pandas() -> tuple[Any, Any]:
    try:
        import numpy as np  # type: ignore
        import pandas as pd  # type: ignore

        return np, pd
    except Exception as e:
        raise RuntimeError("Missing required packages. Install: pip install numpy pandas") from e


def _load_model(model_dir: str) -> tuple[Any, list[str]]:
    """Load XGBoost model + feature order from meta.json."""
    _np, pd = _require_numpy_pandas()

    meta_path = f"{model_dir}/meta.json".replace("\\", "/")
    model_path = f"{model_dir}/model.json".replace("\\", "/")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    feature_names = list(meta.get("feature_names") or [])
    if not feature_names:
        raise ValueError(f"No feature_names found in {meta_path}")

    try:
        import xgboost as xgb  # type: ignore

        model = xgb.XGBClassifier()
        model.load_model(model_path)
        return model, feature_names
    except Exception as e:
        raise RuntimeError(
            f"Failed to load xgboost model from {model_path}. "
            "Ensure xgboost is installed and the model dir is correct."
        ) from e


async def fetch_recent_1m_closes(*, limit: int = 6) -> list[float]:
    """Fetch recent 1-minute closes from Coinbase.

    Returns closes oldest->newest.

    Needs at least 6 closes to compute return_5m + vol_5m + trend_5m.
    """
    limit = int(limit)
    if limit < 6:
        limit = 6

    async with httpx.AsyncClient(timeout=10.0) as client:
        headers = {"Accept": "application/json"}
        params = {"granularity": 60}
        resp = await client.get(COINBASE_CANDLES_URL, params=params, headers=headers)
        resp.raise_for_status()
        raw = resp.json() or []

    # Coinbase candles: [time, low, high, open, close, volume], most-recent-first
    closes: list[float] = []
    for c in reversed(raw[:limit]):
        try:
            closes.append(float(c[4]))
        except Exception:
            continue
    return closes


def _trend_slope_last5(closes_5: list[float]) -> float:
    np, _pd = _require_numpy_pandas()
    if len(closes_5) != 5:
        raise ValueError("Need exactly 5 closes")
    yv = np.asarray(closes_5, dtype=float)
    xv = np.arange(5, dtype=float)
    x_mean = float(np.mean(xv))
    y_mean = float(np.mean(yv))
    denom = float(np.sum((xv - x_mean) ** 2))
    if denom <= 0:
        return 0.0
    return float(np.sum((xv - x_mean) * (yv - y_mean)) / denom)


def _vol_5m_from_6_closes(closes_6: list[float]) -> float:
    np, _pd = _require_numpy_pandas()
    if len(closes_6) < 6:
        raise ValueError("Need at least 6 closes")
    closes = closes_6[-6:]
    rets: list[float] = []
    for i in range(1, 6):
        prev = closes[i - 1]
        cur = closes[i]
        if prev <= 0:
            rets.append(0.0)
        else:
            rets.append((cur / prev) - 1.0)
    return float(np.std(np.asarray(rets, dtype=float), ddof=0))


def build_feature_dict(
    *,
    price_to_beat: float,
    btc_spot_usd: float,
    seconds_to_expiry: int,
    recent_closes_1m: list[float],
) -> dict[str, float]:
    """Build the exact feature dict used in training."""

    K = float(price_to_beat)
    S = float(btc_spot_usd)
    if not (K > 0 and math.isfinite(K)):
        raise ValueError("Invalid price_to_beat")
    if not (S > 0 and math.isfinite(S)):
        raise ValueError("Invalid btc_spot_usd")

    time_remaining = float(seconds_to_expiry) / 60.0

    closes = recent_closes_1m
    if len(closes) < 6:
        raise ValueError("Need >= 6 recent closes")
    c_now = float(closes[-1])

    def r(k: int) -> float:
        prev = float(closes[-1 - k])
        if prev <= 0:
            return 0.0
        return (c_now / prev) - 1.0

    delta = (S - K) / K
    feats = {
        "delta": float(delta),
        "abs_delta": float(abs(delta)),
        "return_1m": float(r(1)),
        "return_3m": float(r(3)),
        "return_5m": float(r(5)),
        "vol_5m": float(_vol_5m_from_6_closes(closes[-6:])),
        "trend_5m": float(_trend_slope_last5([float(x) for x in closes[-5:]])),
        "time_remaining": float(time_remaining),
    }
    return feats


def predict_probability(model: Any, feature_names: list[str], features: dict[str, float]) -> float:
    np, _pd = _require_numpy_pandas()
    x = np.asarray([[float(features[n]) for n in feature_names]], dtype=float)
    p = model.predict_proba(x)[:, 1]
    return float(p[0])


async def infer_for_snapshot(
    model: Any,
    feature_names: list[str],
    snap: MarketSnapshot,
    *,
    contracts: int,
) -> InferenceResult:
    try:
        if snap.price_to_beat is None:
            raise ValueError("missing price_to_beat")
        if snap.btc_spot_usd is None:
            raise ValueError("missing btc_spot_usd")
        if snap.seconds_to_expiry is None:
            raise ValueError("missing seconds_to_expiry")

        closes = await fetch_recent_1m_closes(limit=6)
        feats = build_feature_dict(
            price_to_beat=float(snap.price_to_beat),
            btc_spot_usd=float(snap.btc_spot_usd),
            seconds_to_expiry=int(snap.seconds_to_expiry),
            recent_closes_1m=closes,
        )
        p_up = predict_probability(model, feature_names, feats)

        fees = _fees_from_snapshot(snap=snap, contracts=int(contracts))
        return InferenceResult(
            poll_utc_iso=snap.poll_utc_iso,
            ticker=snap.ticker,
            price_to_beat=snap.price_to_beat,
            btc_spot_usd=snap.btc_spot_usd,
            seconds_to_expiry=snap.seconds_to_expiry,

            contracts=int(contracts),
            taker_fee_yes_usd=fees["taker_fee_yes_usd"],
            maker_fee_yes_usd=fees["maker_fee_yes_usd"],
            taker_fee_no_usd=fees["taker_fee_no_usd"],
            maker_fee_no_usd=fees["maker_fee_no_usd"],

            features=feats,
            p_up=p_up,
            error=None,
        )
    except Exception as e:
        return InferenceResult(
            poll_utc_iso=snap.poll_utc_iso,
            ticker=snap.ticker,
            price_to_beat=snap.price_to_beat,
            btc_spot_usd=snap.btc_spot_usd,
            seconds_to_expiry=snap.seconds_to_expiry,

            contracts=int(contracts),
            taker_fee_yes_usd=None,
            maker_fee_yes_usd=None,
            taker_fee_no_usd=None,
            maker_fee_no_usd=None,

            features=None,
            p_up=None,
            error=str(e),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Live BTC 15m direction inference (Kalshi + Coinbase)")
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Directory containing model.json and meta.json from training",
    )
    parser.add_argument("--live", action="store_true", help="Continuously poll and print predictions")
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--duration-seconds", type=float, default=120.0)
    parser.add_argument("--asset", type=str, default="BTC")
    parser.add_argument("--horizon-minutes", type=int, default=60)
    parser.add_argument("--limit-markets", type=int, default=2)
    parser.add_argument("--contracts", type=int, default=1, help="Contracts used for fee estimates")
    args = parser.parse_args()

    model, feature_names = _load_model(str(args.model_dir))

    async def runner() -> None:
        settings = Settings.load()
        client = KalshiClient.from_settings(settings)
        try:
            if not args.live:
                snaps = await poll_once(
                    client,
                    asset=str(args.asset),
                    horizon_minutes=int(args.horizon_minutes),
                    limit_markets=int(args.limit_markets),
                )
                results = [
                    await infer_for_snapshot(model, feature_names, s, contracts=int(args.contracts))
                    for s in snaps
                ]
                print(json.dumps([asdict(r) for r in results], indent=2, sort_keys=True))
                return

            end = dt.datetime.now().timestamp() + float(args.duration_seconds)
            while dt.datetime.now().timestamp() < end:
                snaps = await poll_once(
                    client,
                    asset=str(args.asset),
                    horizon_minutes=int(args.horizon_minutes),
                    limit_markets=int(args.limit_markets),
                )
                results = [
                    await infer_for_snapshot(model, feature_names, s, contracts=int(args.contracts))
                    for s in snaps
                ]
                print(json.dumps([asdict(r) for r in results], sort_keys=True))
                await asyncio.sleep(max(0.1, float(args.poll_seconds)))
        finally:
            await client.aclose()

    asyncio.run(runner())


if __name__ == "__main__":
    main()
