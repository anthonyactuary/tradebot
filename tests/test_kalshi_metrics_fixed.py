import pandas as pd

from analytics.kalshi_metrics_fixed import build_per_market_metrics, compute_settlement_accounting


def test_paired_payout_revenue_excludes_pairing() -> None:
    # yes_count=no_count=3 implies 3 paired contracts pay out $1 each.
    # If revenue is 0 (because Kalshi revenue excludes paired payout),
    # gross payout should still be $3.
    settlements = pd.DataFrame(
        [
            {
                "ticker": "TEST-1",
                "market_result": "yes",
                "yes_count": 3,
                "no_count": 3,
                "yes_total_cost": 0,
                "no_total_cost": 0,
                "revenue": 0,
                "fee_cost": 0.0,
            }
        ]
    )

    out = compute_settlement_accounting(settlements)
    assert out.shape[0] == 1
    assert float(out.loc[0, "gross_payout_usd"]) == 3.0
    assert int(out.loc[0, "paired_count"]) == 3


def test_early_sell_saved_is_abs_realized_pnl() -> None:
    # Saved is defined as abs(realized_pnl_usd) for early-sold tickers.
    settlements = pd.DataFrame(
        [
            {
                "ticker": "T1",
                "market_result": "no",
                "yes_count": 0,
                "no_count": 0,
                "yes_total_cost": 35,
                "no_total_cost": 0,
                "revenue": 0,
                "fee_cost": 0.0,
                "settled_time": "2026-01-13T00:10:00Z",
            }
        ]
    )
    fills = pd.DataFrame(
        [
            {
                "ticker": "T1",
                "created_time": "2026-01-13T00:00:30Z",
                "side": "yes",
                "action": "buy",
                "count": 2,
                "price": 0.20,
            },
            {
                "ticker": "T1",
                "created_time": "2026-01-13T00:01:00Z",
                "side": "yes",
                "action": "sell",
                "count": 2,
                "price": 0.40,
            },
        ]
    )

    out = build_per_market_metrics(settlements=settlements, fills=fills)
    assert out.shape[0] == 1
    assert int(out.loc[0, "early_sell_contracts"]) == 2
    assert float(out.loc[0, "realized_pnl_usd"]) == -0.35
    assert float(out.loc[0, "early_sell_saved_usd"]) == 0.35


def test_early_sell_lost_is_missed_hold_profit() -> None:
    # Lost is defined as missed upside = exit_qty * (1 - entry_vwap) when outcome == entry_side.
    settlements = pd.DataFrame(
        [
            {
                "ticker": "T1",
                "market_result": "yes",
                "yes_count": 0,
                "no_count": 0,
                "yes_total_cost": 0,
                "no_total_cost": 0,
                "revenue": 0,
                "fee_cost": 0.0,
                "settled_time": "2026-01-13T00:10:00Z",
            }
        ]
    )
    fills = pd.DataFrame(
        [
            {
                "ticker": "T1",
                "created_time": "2026-01-13T00:00:30Z",
                "side": "yes",
                "action": "buy",
                "count": 2,
                "price": 0.20,
            },
            {
                "ticker": "T1",
                "created_time": "2026-01-13T00:01:00Z",
                "side": "yes",
                "action": "sell",
                "count": 2,
                "price": 0.40,
            },
        ]
    )

    out = build_per_market_metrics(settlements=settlements, fills=fills)
    assert out.shape[0] == 1
    assert int(out.loc[0, "early_sell_contracts"]) == 2
    assert abs(float(out.loc[0, "early_sell_lost_usd"]) - 1.6) < 1e-9
    assert float(out.loc[0, "early_sell_saved_usd"]) == 0.0


def test_early_sell_uses_no_price_fixed_for_no_side_buys() -> None:
    # In runs/kalshi_fills.csv, price is YES price even when side==NO.
    # For NO buys, we must use no_price_fixed.
    settlements = pd.DataFrame(
        [
            {
                "ticker": "T1",
                "market_result": "no",
                "yes_count": 0,
                "no_count": 0,
                "yes_total_cost": 0,
                "no_total_cost": 0,
                "revenue": 0,
                "fee_cost": 0.0,
                "settled_time": "2026-01-13T00:10:00Z",
            }
        ]
    )
    fills = pd.DataFrame(
        [
            {
                "ticker": "T1",
                "created_time": "2026-01-13T00:00:10Z",
                "side": "no",
                "action": "buy",
                "count": 2,
                "price": 0.29,
                "yes_price_fixed": 0.29,
                "no_price_fixed": 0.71,
            },
            {
                "ticker": "T1",
                "created_time": "2026-01-13T00:00:20Z",
                "side": "no",
                "action": "buy",
                "count": 1,
                "price": 0.37,
                "yes_price_fixed": 0.37,
                "no_price_fixed": 0.63,
            },
            {
                "ticker": "T1",
                "created_time": "2026-01-13T00:01:00Z",
                "side": "yes",
                "action": "sell",
                "count": 3,
                "price": 0.49,
                "yes_price_fixed": 0.49,
                "no_price_fixed": 0.51,
            },
        ]
    )

    out = build_per_market_metrics(settlements=settlements, fills=fills)
    assert out.shape[0] == 1
    assert int(out.loc[0, "early_sell_contracts"]) == 3
    # entry_vwap = (2*0.71 + 1*0.63) / 3 = 0.683333..., missed hold profit = 3*(1-entry_vwap)=0.95
    assert abs(float(out.loc[0, "early_sell_lost_usd"]) - 0.95) < 1e-9
