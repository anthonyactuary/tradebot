from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd  # type: ignore


@dataclass(frozen=True)
class MarketKpis:
    markets: int
    win_rate: float | None
    net_pnl_usd: float
    total_fees_usd: float

    profit_factor: float | None
    expectancy: float | None

    sell_markets: int
    sell_win_rate: float | None
    payout_lost_usd: float
    payout_saved_usd: float


def build_market_pnl(*, fills: "pd.DataFrame", settlements: "pd.DataFrame") -> "pd.DataFrame":
    """Compute per-market PnL using fills + settlement revenue/fees.

    Per user spec:
    PnL = revenue_usd + sell_value_usd - buy_cost_usd - fee_cost_usd

    Join key: market_ticker
    """

    import pandas as pd  # type: ignore

    if fills is None:
        fills = pd.DataFrame()
    if settlements is None:
        settlements = pd.DataFrame()

    if fills.empty:
        return pd.DataFrame(
            columns=[
                "market_ticker",
                "entry_side",
                "market_result",
                "buy_contracts",
                "sell_contracts",
                "buy_cost_usd",
                "sell_value_usd",
                "revenue_usd",
                "fee_cost_usd",
                "market_pnl_usd",
                "settled_ts",
                "sold_out",
                "sell_win",
                "payout_lost_usd",
                "payout_saved_usd",
            ]
        )

    f = fills.copy()
    if "market_ticker" not in f.columns:
        f["market_ticker"] = f.get("ticker")

    f["ts"] = pd.to_datetime(f.get("ts"), utc=True, errors="coerce")
    f["contracts"] = pd.to_numeric(f.get("contracts"), errors="coerce").fillna(0).astype(int)
    f["price_cents"] = pd.to_numeric(f.get("price_cents"), errors="coerce").fillna(0).astype(float)

    f["price_usd"] = f["price_cents"] / 100.0
    f["notional_usd"] = f["price_usd"] * f["contracts"].astype(float)

    f["action"] = f.get("action").astype(str).str.lower().str.strip()
    f["side"] = f.get("side").astype(str).str.upper().str.strip()
    f["market_ticker"] = f["market_ticker"].astype(str)

    f = f[(f["contracts"] > 0) & (f["market_ticker"].astype(str) != "")].copy()

    # First buy side per market (entry_side)
    buys_only = f[f["action"] == "buy"].sort_values(["market_ticker", "ts"])
    first_buy_side = (
        buys_only.groupby("market_ticker", as_index=True)["side"].first().rename("entry_side")
    )

    grouped = f.groupby(["market_ticker", "action"], as_index=False).agg(
        contracts=("contracts", "sum"),
        notional_usd=("notional_usd", "sum"),
    )

    buy = grouped[grouped["action"] == "buy"].set_index("market_ticker")
    sell = grouped[grouped["action"] == "sell"].set_index("market_ticker")

    base = pd.DataFrame(index=sorted(set(f["market_ticker"])) )
    base.index.name = "market_ticker"

    base["buy_contracts"] = buy["contracts"] if not buy.empty else 0
    base["buy_cost_usd"] = buy["notional_usd"] if not buy.empty else 0.0
    base["sell_contracts"] = sell["contracts"] if not sell.empty else 0
    base["sell_value_usd"] = sell["notional_usd"] if not sell.empty else 0.0

    base = base.fillna({
        "buy_contracts": 0,
        "buy_cost_usd": 0.0,
        "sell_contracts": 0,
        "sell_value_usd": 0.0,
    })

    base["entry_side"] = first_buy_side

    s = settlements.copy()
    if not s.empty:
        if "market_ticker" not in s.columns:
            s["market_ticker"] = s.get("ticker")
        s["market_ticker"] = s["market_ticker"].astype(str)
        s["result"] = s.get("result").astype(str).str.upper().str.strip()
        s["settled_ts"] = pd.to_datetime(s.get("settled_ts"), utc=True, errors="coerce")
        s["revenue_usd"] = pd.to_numeric(s.get("revenue_usd"), errors="coerce").fillna(0.0).astype(float)
        s["fee_cost_usd"] = pd.to_numeric(s.get("fee_cost_usd"), errors="coerce").fillna(0.0).astype(float)
        s["payout_per_contract_usd"] = (
            pd.to_numeric(s.get("payout_per_contract_usd"), errors="coerce").fillna(1.0).astype(float)
        )

        s = s.drop_duplicates(subset=["market_ticker"], keep="first")
        s = s.set_index("market_ticker")

        base["market_result"] = s.get("result")
        base["revenue_usd"] = s.get("revenue_usd").fillna(0.0)
        base["fee_cost_usd"] = s.get("fee_cost_usd").fillna(0.0)
        base["settled_ts"] = s.get("settled_ts")
        base["payout_per_contract_usd"] = s.get("payout_per_contract_usd").fillna(1.0)
    else:
        base["market_result"] = None
        base["revenue_usd"] = 0.0
        base["fee_cost_usd"] = 0.0
        base["settled_ts"] = pd.NaT
        base["payout_per_contract_usd"] = 1.0

    base["market_pnl_usd"] = (
        base["revenue_usd"].astype(float)
        + base["sell_value_usd"].astype(float)
        - base["buy_cost_usd"].astype(float)
        - base["fee_cost_usd"].astype(float)
    )

    # Sell analysis
    base["net_contracts"] = base["buy_contracts"].astype(int) - base["sell_contracts"].astype(int)
    base["sold_out"] = (base["buy_contracts"].astype(int) > 0) & (base["sell_contracts"].astype(int) > 0) & (
        base["net_contracts"].astype(int) <= 0
    )

    # Determine if selling was "good" based on market outcome vs entry side.
    base["sell_win"] = base["sold_out"] & (base["entry_side"].astype(str).str.upper() != base["market_result"].astype(str).str.upper())

    exited_contracts = base[["buy_contracts", "sell_contracts"]].min(axis=1).astype(float)
    payout_per_contract = pd.to_numeric(base["payout_per_contract_usd"], errors="coerce").fillna(1.0).astype(float)

    would_have_payout = exited_contracts * payout_per_contract
    would_have_payout = would_have_payout.where(
        base["entry_side"].astype(str).str.upper() == base["market_result"].astype(str).str.upper(),
        0.0,
    )

    # Compare sell proceeds vs would-have payout for the exited contracts.
    sell_value = pd.to_numeric(base["sell_value_usd"], errors="coerce").fillna(0.0).astype(float)

    base["payout_lost_usd"] = (would_have_payout - sell_value).clip(lower=0.0)
    base["payout_saved_usd"] = (sell_value - would_have_payout).clip(lower=0.0)

    out = base.reset_index()
    return out[
        [
            "market_ticker",
            "entry_side",
            "market_result",
            "buy_contracts",
            "sell_contracts",
            "buy_cost_usd",
            "sell_value_usd",
            "revenue_usd",
            "fee_cost_usd",
            "market_pnl_usd",
            "settled_ts",
            "sold_out",
            "sell_win",
            "payout_lost_usd",
            "payout_saved_usd",
        ]
    ]


def compute_market_kpis(*, market_pnl: "pd.DataFrame") -> MarketKpis:
    import pandas as pd  # type: ignore

    if market_pnl is None or market_pnl.empty:
        return MarketKpis(
            markets=0,
            win_rate=None,
            net_pnl_usd=0.0,
            total_fees_usd=0.0,
            profit_factor=None,
            expectancy=None,
            sell_markets=0,
            sell_win_rate=None,
            payout_lost_usd=0.0,
            payout_saved_usd=0.0,
        )

    df = market_pnl.copy()
    df["market_pnl_usd"] = pd.to_numeric(df.get("market_pnl_usd"), errors="coerce").fillna(0.0).astype(float)
    df["fee_cost_usd"] = pd.to_numeric(df.get("fee_cost_usd"), errors="coerce").fillna(0.0).astype(float)

    markets = int(df.shape[0])
    net = float(df["market_pnl_usd"].sum())
    fees = float(df["fee_cost_usd"].sum())

    wins = df[df["market_pnl_usd"] > 0]
    losses = df[df["market_pnl_usd"] < 0]

    win_rate = float(wins.shape[0] / markets) if markets else None

    sum_wins = float(wins["market_pnl_usd"].sum())
    sum_losses = float(losses["market_pnl_usd"].sum())

    profit_factor = None
    if sum_losses < 0:
        profit_factor = float(sum_wins / abs(sum_losses))

    expectancy = float(net / markets) if markets else None

    sell_mask = df.get("sold_out") == True  # noqa: E712
    sell_df = df[sell_mask] if "sold_out" in df.columns else df.iloc[0:0]
    sell_markets = int(sell_df.shape[0])

    sell_win_rate = None
    if sell_markets and "sell_win" in sell_df.columns:
        sell_win_rate = float(pd.to_numeric(sell_df["sell_win"], errors="coerce").fillna(0).mean())

    payout_lost = float(pd.to_numeric(df.get("payout_lost_usd"), errors="coerce").fillna(0.0).sum())
    payout_saved = float(pd.to_numeric(df.get("payout_saved_usd"), errors="coerce").fillna(0.0).sum())

    return MarketKpis(
        markets=markets,
        win_rate=win_rate,
        net_pnl_usd=net,
        total_fees_usd=fees,
        profit_factor=profit_factor,
        expectancy=expectancy,
        sell_markets=sell_markets,
        sell_win_rate=sell_win_rate,
        payout_lost_usd=payout_lost,
        payout_saved_usd=payout_saved,
    )
