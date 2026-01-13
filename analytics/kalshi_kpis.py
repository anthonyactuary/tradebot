from __future__ import annotations

"""Kalshi KPI computation from fills + settlements CSVs.

This module is intentionally "notebook friendly": all core functions accept/return
pandas DataFrames and plain dicts.

Core accounting (per market ticker):
- Buy cashflow  = -(price_usd * count)
- Sell cashflow = +(price_usd * count)
- realized_pnl_usd = sum(cashflows) + revenue_usd - fees

Where:
- `revenue_usd` and `fee_cost_usd` come from settlements (finalized market-level truth).
- Optional per-fill fees can be included depending on `fee_source`.

Early sell saved/lost vs hold-to-expiry (counterfactual)
--------------------------------------------------------
Example (simple):

- Buy NO 1 @ 0.57
- Sell NO 1 @ 0.41
- Outcome YES

Realized exit PnL = (0.41 - 0.57) * 1 = -0.16
Hold PnL (NO, outcome YES) = (0.0 - 0.57) * 1 = -0.57
Delta vs hold = (-0.16) - (-0.57) = +0.41 => saved $0.41

Notes / assumptions:
- A "market" is keyed by `market_ticker` (preferred), falling back to `ticker`.
- We handle multiple fills per market and (best-effort) multiple "episodes" in the
  same market (flat->enter->exit->flat). Metrics are returned aggregated per market.
- We focus on YES/NO binary contracts.
"""

from dataclasses import dataclass, asdict
from typing import Any, Literal, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd  # type: ignore


FeeSource = Literal["fills", "settlement", "both"]


def _require_pandas() -> Any:
    try:
        import pandas as pd  # type: ignore

        return pd
    except Exception as e:
        raise RuntimeError("Missing dependency 'pandas'. Install with: pip install pandas") from e


def _normalize_price_to_usd(series: "pd.Series") -> "pd.Series":
    pd = _require_pandas()
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().any() and float(s.max()) <= 2.5:
        return s.astype(float)
    # Heuristic: looks like cents.
    return (s.astype(float) / 100.0)


def load_fills(path: str) -> "pd.DataFrame":
    """Load fills CSV and normalize columns.

    Output columns:
    - market_ticker, ts, side, action, count, price_usd, fee_usd
    """

    pd = _require_pandas()
    df = pd.read_csv(path, on_bad_lines="skip")
    df.columns = [str(c).strip() for c in df.columns]

    # market key
    if "market_ticker" not in df.columns:
        df["market_ticker"] = df.get("ticker")

    # timestamp
    if "ts" not in df.columns:
        if "created_time" in df.columns:
            df["ts"] = df["created_time"]
        else:
            df["ts"] = None

    # count / price
    if "count" not in df.columns and "contracts" in df.columns:
        df["count"] = df["contracts"]
    if "price" not in df.columns and "price_cents" in df.columns:
        df["price"] = df["price_cents"]

    df["market_ticker"] = df["market_ticker"].astype(str)
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df["side"] = df.get("side").astype(str).str.upper().str.strip()
    df["action"] = df.get("action").astype(str).str.lower().str.strip()
    df["count"] = pd.to_numeric(df.get("count"), errors="coerce").fillna(0).astype(int)

    df["price_usd"] = _normalize_price_to_usd(df.get("price"))

    # per-fill fees are optional
    if "fee" in df.columns and "fee_usd" not in df.columns:
        df["fee_usd"] = df["fee"]
    if "fee_usd" not in df.columns:
        df["fee_usd"] = 0.0
    df["fee_usd"] = pd.to_numeric(df["fee_usd"], errors="coerce").fillna(0.0).astype(float)

    df = df[(df["market_ticker"].astype(str) != "") & (df["count"] > 0)].copy()
    return df[["market_ticker", "ts", "side", "action", "count", "price_usd", "fee_usd"]]


def load_settlements(path: str) -> "pd.DataFrame":
    """Load settlements CSV and normalize columns.

    Output columns:
    - market_ticker, outcome, settled_ts,
      yes_count, yes_total_cost_usd, no_count, no_total_cost_usd,
      fee_cost_usd, revenue_usd
    """

    pd = _require_pandas()
    df = pd.read_csv(path, on_bad_lines="skip")
    df.columns = [str(c).strip() for c in df.columns]

    if "market_ticker" not in df.columns:
        df["market_ticker"] = df.get("ticker")

    if "outcome" not in df.columns:
        if "result" in df.columns:
            df["outcome"] = df["result"]
        elif "market_result" in df.columns:
            df["outcome"] = df["market_result"]
        else:
            df["outcome"] = None

    if "settled_ts" not in df.columns:
        if "settled_time" in df.columns:
            df["settled_ts"] = df["settled_time"]
        else:
            df["settled_ts"] = None

    df["market_ticker"] = df["market_ticker"].astype(str)
    df["outcome"] = df["outcome"].astype(str).str.upper().str.strip()
    df["settled_ts"] = pd.to_datetime(df["settled_ts"], utc=True, errors="coerce")

    # costs are usually dollar strings
    def _usd(col: str, default: float = 0.0) -> "pd.Series":
        return pd.to_numeric(df.get(col), errors="coerce").fillna(default).astype(float)

    def _int(col: str, default: int = 0) -> "pd.Series":
        return pd.to_numeric(df.get(col), errors="coerce").fillna(default).astype(int)

    df["yes_count"] = _int("yes_count")
    df["no_count"] = _int("no_count")

    df["yes_total_cost_usd"] = _usd("yes_total_cost")
    df["no_total_cost_usd"] = _usd("no_total_cost")

    # settlement fee is dollars string
    df["fee_cost_usd"] = _usd("fee_cost")

    # revenue is usually int cents
    rev = df.get("revenue")
    rev_num = pd.to_numeric(rev, errors="coerce").fillna(0.0)
    # if it looks like cents (max > 2.5), treat as cents
    if rev_num.notna().any() and float(rev_num.max()) > 2.5:
        df["revenue_usd"] = (rev_num.astype(float) / 100.0)
    else:
        df["revenue_usd"] = rev_num.astype(float)

    df = df[(df["market_ticker"].astype(str) != "")].copy()

    # One row per market
    df = df.drop_duplicates(subset=["market_ticker"], keep="first")

    return df[
        [
            "market_ticker",
            "outcome",
            "settled_ts",
            "yes_count",
            "yes_total_cost_usd",
            "no_count",
            "no_total_cost_usd",
            "fee_cost_usd",
            "revenue_usd",
        ]
    ]


@dataclass(frozen=True)
class SummaryKpis:
    markets: int
    realized_pnl_usd: float
    settlement_win_rate_held: float | None
    settlement_win_rate_all: float | None

    early_sell_count: int
    early_sell_saved_usd: float
    early_sell_lost_usd: float

    sell_markets: int
    sell_win_rate: float | None


def compute_per_market_metrics(
    df_fills: "pd.DataFrame",
    df_settlements: "pd.DataFrame",
    *,
    fee_source: FeeSource = "settlement",
) -> "pd.DataFrame":
    """Return per-market aggregated metrics.

    Output columns (minimum):
    market_ticker, outcome, settled_ts,
    realized_pnl_usd,
    held_to_expiry, settlement_win,
    early_flatten,
    early_sell_saved_usd, early_sell_lost_usd,
    entry_side, entry_vwap, exit_vwap, closed_qty
    """

    pd = _require_pandas()

    fills = df_fills.copy() if df_fills is not None else pd.DataFrame()
    settlements = df_settlements.copy() if df_settlements is not None else pd.DataFrame()

    if fills.empty and settlements.empty:
        return pd.DataFrame(
            columns=[
                "market_ticker",
                "outcome",
                "settled_ts",
                "realized_pnl_usd",
                "held_to_expiry",
                "settlement_win",
                "early_flatten",
                "early_sell_saved_usd",
                "early_sell_lost_usd",
                "entry_side",
                "entry_vwap",
                "exit_vwap",
                "closed_qty",
                "sell_markets",
            ]
        )

    # Normalize expected columns if caller uses our existing load_csvs outputs.
    if not fills.empty:
        if "count" not in fills.columns and "contracts" in fills.columns:
            fills["count"] = fills["contracts"]
        if "price_usd" not in fills.columns:
            if "price_cents" in fills.columns:
                fills["price_usd"] = pd.to_numeric(fills["price_cents"], errors="coerce").fillna(0.0).astype(float) / 100.0
            elif "price" in fills.columns:
                fills["price_usd"] = _normalize_price_to_usd(fills["price"])
            else:
                fills["price_usd"] = 0.0
        if "market_ticker" not in fills.columns:
            fills["market_ticker"] = fills.get("ticker")
        if "ts" not in fills.columns:
            fills["ts"] = fills.get("created_time")
        fills["ts"] = pd.to_datetime(fills["ts"], utc=True, errors="coerce")
        fills["market_ticker"] = fills["market_ticker"].astype(str)
        fills["side"] = fills.get("side").astype(str).str.upper().str.strip()
        fills["action"] = fills.get("action").astype(str).str.lower().str.strip()
        fills["count"] = pd.to_numeric(fills.get("count"), errors="coerce").fillna(0).astype(int)
        if "fee_usd" not in fills.columns:
            fills["fee_usd"] = 0.0
        fills["fee_usd"] = pd.to_numeric(fills["fee_usd"], errors="coerce").fillna(0.0).astype(float)

        fills = fills[(fills["market_ticker"].astype(str) != "") & (fills["count"] > 0)].copy()

    if not settlements.empty:
        if "market_ticker" not in settlements.columns:
            settlements["market_ticker"] = settlements.get("ticker")
        if "outcome" not in settlements.columns:
            if "result" in settlements.columns:
                settlements["outcome"] = settlements["result"]
            else:
                settlements["outcome"] = settlements.get("market_result")
        if "settled_ts" not in settlements.columns:
            settlements["settled_ts"] = settlements.get("settled_time")

        settlements["market_ticker"] = settlements["market_ticker"].astype(str)
        settlements["outcome"] = settlements["outcome"].astype(str).str.upper().str.strip()
        settlements["settled_ts"] = pd.to_datetime(settlements["settled_ts"], utc=True, errors="coerce")

        if "revenue_usd" not in settlements.columns:
            rev = pd.to_numeric(settlements.get("revenue"), errors="coerce").fillna(0.0)
            if rev.notna().any() and float(rev.max()) > 2.5:
                settlements["revenue_usd"] = rev.astype(float) / 100.0
            else:
                settlements["revenue_usd"] = rev.astype(float)
        else:
            settlements["revenue_usd"] = pd.to_numeric(settlements["revenue_usd"], errors="coerce").fillna(0.0).astype(float)

        if "fee_cost_usd" not in settlements.columns:
            settlements["fee_cost_usd"] = pd.to_numeric(settlements.get("fee_cost"), errors="coerce").fillna(0.0).astype(float)
        else:
            settlements["fee_cost_usd"] = pd.to_numeric(settlements["fee_cost_usd"], errors="coerce").fillna(0.0).astype(float)

        settlements = settlements.drop_duplicates(subset=["market_ticker"], keep="first")

    # Settlement lookup
    s = settlements.set_index("market_ticker") if not settlements.empty else pd.DataFrame()

    # Aggregate per market from fills
    rows: list[dict[str, Any]] = []
    market_keys = set()
    if not fills.empty:
        market_keys |= set(fills["market_ticker"].astype(str).tolist())
    if not settlements.empty:
        market_keys |= set(settlements["market_ticker"].astype(str).tolist())

    for mt in sorted(market_keys):
        f = fills[fills["market_ticker"] == mt].sort_values("ts") if not fills.empty else fills.iloc[0:0]

        outcome = None
        settled_ts = None
        revenue_usd = 0.0
        fee_cost_usd = 0.0
        if not settlements.empty and mt in s.index:
            outcome = str(s.loc[mt].get("outcome") or "").upper() or None
            settled_ts = s.loc[mt].get("settled_ts")
            revenue_usd = float(s.loc[mt].get("revenue_usd") or 0.0)
            fee_cost_usd = float(s.loc[mt].get("fee_cost_usd") or 0.0)

        # Cashflows
        buy_cost = 0.0
        sell_value = 0.0
        fill_fees = 0.0

        # Position reconstruction and episode/entry/exit vwap (best-effort; assumes single-direction per market)
        entry_side = None
        entry_qty = 0
        entry_notional = 0.0
        exit_qty = 0
        exit_notional = 0.0
        last_sell_time = None

        pos_yes = 0
        pos_no = 0

        for _, r in f.iterrows():
            action = str(r.get("action") or "").lower()
            side = str(r.get("side") or "").upper()
            cnt = int(r.get("count") or 0)
            px = float(r.get("price_usd") or 0.0)
            ts = r.get("ts")
            fee = float(r.get("fee_usd") or 0.0)

            if cnt <= 0:
                continue

            notional = px * float(cnt)

            if action == "buy":
                buy_cost += notional
                if entry_side is None and (pos_yes == 0 and pos_no == 0):
                    entry_side = side

                if side == "YES":
                    pos_yes += cnt
                elif side == "NO":
                    pos_no += cnt

                if side == entry_side:
                    entry_qty += cnt
                    entry_notional += notional

            elif action == "sell":
                sell_value += notional
                last_sell_time = ts

                if side == "YES":
                    pos_yes -= cnt
                elif side == "NO":
                    pos_no -= cnt

                if side == entry_side:
                    exit_qty += cnt
                    exit_notional += notional

            if fee_source in ("fills", "both"):
                fill_fees += fee

        # Determine held_to_expiry: net position at settlement time (use final pos after all fills; assume fills stop before settlement)
        held_side = None
        if pos_yes > 0:
            held_side = "YES"
        elif pos_no > 0:
            held_side = "NO"

        held_to_expiry = bool(held_side is not None)

        settlement_win = None
        if outcome and held_side:
            settlement_win = bool(held_side == outcome)

        # Early flatten: end flat and we have sells before settlement
        early_flatten = False
        if outcome and settled_ts is not None:
            is_flat = (pos_yes == 0 and pos_no == 0)
            if is_flat and last_sell_time is not None:
                try:
                    early_flatten = bool(pd.to_datetime(last_sell_time, utc=True) < pd.to_datetime(settled_ts, utc=True))
                except Exception:
                    early_flatten = True

        entry_vwap = (entry_notional / entry_qty) if entry_qty > 0 else None
        exit_vwap = (exit_notional / exit_qty) if exit_qty > 0 else None
        closed_qty = float(min(entry_qty, exit_qty))

        # Saved vs lost vs hold for the sold quantity (partial exits count too).
        saved = 0.0
        lost = 0.0
        if closed_qty > 0 and entry_vwap is not None and exit_vwap is not None and outcome in ("YES", "NO") and entry_side in ("YES", "NO"):
            hold_pnl_per_contract = (1.0 - float(entry_vwap)) if outcome == entry_side else (0.0 - float(entry_vwap))
            realized_exit_pnl_per_contract = float(exit_vwap) - float(entry_vwap)
            delta_vs_hold = (realized_exit_pnl_per_contract - hold_pnl_per_contract) * float(closed_qty)
            if delta_vs_hold > 0:
                saved = float(delta_vs_hold)
            elif delta_vs_hold < 0:
                lost = float(abs(delta_vs_hold))

        # Fees
        fees = 0.0
        if fee_source in ("settlement", "both"):
            fees += float(fee_cost_usd)
        if fee_source in ("fills", "both"):
            fees += float(fill_fees)

        realized_pnl = float(revenue_usd + sell_value - buy_cost - fees)

        sell_markets = int(exit_qty > 0)

        rows.append(
            {
                "market_ticker": mt,
                "outcome": outcome,
                "settled_ts": settled_ts,
                "realized_pnl_usd": realized_pnl,
                "held_to_expiry": bool(held_to_expiry),
                "settlement_win": settlement_win,
                "early_flatten": bool(early_flatten),
                "early_sell_saved_usd": float(saved),
                "early_sell_lost_usd": float(lost),
                "entry_side": entry_side,
                "entry_vwap": entry_vwap,
                "exit_vwap": exit_vwap,
                "closed_qty": float(closed_qty),
                "sell_markets": int(sell_markets),
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out["settled_ts"] = pd.to_datetime(out["settled_ts"], utc=True, errors="coerce")
        out = out.sort_values(["settled_ts", "market_ticker"], na_position="last").reset_index(drop=True)

    return out


def compute_summary_kpis(df_metrics: "pd.DataFrame") -> dict[str, Any]:
    pandas_lib = _require_pandas()

    if df_metrics is None or df_metrics.empty:
        return asdict(
            SummaryKpis(
                markets=0,
                realized_pnl_usd=0.0,
                settlement_win_rate_held=None,
                settlement_win_rate_all=None,
                early_sell_count=0,
                early_sell_saved_usd=0.0,
                early_sell_lost_usd=0.0,
                sell_markets=0,
                sell_win_rate=None,
            )
        )

    df = df_metrics.copy()
    df["realized_pnl_usd"] = (
        pandas_lib.to_numeric(df.get("realized_pnl_usd"), errors="coerce").fillna(0.0).astype(float)
    )
    df["early_sell_saved_usd"] = (
        pandas_lib.to_numeric(df.get("early_sell_saved_usd"), errors="coerce").fillna(0.0).astype(float)
    )
    df["early_sell_lost_usd"] = (
        pandas_lib.to_numeric(df.get("early_sell_lost_usd"), errors="coerce").fillna(0.0).astype(float)
    )

    markets = int(df.shape[0])
    pnl = float(df["realized_pnl_usd"].sum())

    held = df[df.get("held_to_expiry") == True]  # noqa: E712
    win_rate_held = None
    if not held.empty and "settlement_win" in held.columns:
        s = held["settlement_win"].dropna()
        if not s.empty:
            win_rate_held = float((s == True).mean())  # noqa: E712

    win_rate_all = None
    if "settlement_win" in df.columns:
        s = df["settlement_win"].dropna()
        if not s.empty:
            win_rate_all = float((s == True).mean())  # noqa: E712

    early_sell_count = int((df.get("early_flatten") == True).sum()) if "early_flatten" in df.columns else 0  # noqa: E712

    saved = float(df["early_sell_saved_usd"].sum())
    lost = float(df["early_sell_lost_usd"].sum())

    sell_markets = int((df.get("sell_markets") > 0).sum()) if "sell_markets" in df.columns else 0

    sell_df = df[df.get("sell_markets") > 0] if "sell_markets" in df.columns else df.iloc[0:0]
    sell_win_rate = None
    if not sell_df.empty and "entry_side" in sell_df.columns and "outcome" in sell_df.columns:
        # "Sell win" as: you sold some quantity AND outcome != entry_side.
        ok = (sell_df["outcome"].astype(str).str.upper() != sell_df["entry_side"].astype(str).str.upper())
        sell_win_rate = float(ok.mean()) if len(ok) else None

    return asdict(
        SummaryKpis(
            markets=markets,
            realized_pnl_usd=pnl,
            settlement_win_rate_held=win_rate_held,
            settlement_win_rate_all=win_rate_all,
            early_sell_count=early_sell_count,
            early_sell_saved_usd=saved,
            early_sell_lost_usd=lost,
            sell_markets=sell_markets,
            sell_win_rate=sell_win_rate,
        )
    )


def run_from_csvs(
    *,
    fills_csv: str,
    settlements_csv: str,
    fee_source: FeeSource = "settlement",
) -> tuple["pd.DataFrame", dict[str, Any]]:
    """Convenience helper for scripts/notebooks."""

    df_fills = load_fills(fills_csv)
    df_settlements = load_settlements(settlements_csv)
    metrics = compute_per_market_metrics(df_fills, df_settlements, fee_source=fee_source)
    summary = compute_summary_kpis(metrics)
    return metrics, summary
