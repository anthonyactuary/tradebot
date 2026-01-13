from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _require_pandas() -> Any:
    try:
        import pandas as pd  # type: ignore

        return pd
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Missing dependency 'pandas'. Install with: pip install pandas") from e


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    for i in range(8):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            # Windows can temporarily lock files (Excel, antivirus, preview panes).
            if i == 7:
                raise
            time.sleep(0.15 * (i + 1))


def _atomic_write_csv(df: "Any", path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    for i in range(8):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if i == 7:
                raise
            time.sleep(0.15 * (i + 1))


def _to_int(series: "Any", default: int = 0) -> "Any":
    pd = _require_pandas()
    s = pd.to_numeric(series, errors="coerce")
    # If `series` is missing, pandas can return a scalar (numpy.float64) here.
    if not hasattr(s, "fillna"):
        return int(default)
    return s.fillna(default).astype(int)


def _to_float(series: "Any", default: float = 0.0) -> "Any":
    pd = _require_pandas()
    s = pd.to_numeric(series, errors="coerce")
    if not hasattr(s, "fillna"):
        return float(default)
    return s.fillna(default).astype(float)


def _col_int(df: "Any", name: str, default: int = 0) -> "Any":
    pd = _require_pandas()
    if df is None or df.empty:
        return pd.Series(dtype=int)
    if name not in df.columns:
        return pd.Series([default] * int(df.shape[0]), index=df.index, dtype=int)
    return _to_int(df[name], default=default)


def _col_float(df: "Any", name: str, default: float = 0.0) -> "Any":
    pd = _require_pandas()
    if df is None or df.empty:
        return pd.Series(dtype=float)
    if name not in df.columns:
        return pd.Series([default] * int(df.shape[0]), index=df.index, dtype=float)
    return _to_float(df[name], default=default)


def load_settlements_export(path: str) -> "Any":
    """Load Kalshi settlements export.

    Expected columns (per user spec):
      ticker, market_result, yes_count, no_count, yes_total_cost, no_total_cost, revenue, fee_cost

    Units (based on current exports in runs/):
      - yes_total_cost, no_total_cost, revenue are integer cents
      - fee_cost is USD (string/float)
    """

    pd = _require_pandas()
    df = pd.read_csv(path, on_bad_lines="skip")
    df.columns = [str(c).strip() for c in df.columns]

    out = df.copy()
    if "market_result" not in out.columns and "result" in out.columns:
        out["market_result"] = out["result"]

    out["ticker"] = out.get("ticker").astype(str)
    out["market_result"] = out.get("market_result").astype(str).str.lower().str.strip()

    # Optional, used for filtering sells to pre-settlement.
    if "settled_time" in out.columns:
        out["settled_time"] = pd.to_datetime(out.get("settled_time"), utc=True, errors="coerce")

    out["yes_count"] = _to_int(out.get("yes_count"))
    out["no_count"] = _to_int(out.get("no_count"))

    out["yes_total_cost"] = _to_int(out.get("yes_total_cost"))
    out["no_total_cost"] = _to_int(out.get("no_total_cost"))

    out["revenue"] = _to_int(out.get("revenue"))
    out["fee_cost"] = _to_float(out.get("fee_cost"))

    out = out[out["ticker"].astype(str) != ""].copy()
    out = out.drop_duplicates(subset=["ticker"], keep="first")

    cols = [
        "ticker",
        "market_result",
        "yes_count",
        "no_count",
        "yes_total_cost",
        "no_total_cost",
        "revenue",
        "fee_cost",
    ]
    if "settled_time" in out.columns:
        cols.append("settled_time")

    return out[cols]


def load_fills_export(path: str) -> "Any":
    """Load Kalshi fills export.

    Expected columns (per user spec):
      ticker, created_time, side (yes/no), action (buy/sell), count, price

    Price is USD in current exports.
    """

    pd = _require_pandas()
    df = pd.read_csv(path, on_bad_lines="skip")
    df.columns = [str(c).strip() for c in df.columns]

    out = df.copy()
    if "created_time" not in out.columns and "ts" in out.columns:
        out["created_time"] = out["ts"]

    out["ticker"] = out.get("ticker").astype(str)
    out["created_time"] = pd.to_datetime(out.get("created_time"), utc=True, errors="coerce")

    out["side"] = out.get("side").astype(str).str.lower().str.strip()
    out["action"] = out.get("action").astype(str).str.lower().str.strip()

    out["count"] = _to_int(out.get("count"))

    # Prefer explicit USD price (current export uses 0.xx)
    if "price" not in out.columns and "price_cents" in out.columns:
        out["price"] = _to_float(out.get("price_cents")) / 100.0
    out["price"] = _to_float(out.get("price"))

    out = out[(out["ticker"].astype(str) != "") & (out["count"] > 0)].copy()

    return out[["ticker", "created_time", "side", "action", "count", "price"]]


def compute_settlement_accounting(settlements: "Any") -> "Any":
    """Compute per-market accounting using the paired-payout adjustment."""

    pd = _require_pandas()
    if settlements is None or settlements.empty:
        return pd.DataFrame(
            columns=[
                "ticker",
                "realized_pnl_usd",
                "is_win",
                "gross_payout_usd",
                "gross_cost_usd",
                "fee_cost",
                "yes_count",
                "no_count",
                "paired_count",
                "net_count",
                "market_result",
            ]
        )

    s = settlements.copy()

    s["yes_count"] = _col_int(s, "yes_count", default=0)
    s["no_count"] = _col_int(s, "no_count", default=0)
    s["paired_count"] = s[["yes_count", "no_count"]].min(axis=1).astype(int)

    s["revenue"] = _col_int(s, "revenue", default=0)
    s["yes_total_cost"] = _col_int(s, "yes_total_cost", default=0)
    s["no_total_cost"] = _col_int(s, "no_total_cost", default=0)
    s["fee_cost"] = _col_float(s, "fee_cost", default=0.0)

    payout_cents = s["revenue"].astype(int) + 100 * s["paired_count"].astype(int)
    cost_cents = s["yes_total_cost"].astype(int) + s["no_total_cost"].astype(int)

    s["gross_payout_usd"] = payout_cents.astype(float) / 100.0
    s["gross_cost_usd"] = cost_cents.astype(float) / 100.0

    s["realized_pnl_usd"] = (payout_cents.astype(float) - cost_cents.astype(float)) / 100.0 - s[
        "fee_cost"
    ].astype(float)

    s["is_win"] = s["realized_pnl_usd"].astype(float) > 0.0
    s["net_count"] = (s["yes_count"].astype(int) - s["no_count"].astype(int)).abs().astype(int)

    cols = [
        "ticker",
        "realized_pnl_usd",
        "is_win",
        "gross_payout_usd",
        "gross_cost_usd",
        "fee_cost",
        "yes_count",
        "no_count",
        "paired_count",
        "net_count",
        "market_result",
    ]
    for c in cols:
        if c not in s.columns:
            s[c] = None

    return s[cols].copy()


def _settlement_value_usd(*, side: str, market_result: str) -> float:
    side = (side or "").lower().strip()
    mr = (market_result or "").lower().strip()

    if side == "yes":
        return 1.0 if mr == "yes" else 0.0
    if side == "no":
        return 1.0 if mr == "no" else 0.0
    return 0.0


def compute_early_sell_metrics(*, fills: "Any", settlements: "Any") -> "Any":
    """Compute early-sell/flatten metrics.

    IMPORTANT:
    Fills windows may not include the opening BUY. Kalshi does not allow shorting
    without inventory, so every SELL implies a close/reduce. Therefore, we treat
    every SELL fill row as an early-close event without requiring BUY-lot matching.

        Early-sell is computed per ticker using the "position direction" = side of the
        first BUY fill for that ticker.

        All SELL fills (regardless of side) indicate an early-close attempt for this
        ticker. For a given ticker we compute:

            entry_vwap = VWAP of BUY fills on the entry_side
            exit_qty   = total SELL quantity (all sides)

        Early-sell lost (missed upside) is defined as the profit you would have made
        by holding the exited contracts to expiry, based on entry_vwap:

            if outcome == entry_side:
                lost_usd = exit_qty * (1 - entry_vwap)
            else:
                lost_usd = 0

        Early-sell saved is computed later as abs(realized_pnl_usd) for early-sold
        tickers (since realized PnL comes from settlement accounting).

    Optionally filters to sells that occur strictly before settlement time if a
    settlement timestamp column is available.
    """

    pd = _require_pandas()
    if fills is None or fills.empty:
        return pd.DataFrame(columns=["ticker", "early_sell_contracts", "early_sell_lost_usd"])

    f = fills.copy()
    if "created_time" not in f.columns and "ts" in f.columns:
        f["created_time"] = f["ts"]
    f["created_time"] = pd.to_datetime(f.get("created_time"), utc=True, errors="coerce")
    f["ticker"] = f.get("ticker").astype(str)
    f["side"] = f.get("side").astype(str).str.lower().str.strip()
    f["action"] = f.get("action").astype(str).str.lower().str.strip()
    f["count"] = _to_int(f.get("count"))
    f["price"] = _to_float(f.get("price"))

    # Kalshi exports often include both sides' prices; in our runs/kalshi_fills.csv,
    # the "price" column is the YES price even when side==NO.
    if "yes_price_fixed" in f.columns:
        f["yes_price_fixed"] = _to_float(f["yes_price_fixed"], default=0.0)
    else:
        f["yes_price_fixed"] = pd.to_numeric(None, errors="coerce")

    if "no_price_fixed" in f.columns:
        f["no_price_fixed"] = _to_float(f["no_price_fixed"], default=0.0)
    else:
        f["no_price_fixed"] = pd.to_numeric(None, errors="coerce")

    # Normalize: if any price-like column looks like cents, convert to dollars.
    for col in ["price", "yes_price_fixed", "no_price_fixed"]:
        if col not in f.columns:
            continue
        mask = f[col].astype(float) > 1.5
        if mask.any():
            f.loc[mask, col] = f.loc[mask, col] / 100.0

    def _effective_price_usd(row: "Any") -> float:
        side = str(row.get("side") or "").lower().strip()
        if side == "yes":
            v = row.get("yes_price_fixed")
            if v is None or (isinstance(v, float) and pd.isna(v)):
                v = row.get("price")
            return float(0.0 if v is None or (isinstance(v, float) and pd.isna(v)) else v)
        if side == "no":
            v = row.get("no_price_fixed")
            if v is None or (isinstance(v, float) and pd.isna(v)):
                p = row.get("price")
                if p is None or (isinstance(p, float) and pd.isna(p)):
                    return 0.0
                v = 1.0 - float(p)
            return float(0.0 if v is None or (isinstance(v, float) and pd.isna(v)) else v)
        return 0.0

    f["_px"] = f.apply(_effective_price_usd, axis=1)

    # Settlement lookup: outcome + optional settled time
    outcome_by_ticker: dict[str, str] = {}
    settled_time_by_ticker: dict[str, Any] = {}
    if settlements is not None and not settlements.empty:
        s = settlements.copy()
        if "market_result" not in s.columns and "result" in s.columns:
            s["market_result"] = s["result"]
        s["ticker"] = s.get("ticker").astype(str)
        s["market_result"] = s.get("market_result").astype(str).str.lower().str.strip()

        # Try to find a settlement timestamp column.
        settled_col = None
        for c in ["settled_time", "settled_ts", "settled_at"]:
            if c in s.columns:
                settled_col = c
                break
        if settled_col is not None:
            s[settled_col] = pd.to_datetime(s.get(settled_col), utc=True, errors="coerce")

        for _, r in s.iterrows():
            t = str(r.get("ticker") or "")
            if not t:
                continue
            outcome_by_ticker[t] = str(r.get("market_result") or "")
            if settled_col is not None:
                settled_time_by_ticker[t] = r.get(settled_col)

    # Optional: filter to fills strictly before settlement time.
    if "created_time" in f.columns and settled_time_by_ticker:
        st = f["ticker"].map(settled_time_by_ticker)
        st = pd.to_datetime(st, utc=True, errors="coerce")
        f["_settled_time"] = st
        f = f[(f["_settled_time"].isna()) | (f["created_time"] < f["_settled_time"])].copy()

    rows: list[dict[str, Any]] = []

    for ticker, grp in f.groupby("ticker", sort=True):
        g = grp.sort_values("created_time")

        # Position direction = first BUY side.
        buys = g[g["action"] == "buy"]
        if buys.empty:
            continue
        entry_side = str(buys.iloc[0].get("side") or "").lower().strip()
        if entry_side not in ("yes", "no"):
            continue

        outcome = str(outcome_by_ticker.get(str(ticker), "")).lower().strip()
        if outcome not in ("yes", "no"):
            # Unknown outcome => can't score saved/lost.
            continue

        sells = g[g["action"] == "sell"]
        if sells.empty:
            continue

        exit_qty = int(_to_int(sells["count"]).sum())
        if exit_qty <= 0:
            continue

        entry_buys = buys[buys["side"] == entry_side]
        entry_qty = float(_to_int(entry_buys.get("count")).sum())
        entry_cost = float((entry_buys.get("_px").astype(float) * entry_buys.get("count").astype(float)).sum())
        entry_vwap = float(entry_cost / entry_qty) if entry_qty > 0 else 0.0

        lost = 0.0
        if outcome == entry_side:
            lost = float(max(0.0, float(exit_qty) * (1.0 - entry_vwap)))

        rows.append(
            {
                "ticker": str(ticker),
                "early_sell_contracts": int(exit_qty),
                "early_sell_lost_usd": float(lost),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["ticker", "early_sell_contracts", "early_sell_lost_usd"])

    out["ticker"] = out["ticker"].astype(str)
    out["early_sell_contracts"] = _to_int(out.get("early_sell_contracts"))
    out["early_sell_lost_usd"] = _to_float(out.get("early_sell_lost_usd"))

    return out[["ticker", "early_sell_contracts", "early_sell_lost_usd"]]


def build_per_market_metrics(*, settlements: "Any", fills: "Any") -> "Any":
    pd = _require_pandas()

    acct = compute_settlement_accounting(settlements)
    early = compute_early_sell_metrics(fills=fills, settlements=settlements)

    if acct.empty and early.empty:
        return pd.DataFrame(
            columns=[
                "ticker",
                "realized_pnl_usd",
                "is_win",
                "gross_payout_usd",
                "gross_cost_usd",
                "fee_cost",
                "yes_count",
                "no_count",
                "paired_count",
                "net_count",
                "early_sell_contracts",
                "early_sell_saved_usd",
                "early_sell_lost_usd",
                "market_result",
            ]
        )

    base = acct
    if base.empty:
        base = pd.DataFrame({"ticker": early["ticker"].astype(str)})

    out = base.merge(early, on="ticker", how="left")
    out["early_sell_contracts"] = _to_int(out.get("early_sell_contracts"))
    out["early_sell_lost_usd"] = _to_float(out.get("early_sell_lost_usd"))

    # Early-sell saved is defined as abs(PnL) for early-sold tickers,
    # but must be 0 if we attribute an early-sell lost.
    out["early_sell_saved_usd"] = 0.0
    mask_early = out.get("early_sell_contracts").astype(int) > 0
    mask_lost = _to_float(out.get("early_sell_lost_usd")) > 0.0
    if "realized_pnl_usd" in out.columns:
        mask_saved = mask_early & (~mask_lost)
        out.loc[mask_saved, "early_sell_saved_usd"] = out.loc[mask_saved, "realized_pnl_usd"].abs()
    out["early_sell_saved_usd"] = _to_float(out.get("early_sell_saved_usd"))

    # Ensure stable dtypes for downstream
    out["ticker"] = out.get("ticker").astype(str)

    return out


def build_summary_kpis(per_market: "Any") -> dict[str, Any]:
    pd = _require_pandas()
    if per_market is None or per_market.empty:
        return {
            "markets": 0,
            "total_realized_pnl": 0.0,
            "win_rate": None,
            "avg_pnl": None,
            "median_pnl": None,
            "profit_factor": None,
            "total_early_sell_contracts": 0,
            "total_early_sell_saved_usd": 0.0,
            "total_early_sell_lost_usd": 0.0,
        }

    df = per_market.copy()
    df["realized_pnl_usd"] = _to_float(df.get("realized_pnl_usd"))

    markets = int(df.shape[0])
    total_pnl = float(df["realized_pnl_usd"].sum())

    wins = df[df["realized_pnl_usd"] > 0.0]
    losses = df[df["realized_pnl_usd"] < 0.0]

    win_rate = float(wins.shape[0] / markets) if markets else None
    avg_pnl = float(df["realized_pnl_usd"].mean()) if markets else None
    median_pnl = float(df["realized_pnl_usd"].median()) if markets else None

    sum_wins = float(wins["realized_pnl_usd"].sum())
    sum_losses = float(losses["realized_pnl_usd"].sum())

    profit_factor = None
    if sum_losses < 0:
        profit_factor = float(sum_wins / abs(sum_losses))

    df["early_sell_contracts"] = _to_int(df.get("early_sell_contracts"))
    df["early_sell_saved_usd"] = _to_float(df.get("early_sell_saved_usd"))
    df["early_sell_lost_usd"] = _to_float(df.get("early_sell_lost_usd"))

    return {
        "markets": markets,
        "total_realized_pnl": total_pnl,
        "win_rate": win_rate,
        "avg_pnl": avg_pnl,
        "median_pnl": median_pnl,
        "profit_factor": profit_factor,
        "total_early_sell_contracts": int(df["early_sell_contracts"].sum()),
        "total_early_sell_saved_usd": float(df["early_sell_saved_usd"].sum()),
        "total_early_sell_lost_usd": float(df["early_sell_lost_usd"].sum()),
    }


def compute_and_write(
    *,
    settlements_csv: str,
    fills_csv: str,
    out_market_csv: str,
    out_summary_json: str,
) -> tuple["Any", dict[str, Any]]:
    settlements = load_settlements_export(settlements_csv)
    fills = load_fills_export(fills_csv)

    per_market = build_per_market_metrics(settlements=settlements, fills=fills)
    summary = build_summary_kpis(per_market)

    _atomic_write_csv(per_market, Path(out_market_csv))
    _atomic_write_text(Path(out_summary_json), json.dumps(summary, indent=2, sort_keys=True))

    return per_market, summary


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute Kalshi paired-payout KPIs from CSV exports")
    p.add_argument("--settlements-csv", default="runs/kalshi_settlements.csv")
    p.add_argument("--fills-csv", default="runs/kalshi_fills.csv")
    p.add_argument("--out-market-csv", default="runs/kalshi_per_market_metrics_fixed.csv")
    p.add_argument("--out-summary-json", default="runs/kalshi_summary_metrics.json")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    compute_and_write(
        settlements_csv=str(args.settlements_csv),
        fills_csv=str(args.fills_csv),
        out_market_csv=str(args.out_market_csv),
        out_summary_json=str(args.out_summary_json),
    )


if __name__ == "__main__":
    main()
