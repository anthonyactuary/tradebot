from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd  # type: ignore


@dataclass(frozen=True)
class LoadResult:
    fills: "pd.DataFrame"
    settlements: "pd.DataFrame"
    diagnostics: dict[str, Any]


def _require_pandas() -> Any:
    try:
        import pandas as pd  # type: ignore

        return pd
    except Exception as e:
        raise RuntimeError("Missing dependency 'pandas'. Install with: pip install pandas") from e


def _read_csv_best_effort(path: str) -> "pd.DataFrame":
    pd = _require_pandas()

    if not path or not os.path.exists(path):
        return pd.DataFrame()

    # CSVs might be append-only and occasionally partially written.
    # on_bad_lines='skip' keeps it robust.
    try:
        return pd.read_csv(path, on_bad_lines="skip")
    except TypeError:
        # Older pandas may not support on_bad_lines.
        return pd.read_csv(path)


def _normalize_fills(df: "pd.DataFrame") -> "pd.DataFrame":
    pd = _require_pandas()

    if df is None or df.empty:
        return pd.DataFrame(
            columns=[
                "ts",
                "ticker",
                "market_ticker",
                "side",
                "action",
                "contracts",
                "price_cents",
                "fee_usd",
                "order_id",
                "strategy_tag",
            ]
        )

    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]

    # Column aliases (support both the requested schema and common Kalshi exports)
    if "contracts" not in out.columns and "count" in out.columns:
        out["contracts"] = out["count"]
    if "price_cents" not in out.columns and "price" in out.columns:
        out["price_cents"] = out["price"]
    if "ts" not in out.columns and "created_time" in out.columns:
        out["ts"] = out["created_time"]

    # Prefer explicit market_ticker if present; otherwise fall back to ticker.
    if "market_ticker" not in out.columns:
        out["market_ticker"] = out.get("ticker")

    if "fee_usd" not in out.columns:
        out["fee_usd"] = 0.0

    # Ensure required cols exist
    for col in ["ts", "ticker", "market_ticker", "side", "action", "contracts", "price_cents"]:
        if col not in out.columns:
            out[col] = None

    # Clean
    out["ticker"] = out["ticker"].astype(str)
    out["market_ticker"] = out["market_ticker"].astype(str)
    out["side"] = out["side"].astype(str).str.upper().str.strip()
    out["action"] = out["action"].astype(str).str.lower().str.strip()

    out["contracts"] = pd.to_numeric(out["contracts"], errors="coerce").fillna(0).astype(int)

    # Price may be cents or dollars depending on upstream.
    pc = pd.to_numeric(out["price_cents"], errors="coerce")
    # If it looks like dollars (<= 2), convert to cents.
    if pc.notna().any() and float(pc.max()) <= 2.5:
        pc = (pc * 100.0).round(0)
    out["price_cents"] = pc.fillna(0).astype(int)

    out["fee_usd"] = pd.to_numeric(out["fee_usd"], errors="coerce").fillna(0.0).astype(float)

    if "order_id" not in out.columns:
        out["order_id"] = out.get("order_id", out.get("fill_id", ""))
    if "strategy_tag" not in out.columns:
        out["strategy_tag"] = out.get("strategy_tag", "")

    out = out[["ts", "ticker", "market_ticker", "side", "action", "contracts", "price_cents", "fee_usd", "order_id", "strategy_tag"]]
    return out


def _normalize_settlements(df: "pd.DataFrame") -> "pd.DataFrame":
    pd = _require_pandas()

    if df is None or df.empty:
        return pd.DataFrame(
            columns=[
                "ticker",
                "market_ticker",
                "market_close_ts",
                "result",
                "payout_per_contract_usd",
                "settled_ts",
                "fee_cost_usd",
                "revenue_cents",
                "revenue_usd",
                "net_pnl_usd",
            ]
        )

    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]

    if "result" not in out.columns and "market_result" in out.columns:
        out["result"] = out["market_result"]
    if "settled_ts" not in out.columns and "settled_time" in out.columns:
        out["settled_ts"] = out["settled_time"]

    # Fees in settlements export are usually a dollars string field named fee_cost.
    if "fee_cost_usd" not in out.columns:
        if "fee_cost" in out.columns:
            out["fee_cost_usd"] = out["fee_cost"]
        else:
            out["fee_cost_usd"] = 0.0

    # Revenue in settlements export is usually integer cents.
    if "revenue_cents" not in out.columns:
        if "revenue" in out.columns:
            out["revenue_cents"] = out["revenue"]
        else:
            out["revenue_cents"] = 0

    if "payout_per_contract_usd" not in out.columns:
        # Best-effort default; most Kalshi binary contracts settle at $1.00 for the winning side.
        out["payout_per_contract_usd"] = 1.0

    if "ticker" not in out.columns and "event_ticker" in out.columns:
        out["ticker"] = out["event_ticker"]

    # In the exported settlements CSV, `ticker` is the market ticker (e.g. KXBTC15M-...).
    if "market_ticker" not in out.columns:
        out["market_ticker"] = out.get("ticker")

    for col in [
        "ticker",
        "market_ticker",
        "market_close_ts",
        "result",
        "payout_per_contract_usd",
        "settled_ts",
        "fee_cost_usd",
        "revenue_cents",
    ]:
        if col not in out.columns:
            out[col] = None

    out["ticker"] = out["ticker"].astype(str)
    out["market_ticker"] = out["market_ticker"].astype(str)
    out["result"] = out["result"].astype(str).str.upper().str.strip()

    out["payout_per_contract_usd"] = (
        pd.to_numeric(out["payout_per_contract_usd"], errors="coerce").fillna(1.0).astype(float)
    )

    out["fee_cost_usd"] = pd.to_numeric(out["fee_cost_usd"], errors="coerce").fillna(0.0).astype(float)

    out["revenue_cents"] = pd.to_numeric(out["revenue_cents"], errors="coerce").fillna(0).astype(int)
    out["revenue_usd"] = out["revenue_cents"].astype(float) / 100.0
    out["net_pnl_usd"] = out["revenue_usd"] - out["fee_cost_usd"]

    out = out[
        [
            "ticker",
            "market_ticker",
            "market_close_ts",
            "result",
            "payout_per_contract_usd",
            "settled_ts",
            "fee_cost_usd",
            "revenue_cents",
            "revenue_usd",
            "net_pnl_usd",
        ]
    ]
    return out


def load_csvs(*, fills_csv: str, settlements_csv: str) -> LoadResult:
    """Load and normalize fills/settlements CSVs.

    This function performs no caching: callers should call it on every refresh.
    """

    pd = _require_pandas()

    raw_fills = _read_csv_best_effort(str(fills_csv))
    raw_settlements = _read_csv_best_effort(str(settlements_csv))

    fills = _normalize_fills(raw_fills)
    settlements = _normalize_settlements(raw_settlements)

    diag = {
        "fills_path": str(fills_csv),
        "settlements_path": str(settlements_csv),
        "fills_rows": int(fills.shape[0]),
        "settlements_rows": int(settlements.shape[0]),
        "fills_cols": list(fills.columns),
        "settlements_cols": list(settlements.columns),
    }

    return LoadResult(fills=fills, settlements=settlements, diagnostics=diag)
