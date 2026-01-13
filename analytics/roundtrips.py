from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd  # type: ignore


@dataclass
class ReconstructionDiagnostics:
    unmatched_sells: int = 0
    unexpected_side_switches: int = 0
    settlements_used: int = 0


def reconstruct_roundtrips(
    *,
    fills: "pd.DataFrame",
    settlements: "pd.DataFrame",
) -> tuple["pd.DataFrame", "pd.DataFrame", ReconstructionDiagnostics]:
    """Reconstruct per-ticker round trips.

    Assumption (per strategy spec): single position only per market/ticker.

    We treat a "round trip" as the lifecycle of a position in a given ticker:
    - Entry begins with the first BUY from flat.
    - Exit occurs when position returns to 0 via SELLs, OR when a settlement row
      arrives for the ticker (closing any remaining position at payout).

    Robustness:
    - Handles partial fills and multiple fills at different prices.
    - If a settlement occurs while still holding some contracts, we close the
      remaining contracts at payout, while incorporating any earlier sells.

    Returns:
    - roundtrips_df: one row per closed round trip
    - open_positions_df: current open positions inferred from fills minus exits/settlements
    """

    import pandas as pd  # type: ignore

    if fills is None:
        fills = pd.DataFrame()
    if settlements is None:
        settlements = pd.DataFrame()

    diag = ReconstructionDiagnostics()

    fills = fills.copy()
    settlements = settlements.copy()

    # Parse timestamps into UTC
    fills["ts"] = pd.to_datetime(fills["ts"], utc=True, errors="coerce")
    settlements["settled_ts"] = pd.to_datetime(settlements["settled_ts"], utc=True, errors="coerce")

    # Prefer per-market reconstruction if market_ticker exists.
    group_key = "market_ticker" if "market_ticker" in fills.columns else "ticker"
    fills = fills.dropna(subset=["ts", group_key, "action", "side"]).copy()
    fills = fills.sort_values([group_key, "ts"]).reset_index(drop=True)

    settlements = settlements.dropna(subset=["ticker", "settled_ts", "result"]).copy()
    settlements = settlements.sort_values(["ticker", "settled_ts"]).reset_index(drop=True)

    # One settlement per ticker is typical; if multiple, use the first by time.
    settlement_by_ticker: dict[str, dict[str, Any]] = {}
    for _, r in settlements.iterrows():
        t = str(r["ticker"])
        if t and t not in settlement_by_ticker:
            settlement_by_ticker[t] = {
                "settled_ts": r["settled_ts"],
                "result": str(r["result"]).upper(),
                "payout": float(r.get("payout_per_contract_usd") or 1.0),
            }

    rows: list[dict[str, Any]] = []
    open_rows: list[dict[str, Any]] = []

    for ticker, g in fills.groupby(group_key, sort=False):
        ticker = str(ticker)
        if not ticker:
            continue

        settle = settlement_by_ticker.get(ticker)
        settle_ts = settle.get("settled_ts") if settle else None
        settle_result = settle.get("result") if settle else None
        settle_payout = float(settle.get("payout") or 1.0) if settle else None

        # State for current ticker
        pos = 0
        cur_side: str | None = None
        entry_ts = None
        entry_notional = 0.0
        entry_contracts = 0
        exit_notional = 0.0
        exit_contracts = 0
        fees = 0.0
        strategy_tag = None

        def _close(exit_time, *, closed_by: str) -> None:
            nonlocal pos, cur_side, entry_ts, entry_notional, entry_contracts, exit_notional, exit_contracts, fees, strategy_tag

            if entry_ts is None or cur_side is None or entry_contracts <= 0:
                return

            # If we still have remaining contracts, value them at settlement payout.
            remaining = max(0, int(pos))
            settlement_value = 0.0
            payout_used = None
            if remaining > 0 and settle_result and settle_payout is not None:
                payout_used = float(settle_payout) if str(cur_side).upper() == str(settle_result).upper() else 0.0
                settlement_value = float(remaining) * float(payout_used)

            gross_pnl = float(exit_notional + settlement_value - entry_notional)
            net_pnl = float(gross_pnl - float(fees))

            avg_entry = float(entry_notional / entry_contracts) if entry_contracts else None
            total_exit_value = float(exit_notional + settlement_value)
            avg_exit = float(total_exit_value / entry_contracts) if entry_contracts else None

            rows.append(
                {
                    "ticker": ticker,
                    "side": str(cur_side).upper(),
                    "strategy_tag": strategy_tag or "",
                    "entry_time": entry_ts,
                    "exit_time": exit_time,
                    "holding_time_seconds": float((exit_time - entry_ts).total_seconds()) if exit_time and entry_ts else None,
                    "total_contracts": int(entry_contracts),
                    "avg_entry_price_usd": avg_entry,
                    "avg_exit_price_usd": avg_exit,
                    "gross_pnl_usd": gross_pnl,
                    "total_fees_usd": float(fees),
                    "net_pnl_usd": net_pnl,
                    "closed_by": str(closed_by),
                    "settlement_payout_used_usd": payout_used,
                }
            )

            # Reset state
            pos = 0
            cur_side = None
            entry_ts = None
            entry_notional = 0.0
            entry_contracts = 0
            exit_notional = 0.0
            exit_contracts = 0
            fees = 0.0
            strategy_tag = None

        for _, r in g.iterrows():
            ts = r["ts"]
            if settle_ts is not None and ts is not None and ts > settle_ts:
                # If settlement is earlier than later fills, close before processing post-settlement fills.
                _close(settle_ts, closed_by="settlement")
                diag.settlements_used += 1
                settle_ts = None

            action = str(r.get("action") or "").lower()
            side = str(r.get("side") or "").upper()
            contracts = int(r.get("contracts") or 0)
            price_cents = int(r.get("price_cents") or 0)
            fee = float(r.get("fee_usd") or 0.0)

            if contracts <= 0:
                continue

            fees += float(fee)
            if not strategy_tag:
                try:
                    st = str(r.get("strategy_tag") or "").strip()
                    strategy_tag = st if st else ""
                except Exception:
                    strategy_tag = ""

            price_usd = float(price_cents) / 100.0
            notional = float(price_usd) * float(contracts)

            if action == "buy":
                if pos == 0:
                    entry_ts = ts
                    cur_side = side

                if cur_side is not None and side != str(cur_side).upper():
                    diag.unexpected_side_switches += 1
                    # Start a new round trip; close any existing (should be flat per strategy).
                    _close(ts, closed_by="side_switch")
                    entry_ts = ts
                    cur_side = side

                pos += int(contracts)
                entry_notional += float(notional)
                entry_contracts += int(contracts)

            elif action == "sell":
                if pos <= 0 or entry_ts is None or cur_side is None:
                    diag.unmatched_sells += 1
                    continue

                pos -= int(contracts)
                exit_notional += float(notional)
                exit_contracts += int(contracts)

                if pos <= 0:
                    # Fully flat; close by sells at this timestamp.
                    _close(ts, closed_by="sell")
            else:
                continue

        # Close any remaining position by settlement if present.
        if entry_ts is not None and cur_side is not None and pos > 0 and settle_ts is not None:
            _close(settle_ts, closed_by="settlement")
            diag.settlements_used += 1
            pos = 0

        # If still open after processing: report open position.
        if entry_ts is not None and cur_side is not None and pos > 0:
            open_rows.append(
                {
                    "ticker": ticker,
                    "side": str(cur_side).upper(),
                    "contracts": int(pos),
                    "entry_time": entry_ts,
                    "avg_entry_price_usd": float(entry_notional / entry_contracts) if entry_contracts else None,
                    "fees_to_date_usd": float(fees),
                    "strategy_tag": strategy_tag or "",
                }
            )

    roundtrips_df = pd.DataFrame(rows)
    if not roundtrips_df.empty:
        roundtrips_df["entry_time"] = pd.to_datetime(roundtrips_df["entry_time"], utc=True, errors="coerce")
        roundtrips_df["exit_time"] = pd.to_datetime(roundtrips_df["exit_time"], utc=True, errors="coerce")
        roundtrips_df = roundtrips_df.sort_values(["exit_time", "ticker"]).reset_index(drop=True)

    open_positions_df = pd.DataFrame(open_rows)
    if not open_positions_df.empty:
        open_positions_df["entry_time"] = pd.to_datetime(open_positions_df["entry_time"], utc=True, errors="coerce")
        open_positions_df = open_positions_df.sort_values(["entry_time", "ticker"]).reset_index(drop=True)

    return roundtrips_df, open_positions_df, diag
