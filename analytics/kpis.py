from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd  # type: ignore


@dataclass(frozen=True)
class KpiBundle:
    total_trades: int
    win_rate: float | None

    gross_pnl_usd: float
    net_pnl_usd: float
    total_fees_usd: float

    profit_factor: float | None
    expectancy: float | None

    avg_win: float | None
    avg_loss: float | None
    payoff_ratio: float | None

    max_drawdown_usd: float
    avg_hold_time_seconds: float | None
    pnl_per_hour: float | None


def compute_kpis(*, roundtrips: "pd.DataFrame") -> KpiBundle:
    import pandas as pd  # type: ignore

    if roundtrips is None or roundtrips.empty:
        return KpiBundle(
            total_trades=0,
            win_rate=None,
            gross_pnl_usd=0.0,
            net_pnl_usd=0.0,
            total_fees_usd=0.0,
            profit_factor=None,
            expectancy=None,
            avg_win=None,
            avg_loss=None,
            payoff_ratio=None,
            max_drawdown_usd=0.0,
            avg_hold_time_seconds=None,
            pnl_per_hour=None,
        )

    df = roundtrips.copy()
    for c in ["gross_pnl_usd", "net_pnl_usd", "total_fees_usd", "holding_time_seconds"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(float)
        else:
            df[c] = 0.0

    total = int(df.shape[0])
    net = float(df["net_pnl_usd"].sum())
    gross = float(df["gross_pnl_usd"].sum())
    fees = float(df["total_fees_usd"].sum())

    wins = df[df["net_pnl_usd"] > 0.0]
    losses = df[df["net_pnl_usd"] < 0.0]

    win_rate = float(wins.shape[0] / total) if total else None

    sum_wins = float(wins["net_pnl_usd"].sum())
    sum_losses = float(losses["net_pnl_usd"].sum())

    profit_factor = None
    if sum_losses < 0:
        profit_factor = float(sum_wins / abs(sum_losses))

    expectancy = float(net / total) if total else None

    avg_win = float(wins["net_pnl_usd"].mean()) if wins.shape[0] else None
    avg_loss = float(losses["net_pnl_usd"].mean()) if losses.shape[0] else None

    payoff_ratio = None
    if avg_win is not None and avg_loss is not None and avg_loss < 0:
        payoff_ratio = float(avg_win / abs(avg_loss))

    avg_hold = float(df["holding_time_seconds"].mean()) if total else None

    pnl_per_hour = None
    total_hold = float(df["holding_time_seconds"].sum())
    if total_hold > 0:
        pnl_per_hour = float(net / (total_hold / 3600.0))

    # Max drawdown from realized equity curve
    from analytics.equity_curve import build_equity_curve, max_drawdown

    eq = build_equity_curve(roundtrips=df)
    dd = max_drawdown(equity_curve=eq)

    return KpiBundle(
        total_trades=total,
        win_rate=win_rate,
        gross_pnl_usd=gross,
        net_pnl_usd=net,
        total_fees_usd=fees,
        profit_factor=profit_factor,
        expectancy=expectancy,
        avg_win=avg_win,
        avg_loss=avg_loss,
        payoff_ratio=payoff_ratio,
        max_drawdown_usd=float(dd.max_drawdown),
        avg_hold_time_seconds=avg_hold,
        pnl_per_hour=pnl_per_hour,
    )
