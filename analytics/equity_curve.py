from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd  # type: ignore


@dataclass(frozen=True)
class DrawdownResult:
    max_drawdown: float
    peak_equity: float
    trough_equity: float


def build_equity_curve(*, roundtrips: "pd.DataFrame") -> "pd.DataFrame":
    """Build realized equity curve from net PnL.

    Expects columns:
    - exit_time (datetime)
    - net_pnl_usd (float)

    Returns a DataFrame with:
    - exit_time
    - net_pnl_usd
    - equity
    - peak
    - drawdown
    """

    import pandas as pd  # type: ignore

    if roundtrips is None or roundtrips.empty:
        return pd.DataFrame(columns=["exit_time", "net_pnl_usd", "equity", "peak", "drawdown"])

    df = roundtrips[["exit_time", "net_pnl_usd"]].copy()
    df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True, errors="coerce")
    df["net_pnl_usd"] = pd.to_numeric(df["net_pnl_usd"], errors="coerce").fillna(0.0).astype(float)

    df = df.dropna(subset=["exit_time"]).sort_values("exit_time").reset_index(drop=True)
    df["equity"] = df["net_pnl_usd"].cumsum()
    df["peak"] = df["equity"].cummax()
    df["drawdown"] = df["equity"] - df["peak"]
    return df


def max_drawdown(*, equity_curve: "pd.DataFrame") -> DrawdownResult:
    import pandas as pd  # type: ignore

    if equity_curve is None or equity_curve.empty:
        return DrawdownResult(max_drawdown=0.0, peak_equity=0.0, trough_equity=0.0)

    dd = equity_curve["drawdown"].min()
    dd = float(dd) if dd is not None else 0.0

    # Best-effort peak/trough
    peak = float(equity_curve["peak"].max()) if "peak" in equity_curve.columns else 0.0
    trough = float(equity_curve["equity"].min()) if "equity" in equity_curve.columns else 0.0

    return DrawdownResult(max_drawdown=abs(dd), peak_equity=peak, trough_equity=trough)
