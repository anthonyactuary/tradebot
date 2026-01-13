from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings


def _default_kalshi_private_key_path() -> str | None:
    # Default to a repo-local Key.txt if present. This keeps things working if the
    # project folder is moved within a larger repo.
    here = __file__
    try:
        from pathlib import Path

        project_root = Path(here).resolve().parents[2]
        candidate = project_root / "Key.txt"
        if candidate.exists():
            return str(candidate)
    except Exception:
        pass
    return None


class Settings(BaseSettings):
    # Logging
    log_level: str = "INFO"

    # Kalshi
    kalshi_env: str = Field(default="demo", validation_alias="KALSHI_ENV")
    kalshi_base_url: str | None = Field(default=None, validation_alias="KALSHI_BASE_URL")
    kalshi_key_id: str | None = Field(default=None, validation_alias="KALSHI_KEY_ID")
    kalshi_private_key_path: str | None = Field(
        default_factory=_default_kalshi_private_key_path, validation_alias="KALSHI_PRIVATE_KEY_PATH"
    )

    # Bot
    market_tickers_raw: str | None = Field(default=None, validation_alias="MARKET_TICKERS")
    poll_seconds: float = Field(default=2.0, validation_alias="POLL_SECONDS")

    # Alerts
    whale_trade_count: int = Field(default=500, validation_alias="WHALE_TRADE_COUNT")
    whale_book_level_qty: int = Field(default=1000, validation_alias="WHALE_BOOK_LEVEL_QTY")

    # Market maker
    mm_order_count: int = Field(default=5, validation_alias="MM_ORDER_COUNT")
    mm_min_spread_cents: int = Field(default=2, validation_alias="MM_MIN_SPREAD_CENTS")
    mm_improve_cents: int = Field(default=1, validation_alias="MM_IMPROVE_CENTS")
    mm_post_only: bool = Field(default=True, validation_alias="MM_POST_ONLY")

    model_config = {
        "env_file": ".env",
        "extra": "ignore",
    }

    @property
    def market_tickers(self) -> Sequence[str]:
        raw = (self.market_tickers_raw or "").strip()
        if not raw:
            return []
        return [t.strip() for t in raw.split(",") if t.strip()]

    @classmethod
    def load(cls) -> "Settings":
        load_dotenv(override=False)
        return cls()


@dataclass(frozen=True)
class KalshiEnvDefaults:
    base_url: str


def env_defaults(env: str) -> KalshiEnvDefaults:
    env = env.lower().strip()
    if env == "demo":
        return KalshiEnvDefaults(base_url="https://demo-api.kalshi.co")
    if env in {"prod", "production"}:
        # Kalshi production host varies by deployment; this is the docs example host.
        return KalshiEnvDefaults(base_url="https://api.elections.kalshi.com")
    raise ValueError(f"Unknown KALSHI_ENV: {env}")
