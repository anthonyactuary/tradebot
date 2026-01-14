"""Kalshi BTC 15m market maker (model-driven fair value).

This strategy polls active BTC 15m markets, computes a fair probability for YES
using the existing 15m direction model, and posts maker (GTC, post-only) quotes.

Key safety features:
- Uses `fetch_inventory_summary()` to compute signed position per ticker
  (positive=long YES, negative=long NO).
- Cancels/replace outstanding quotes per ticker.
- Applies an inventory-based skew to tilt quotes toward flattening.
- Enforces an absolute position cap per ticker.

Run:
	# From tradebot/ (works without installing the package)
	python src/tradebot/strategies/market_maker.py --model-dir src/tradebot/data/btc15m_model_coinbase_80d_purged30

	# If you prefer `-m`, install the src-layout package first:
	#   python -m pip install -e .
	#   python -m tradebot.strategies.market_maker --model-dir src/tradebot/data/btc15m_model_coinbase_80d_purged30
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import math
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
log = logging.getLogger(__name__)


# Allow running this file directly ("python src/tradebot/strategies/market_maker.py")
# in a src-layout repo by ensuring the src/ root is on sys.path.
_THIS_FILE = Path(__file__).resolve()
_SRC_ROOT = _THIS_FILE.parents[2]  # .../src
if (_SRC_ROOT / "tradebot").is_dir():
	src_root_str = str(_SRC_ROOT)
	if src_root_str not in sys.path:
		sys.path.insert(0, src_root_str)


from tradebot.config import Settings
from tradebot.kalshi.client import KalshiClient
from tradebot.tools.btc15m_live_inference import (
	build_feature_dict,
	fetch_recent_1m_closes,
	load_model,
	predict_probability,
)
from tradebot.tools.inventory_check import fetch_inventory_summary
from tradebot.tools.kalshi_market_poll import MarketSnapshot, poll_once


def _clamp_price_cents(v: float | int) -> int:
	try:
		c = int(round(float(v)))
	except Exception:
		c = 50
	return max(1, min(99, int(c)))


def _max_count_for_action(*, pos: int, max_abs_pos: int, action: str) -> int:
	"""Max count such that resulting |pos'| <= max_abs_pos.

Position convention matches `inventory_check` / `order_execution`:
	buy YES  -> pos += count
	sell YES -> pos -= count

We implement market making by quoting YES book only.
"""

	max_abs_pos = int(max_abs_pos)
	pos = int(pos)
	if max_abs_pos <= 0:
		return 0

	if action == "buy":
		# pos' = pos + c => c <= max_abs_pos - pos
		return max(0, int(max_abs_pos - pos))

	# action == "sell"
	# pos' = pos - c => c <= pos + max_abs_pos
	return max(0, int(pos + max_abs_pos))


def _parse_iso8601_utc(ts: object) -> datetime | None:
	"""Best-effort ISO8601 parser for Kalshi timestamps.

	Expected shapes include: "2026-01-14T02:37:23.047192Z".
	Returns an aware datetime in UTC.
	"""
	if ts is None:
		return None
	s = str(ts).strip()
	if not s:
		return None
	try:
		# Handle trailing 'Z' (UTC) for Python's fromisoformat.
		if s.endswith("Z"):
			s = s[:-1] + "+00:00"
		dt = datetime.fromisoformat(s)
		if dt.tzinfo is None:
			# Assume UTC if missing tz.
			return dt.replace(tzinfo=timezone.utc)
		return dt.astimezone(timezone.utc)
	except Exception:
		return None


def _cfg_default(name: str):
	field = MarketMakerConfig.__dataclass_fields__.get(name)  # type: ignore[attr-defined]
	if field is None:
		raise KeyError(f"Unknown MarketMakerConfig field: {name}")
	return field.default


def _is_post_only_cross_error(exc: BaseException) -> bool:
	s = str(exc).lower()
	return "post only cross" in s or "post_only_cross" in s


@dataclass(frozen=True)
class MarketMakerConfig:
	model_dir: str

	# Polling
	asset: str = "BTC"
	horizon_minutes: int = 600
	limit_markets: int = 1
	min_seconds_to_expiry: int = 30
	poll_interval_sec: float = 2
	duration_sec: float = 0.0

	# Quoting / spread shaping
	base_spread_cents: int = 6
	min_spread_cents: int = 5
	max_spread_cents: int = 20

	# Direction / regime
	one_sided_threshold: float = 0.60
	one_sided_threshold_low: float = 0.40
	deadzone_neutral_low: float = 0.45
	deadzone_neutral_high: float = 0.55

	# Momentum pause
	momentum_pause_abs_return_1m: float = 0.0015
	momentum_pause_seconds: int = 10

	# Expiry controls
	force_widen_tte_sec: int = 120
	force_stop_tte_sec: int = 30

	# Order placement
	quote_size: int = 10
	post_only: bool = True
	time_in_force: str = "good_till_canceled"
	dry_run: bool = False
	min_quote_refresh_seconds: int = 2
	cancel_before_replace: bool = True

	# Inventory controls
	max_abs_pos: int = 25
	inventory_soft_limit: int = 10
	inventory_hard_limit: int = 20
	inventory_skew_cents_per_contract: float = 0.5
	inventory_skew_quadratic: bool = True

	# Monitoring
	monitor_interval_sec: float = 30.0
	fills_limit: int = 10


class KalshiMarketMaker:
	def __init__(self, *, client: KalshiClient, cfg: MarketMakerConfig):
		self.client = client
		self.cfg = cfg
		self.model, self.feature_names = load_model(str(cfg.model_dir))
		self.started_at_utc = datetime.now(timezone.utc)

		# Momentum pause state (shared across tickers since all markets share BTC spot).
		self._spot_history: list[tuple[float, float]] = []  # (monotonic_ts, spot_usd)
		self._pause_until_ts: float = 0.0
		self._last_pause_logged_until_ts: float = 0.0

		# Track open quote order_ids by ticker (YES book only).
		self.open_orders: dict[str, dict[str, str]] = {}
		self._last_polled_tickers: list[str] = []
		self._seen_fill_ids: list[str] = []
		self._seen_fill_id_set: set[str] = set()
		self._last_realized_pnl_cents_by_ticker: dict[str, float] = {}
		self._last_fair_yes_cents_by_ticker: dict[str, int] = {}
		self._last_pos_by_ticker: dict[str, int] = {}
		self._last_quote_ts_by_ticker: dict[str, float] = {}
		self._last_desired_quote_by_ticker: dict[str, tuple[int | None, int | None]] = {}
		self._last_block_log_by_ticker: dict[str, tuple[str, float]] = {}

	def _tickers_to_monitor(self) -> list[str]:
		# Union of tickers we're currently quoting and tickers we've recently polled.
		t = set(self.open_orders.keys())
		for x in self._last_polled_tickers:
			if x:
				t.add(str(x))
		return sorted(t)

	async def monitor_fills(self) -> None:
		"""Background monitor: recent fills + PnL from positions API."""

		while True:
			try:
				# ---- Recent fills (best-effort) ----
				fills_payload = await self.client.get_fills(limit=int(self.cfg.fills_limit))
				fills = list(fills_payload.get("fills") or [])
				new_fills: list[dict] = []
				for f in fills:
					if not isinstance(f, dict):
						continue
					fid = str(f.get("fill_id") or "").strip()
					if not fid or fid in self._seen_fill_id_set:
						continue

					created_time = f.get("created_time")
					dt = _parse_iso8601_utc(created_time)
					# Mark fills older than bot start as seen (warmup), but don't log them.
					if dt is not None and dt < self.started_at_utc:
						self._seen_fill_ids.append(fid)
						self._seen_fill_id_set.add(fid)
						continue

					self._seen_fill_ids.append(fid)
					self._seen_fill_id_set.add(fid)
					new_fills.append(f)

				# Trim memory.
				if len(self._seen_fill_ids) > 200:
					keep = self._seen_fill_ids[-100:]
					self._seen_fill_ids = keep
					self._seen_fill_id_set = set(keep)

				if new_fills:
					# Log oldest->newest for readability.
					for f in reversed(new_fills):
						ticker = str(f.get("ticker") or "")
						action = str(f.get("action") or "")
						side = str(f.get("side") or "")
						count = int(f.get("count") or 0)
						price_cents = int(f.get("price") or 0)
						is_taker = f.get("is_taker")
						created_time = f.get("created_time")
						sign = 1.0 if action == "sell" else -1.0
						cashflow_usd = sign * (float(count) * float(price_cents) / 100.0)
						log.info(
							"MM_FILL %s action=%s side=%s count=%d price=%dc cashflow=%+.2f is_taker=%s ts=%s",
							ticker,
							action,
							side,
							int(count),
							int(price_cents),
							float(cashflow_usd),
							str(is_taker),
							str(created_time),
						)

				# ---- Positions + realized PnL ----
				tickers = self._tickers_to_monitor()
				if tickers:
					pos_payload = await self.client.get_positions(limit=1000)
					mps = list(pos_payload.get("market_positions") or [])
					tset = set(tickers)

					realized_total_cents = 0.0
					exposure_total_cents = 0.0
					pos_nonzero = 0

					for mp in mps:
						if not isinstance(mp, dict):
							continue
						ticker = str(mp.get("ticker") or "").strip()
						if not ticker or ticker not in tset:
							continue

						try:
							pos = int(mp.get("position") or 0)
						except Exception:
							pos = 0
						try:
							exposure_cents = float(mp.get("market_exposure") or 0.0)
						except Exception:
							exposure_cents = 0.0
						try:
							realized_cents = float(mp.get("realized_pnl") or 0.0)
						except Exception:
							realized_cents = 0.0

						realized_total_cents += float(realized_cents)
						exposure_total_cents += abs(float(exposure_cents))
						if int(pos) != 0:
							pos_nonzero += 1

						last = float(self._last_realized_pnl_cents_by_ticker.get(ticker, 0.0))
						delta = float(realized_cents) - float(last)
						self._last_realized_pnl_cents_by_ticker[ticker] = float(realized_cents)

						# Only log per-ticker if there's something interesting.
						if int(pos) != 0 or abs(float(delta)) >= 0.01:
							log.info(
								"MM_POS %s pos=%d exposure=%.2f realized_pnl=%.2f delta=%.2f",
								ticker,
								int(pos),
								float(abs(exposure_cents)) / 100.0,
								float(realized_cents) / 100.0,
								float(delta) / 100.0,
							)

					log.info(
						"MM_PNL tickers=%d nonzero_pos=%d realized_total=%.2f exposure_total=%.2f",
						int(len(tickers)),
						int(pos_nonzero),
						float(realized_total_cents) / 100.0,
						float(exposure_total_cents) / 100.0,
					)
			except asyncio.CancelledError:
				raise
			except Exception as e:
				log.warning("MM_MONITOR_ERROR %s", e)

			await asyncio.sleep(float(self.cfg.monitor_interval_sec))

	def _compute_return_1m_proxy(self, *, now_ts: float, spot_usd: float) -> float | None:
		"""Return proxy over ~1m based on recent spot observations."""
		spot_usd = float(spot_usd)
		if not math.isfinite(spot_usd) or spot_usd <= 0:
			return None
		# Add current observation.
		self._spot_history.append((float(now_ts), float(spot_usd)))
		# Prune old.
		cutoff = float(now_ts) - 120.0
		while self._spot_history and float(self._spot_history[0][0]) < cutoff:
			self._spot_history.pop(0)
		# Find a reference point ~60s ago.
		ref_spot = None
		for ts, px in self._spot_history:
			if float(ts) <= float(now_ts) - 60.0:
				ref_spot = float(px)
		# If we don't have a 60s-old point yet, use the oldest available.
		if ref_spot is None and self._spot_history:
			ref_spot = float(self._spot_history[0][1])
		if ref_spot is None or ref_spot <= 0:
			return None
		return float(spot_usd / ref_spot - 1.0)

	def _log_block(
		self,
		*,
		ticker: str,
		reason: str,
		now_ts: float,
		p_yes: float | None = None,
		pos: int | None = None,
		fair_yes_cents: int | None = None,
		desired_bid_yes: int | None = None,
		desired_ask_yes: int | None = None,
		snap: MarketSnapshot | None = None,
	) -> None:
		"""Structured block log (rate-limited) for grepability."""
		ticker = str(ticker)
		reason = str(reason)
		prev = self._last_block_log_by_ticker.get(ticker)
		if prev is not None:
			prev_reason, prev_ts = prev
			# Log if reason changed or last log was >5s ago.
			if str(prev_reason) == reason and float(now_ts - float(prev_ts)) < 5.0:
				return
		self._last_block_log_by_ticker[ticker] = (reason, float(now_ts))

		tte = None
		best_bid = None
		best_ask = None
		spot = None
		strike = None
		if snap is not None:
			tte = snap.seconds_to_expiry
			best_bid = snap.best_yes_bid
			best_ask = snap.best_yes_ask
			spot = snap.btc_spot_usd
			strike = snap.price_to_beat

		log.info(
			"MM_BLOCK %s reason=%s p_yes=%s pos=%s fair_yes=%s bid=%s ask=%s best_bid=%s best_ask=%s spot=%s strike=%s tte=%s",
			ticker,
			reason,
			"-" if p_yes is None else f"{float(p_yes):.3f}",
			"-" if pos is None else str(int(pos)),
			"-" if fair_yes_cents is None else str(int(fair_yes_cents)),
			"-" if desired_bid_yes is None else str(int(desired_bid_yes)),
			"-" if desired_ask_yes is None else str(int(desired_ask_yes)),
			"-" if best_bid is None else str(int(best_bid)),
			"-" if best_ask is None else str(int(best_ask)),
			"-" if spot is None else f"{float(spot):.2f}",
			"-" if strike is None else f"{float(strike):.2f}",
			"-" if tte is None else str(int(tte)),
		)

	async def cancel_all_open_quotes(self) -> None:
		for ticker in list(self.open_orders.keys()):
			try:
				await self.cancel_open_quotes(str(ticker))
			except Exception:
				pass

	def _desired_sides_for_probability(self, *, p_yes: float) -> tuple[bool, bool, str]:
		"""Return (quote_buy_yes, quote_sell_yes, mode)."""
		p = float(p_yes)
		if p >= float(self.cfg.one_sided_threshold):
			return True, False, "p_high"
		if p <= float(self.cfg.one_sided_threshold_low):
			return False, True, "p_low"
		if float(self.cfg.deadzone_neutral_low) <= p <= float(self.cfg.deadzone_neutral_high):
			return True, True, "neutral"
		# Conservative bands: only quote the side consistent with probability.
		if p > float(self.cfg.deadzone_neutral_high):
			return True, False, "tilt_buy"
		return False, True, "tilt_sell"

	def _compute_inventory_skew_cents(self, *, pos: int) -> int:
		k = abs(int(pos))
		if k <= 0:
			return 0
		base = float(self.cfg.inventory_skew_cents_per_contract)
		if bool(self.cfg.inventory_skew_quadratic):
			mag = base * float(k * k)
		else:
			mag = base * float(k)
		skew = float(mag) if int(pos) > 0 else -float(mag)
		return int(round(skew))

	def _compute_dynamic_spread_cents(self, *, snap: MarketSnapshot, pos: int) -> tuple[int, str | None]:
		"""Compute spread based on base, inventory, and time-to-expiry widening."""
		reason: list[str] = []
		spread = int(self.cfg.base_spread_cents)

		# Inventory-based widening (soft limit).
		abs_pos = abs(int(pos))
		soft = int(self.cfg.inventory_soft_limit)
		hard = int(self.cfg.inventory_hard_limit)
		if abs_pos >= soft and hard > soft:
			# Add +2..+6 cents as abs(pos) approaches hard.
			frac = min(1.0, max(0.0, float(abs_pos - soft) / float(hard - soft)))
			extra = int(round(2.0 + 4.0 * float(frac)))
			spread += int(extra)
			reason.append(f"inv+{extra}")

		# Time-to-expiry widening.
		tte = snap.seconds_to_expiry
		if tte is not None:
			try:
				tte_i = int(tte)
			except Exception:
				tte_i = None
			if tte_i is not None and tte_i > 0 and int(self.cfg.force_widen_tte_sec) > 0:
				if tte_i < int(self.cfg.force_widen_tte_sec):
					# Linearly widen toward max as expiry approaches.
					wfrac = 1.0 - float(tte_i) / float(self.cfg.force_widen_tte_sec)
					target = float(self.cfg.base_spread_cents) + float(wfrac) * float(
						int(self.cfg.max_spread_cents) - int(self.cfg.base_spread_cents)
					)
					widen = int(round(max(float(spread), float(target))))
					if widen > spread:
						reason.append("tte")
						spread = int(widen)

		# Clamp to [min, max].
		spread = max(int(self.cfg.min_spread_cents), int(spread))
		spread = min(int(self.cfg.max_spread_cents), int(spread))
		return int(spread), (" ".join(reason) if reason else None)

	def _should_refresh_quotes(
		self,
		*,
		ticker: str,
		desired_bid_yes: int | None,
		desired_ask_yes: int | None,
		now_ts: float,
	) -> bool:
		"""Anti-spam refresh guard.

		Refresh if:
		- desired prices/sides changed by >= 1c, OR
		- last update older than min_quote_refresh_seconds.
		"""
		last = self._last_desired_quote_by_ticker.get(str(ticker))
		last_ts = self._last_quote_ts_by_ticker.get(str(ticker))
		if last is None or last_ts is None:
			return True
		age = float(now_ts - float(last_ts))
		if age >= float(int(self.cfg.min_quote_refresh_seconds)):
			return True
		last_bid, last_ask = last
		# Side toggles count as a change.
		if (last_bid is None) != (desired_bid_yes is None) or (last_ask is None) != (desired_ask_yes is None):
			return True
		# Price deltas.
		if desired_bid_yes is not None and last_bid is not None and abs(int(desired_bid_yes) - int(last_bid)) >= 1:
			return True
		if desired_ask_yes is not None and last_ask is not None and abs(int(desired_ask_yes) - int(last_ask)) >= 1:
			return True
		return False

	async def get_fair_prob_yes(self, snap: MarketSnapshot) -> float:
		"""Return model fair P(YES) in [0, 1] for this market snapshot."""

		if snap.price_to_beat is None or snap.btc_spot_usd is None or snap.seconds_to_expiry is None:
			raise ValueError("missing snap fields")

		closes = await fetch_recent_1m_closes(limit=6)
		feats = build_feature_dict(
			price_to_beat=float(snap.price_to_beat),
			btc_spot_usd=float(snap.btc_spot_usd),
			seconds_to_expiry=int(snap.seconds_to_expiry),
			recent_closes_1m=closes,
		)
		return float(predict_probability(self.model, self.feature_names, feats))

	async def get_position(self, ticker: str) -> int:
		"""Fetch signed position for ticker (positive=YES, negative=NO)."""

		inv = await fetch_inventory_summary(client=self.client, tickers=[str(ticker)])
		ti = inv.per_ticker.get(str(ticker))
		if ti is None:
			return 0
		return int(ti.position)

	async def cancel_open_quotes(self, ticker: str) -> None:
		"""Cancel any tracked open quote orders for this ticker."""

		orders = dict(self.open_orders.get(ticker, {}))
		if not orders:
			return

		if bool(self.cfg.dry_run):
			# In dry-run we only track fake ids; never call the real cancel endpoint.
			self.open_orders[ticker] = {}
			return

		for label, order_id in orders.items():
			if not order_id:
				continue
			try:
				await self.client.cancel_order(order_id=str(order_id))
			except Exception as e:
				# Best-effort; the order may already be filled/canceled.
				log.debug("CANCEL_QUOTE_ERROR %s label=%s order_id=%s error=%s", ticker, label, order_id, e)

		self.open_orders[ticker] = {}

	async def place_yes_quote(self, *, ticker: str, action: str, price_cents: int, count: int) -> str | None:
		"""Place a YES-side limit order (GTC)."""

		if int(count) <= 0:
			return None

		if bool(self.cfg.dry_run):
			log.info(
				"DRY_RUN_QUOTE %s action=%s side=YES price=%dc count=%d",
				str(ticker),
				str(action),
				int(price_cents),
				int(count),
			)
			return f"dry-{uuid.uuid4()}"

		try:
			resp = await self.client.create_order(
				ticker=str(ticker),
				side="yes",
				action=str(action),  # "buy" or "sell"
				count=int(count),
				order_type="limit",
				yes_price=int(price_cents),
				no_price=None,
				client_order_id=str(uuid.uuid4()),
				post_only=bool(self.cfg.post_only),
				reduce_only=None,
				time_in_force=str(self.cfg.time_in_force),
			)
		except Exception as e:
			# This can happen if the book moves between our snapshot and order placement.
			# Treat as non-fatal; we'll refresh next poll.
			if bool(self.cfg.post_only) and _is_post_only_cross_error(e):
				log.info(
					"MM_POSTONLY_CROSS %s action=%s side=YES price=%dc count=%d",
					str(ticker),
					str(action),
					int(price_cents),
					int(count),
				)
				return None
			raise

		# Response shape varies; accept either style.
		order = resp.get("order") if isinstance(resp.get("order"), dict) else None
		if order and isinstance(order.get("order_id"), str):
			return str(order.get("order_id"))
		if isinstance(resp.get("order_id"), str):
			return str(resp.get("order_id"))
		if isinstance(resp.get("id"), str):
			return str(resp.get("id"))
		return None

	async def quote_market(self, snap: MarketSnapshot) -> None:
		ticker = str(snap.ticker)
		loop = asyncio.get_running_loop()
		now_ts = float(loop.time())

		# Global momentum pause.
		if float(self._pause_until_ts) > 0.0 and float(now_ts) < float(self._pause_until_ts):
			# Avoid spamming a per-ticker block log here; run() emits the pause log + cancels.
			return

		# Expiry kill-switch.
		if snap.seconds_to_expiry is not None:
			try:
				tte = int(snap.seconds_to_expiry)
			except Exception:
				tte = None
			if tte is not None and tte < int(self.cfg.force_stop_tte_sec):
				await self.cancel_open_quotes(ticker)
				self._log_block(
					ticker=ticker,
					reason="tte_stop",
					now_ts=float(now_ts),
					snap=snap,
				)
				log.info(
					"MM_STOP_TTE %s tte=%ss < force_stop_tte_sec=%ss",
					ticker,
					str(int(tte)),
					str(int(self.cfg.force_stop_tte_sec)),
				)
				return

		# Position + limits
		pos = await self.get_position(ticker)
		if abs(int(pos)) >= int(self.cfg.max_abs_pos):
			# At cap: only quote the side that reduces abs(pos).
			log.info("MM_POS_CAP %s pos=%d max_abs_pos=%d", ticker, int(pos), int(self.cfg.max_abs_pos))

		# Fair price from model
		p_yes = await self.get_fair_prob_yes(snap)
		fair_yes_cents = _clamp_price_cents(float(p_yes) * 100.0)

		# Direction-aware quoting (probability bands).
		quote_buy, quote_sell, p_mode = self._desired_sides_for_probability(p_yes=float(p_yes))
		if p_mode != "neutral":
			log.info("MM_ONE_SIDED %s p_yes=%.3f mode=%s", ticker, float(p_yes), str(p_mode))

		# Inventory hard limits: block the side that increases inventory.
		if int(pos) >= int(self.cfg.inventory_hard_limit):
			if quote_buy:
				log.info(
					"MM_INVENTORY_BLOCK %s side=BUY_YES pos=%d >= hard=%d",
					ticker,
					int(pos),
					int(self.cfg.inventory_hard_limit),
				)
			quote_buy = False
		if int(pos) <= -int(self.cfg.inventory_hard_limit):
			if quote_sell:
				log.info(
					"MM_INVENTORY_BLOCK %s side=SELL_YES pos=%d <= -hard=%d",
					ticker,
					int(pos),
					int(self.cfg.inventory_hard_limit),
				)
			quote_sell = False

		# Inventory skew: shift fair to encourage flattening.
		skew_cents = self._compute_inventory_skew_cents(pos=int(pos))
		if int(skew_cents) != 0:
			log.info(
				"MM_INVENTORY_SKEW %s pos=%d skew=%+dc quadratic=%s",
				ticker,
				int(pos),
				int(skew_cents),
				str(bool(self.cfg.inventory_skew_quadratic)),
			)
		# If long YES (pos>0), skew>0 => subtract moves fair down (discourage buying more YES).
		fair_yes_cents = _clamp_price_cents(float(fair_yes_cents) - float(skew_cents))

		self._last_fair_yes_cents_by_ticker[ticker] = int(fair_yes_cents)
		self._last_pos_by_ticker[ticker] = int(pos)

		# Spread shaping (inventory + expiry widening).
		spread_cents, spread_reason = self._compute_dynamic_spread_cents(snap=snap, pos=int(pos))
		if spread_reason and snap.seconds_to_expiry is not None and "tte" in spread_reason:
			try:
				tte = int(snap.seconds_to_expiry)
			except Exception:
				tte = None
			log.info(
				"MM_WIDEN_TTE %s tte=%s spread=%dc reason=%s",
				ticker,
				"?" if tte is None else str(int(tte)),
				int(spread_cents),
				str(spread_reason),
			)

		half_spread = max(float(int(spread_cents)) / 2.0, 0.5)
		desired_bid_yes = _clamp_price_cents(float(fair_yes_cents) - float(half_spread)) if quote_buy else None
		desired_ask_yes = _clamp_price_cents(float(fair_yes_cents) + float(half_spread)) if quote_sell else None

		# If quoting both sides, enforce at least 1c gap.
		if desired_bid_yes is not None and desired_ask_yes is not None and int(desired_ask_yes) <= int(desired_bid_yes):
			desired_ask_yes = min(99, int(desired_bid_yes) + 1)

		# Post-only safety clamping against current best YES book.
		best_yes_bid = snap.best_yes_bid
		best_yes_ask = snap.best_yes_ask
		orig_bid = desired_bid_yes
		orig_ask = desired_ask_yes
		if desired_bid_yes is not None and best_yes_ask is not None:
			desired_bid_yes = max(1, min(int(desired_bid_yes), int(best_yes_ask) - 1))
			if int(desired_bid_yes) >= int(best_yes_ask):
				desired_bid_yes = None
		if desired_ask_yes is not None and best_yes_bid is not None:
			desired_ask_yes = min(99, max(int(desired_ask_yes), int(best_yes_bid) + 1))
			if int(desired_ask_yes) <= int(best_yes_bid):
				desired_ask_yes = None
		if (orig_bid != desired_bid_yes) or (orig_ask != desired_ask_yes):
			log.debug(
				"MM_POSTONLY_ADJUST %s best_bid=%s best_ask=%s bid=%s->%s ask=%s->%s",
				ticker,
				"-" if best_yes_bid is None else str(int(best_yes_bid)),
				"-" if best_yes_ask is None else str(int(best_yes_ask)),
				"-" if orig_bid is None else str(int(orig_bid)),
				"-" if desired_bid_yes is None else str(int(desired_bid_yes)),
				"-" if orig_ask is None else str(int(orig_ask)),
				"-" if desired_ask_yes is None else str(int(desired_ask_yes)),
			)

		if desired_bid_yes is None and desired_ask_yes is None:
			# If our policy says to quote nothing (or post-only room is gone), ensure no stale quotes remain.
			await self.cancel_open_quotes(ticker)
			self._log_block(
				ticker=ticker,
				reason="no_quote_policy_or_postonly",
				now_ts=float(now_ts),
				p_yes=float(p_yes),
				pos=int(pos),
				fair_yes_cents=int(fair_yes_cents),
				desired_bid_yes=None,
				desired_ask_yes=None,
				snap=snap,
			)
			log.info(
				"MM_NO_QUOTE %s p_yes=%.3f pos=%d fair_yes=%dc",
				ticker,
				float(p_yes),
				int(pos),
				int(fair_yes_cents),
			)
			return

		# Anti-spam refresh guard.
		if not self._should_refresh_quotes(
			ticker=ticker,
			desired_bid_yes=(int(desired_bid_yes) if desired_bid_yes is not None else None),
			desired_ask_yes=(int(desired_ask_yes) if desired_ask_yes is not None else None),
			now_ts=float(now_ts),
		):
			log.debug("MM_QUOTE_SKIP %s min_refresh not reached", ticker)
			return

		prev_orders = dict(self.open_orders.get(ticker, {}))
		if bool(self.cfg.cancel_before_replace):
			await self.cancel_open_quotes(ticker)

		# Size each side so fills cannot violate the abs position cap.
		max_buy = _max_count_for_action(pos=int(pos), max_abs_pos=int(self.cfg.max_abs_pos), action="buy")
		max_sell = _max_count_for_action(pos=int(pos), max_abs_pos=int(self.cfg.max_abs_pos), action="sell")

		buy_size = 0
		sell_size = 0
		if desired_bid_yes is not None and quote_buy:
			buy_size = min(int(self.cfg.quote_size), int(max_buy))
		if desired_ask_yes is not None and quote_sell:
			sell_size = min(int(self.cfg.quote_size), int(max_sell))

		if int(buy_size) <= 0:
			desired_bid_yes = None
		if int(sell_size) <= 0:
			desired_ask_yes = None

		if desired_bid_yes is None and desired_ask_yes is None:
			await self.cancel_open_quotes(ticker)
			self._log_block(
				ticker=ticker,
				reason="no_quote_size_or_limits",
				now_ts=float(now_ts),
				p_yes=float(p_yes),
				pos=int(pos),
				fair_yes_cents=int(fair_yes_cents),
				desired_bid_yes=None,
				desired_ask_yes=None,
				snap=snap,
			)
			log.info(
				"MM_NO_QUOTE %s reason=size_or_limits p_yes=%.3f pos=%d",
				ticker,
				float(p_yes),
				int(pos),
			)
			return

		yes_bid_id = None
		yes_ask_id = None
		if desired_bid_yes is not None:
			yes_bid_id = await self.place_yes_quote(
				ticker=ticker,
				action="buy",
				price_cents=int(desired_bid_yes),
				count=int(buy_size),
			)
		if desired_ask_yes is not None:
			yes_ask_id = await self.place_yes_quote(
				ticker=ticker,
				action="sell",
				price_cents=int(desired_ask_yes),
				count=int(sell_size),
			)

		# If we didn't cancel-before-replace, clean up prior orders after placing new ones.
		if not bool(self.cfg.cancel_before_replace) and prev_orders and not bool(self.cfg.dry_run):
			for oid in [prev_orders.get("yes_bid_id"), prev_orders.get("yes_ask_id")]:
				if oid:
					try:
						await self.client.cancel_order(order_id=str(oid))
					except Exception:
						pass

		self.open_orders[ticker] = {
			"yes_bid_id": str(yes_bid_id or ""),
			"yes_ask_id": str(yes_ask_id or ""),
		}
		self._last_quote_ts_by_ticker[ticker] = float(now_ts)
		self._last_desired_quote_by_ticker[ticker] = (
			int(desired_bid_yes) if desired_bid_yes is not None else None,
			int(desired_ask_yes) if desired_ask_yes is not None else None,
		)

		log.info(
			"MM_QUOTE %s pos=%d p_yes=%.3f fair_yes=%dc spread=%dc bid=%s(%d) ask=%s(%d) spot=%s strike=%s tte=%ss",
			ticker,
			int(pos),
			float(p_yes),
			int(fair_yes_cents),
			int(spread_cents),
			"-" if desired_bid_yes is None else str(int(desired_bid_yes)),
			int(buy_size),
			"-" if desired_ask_yes is None else str(int(desired_ask_yes)),
			int(sell_size),
			"-" if snap.btc_spot_usd is None else f"${float(snap.btc_spot_usd):.2f}",
			"-" if snap.price_to_beat is None else f"${float(snap.price_to_beat):.2f}",
			"?" if snap.seconds_to_expiry is None else str(int(snap.seconds_to_expiry)),
		)

	async def run(self) -> None:
		monitor_task = asyncio.create_task(self.monitor_fills())
		loop = asyncio.get_running_loop()
		start_ts = float(loop.time())
		try:
			while True:
				now_ts = float(loop.time())
				elapsed = float(now_ts - float(start_ts))
				if float(self.cfg.duration_sec) > 0.0 and float(elapsed) >= float(self.cfg.duration_sec):
					log.info("MM_STOP reason=duration_sec elapsed=%.1fs", float(elapsed))
					return

				# If currently paused due to momentum, keep canceling stale quotes and wait it out.
				if float(self._pause_until_ts) > 0.0 and float(now_ts) < float(self._pause_until_ts):
					if float(self._last_pause_logged_until_ts) != float(self._pause_until_ts):
						log.info(
							"MM_PAUSE_MOMENTUM pause_for=%ss",
							str(int(max(0.0, float(self._pause_until_ts - float(now_ts))))),
						)
						self._last_pause_logged_until_ts = float(self._pause_until_ts)
					await self.cancel_all_open_quotes()
					await asyncio.sleep(float(self.cfg.poll_interval_sec))
					continue

				try:
					snaps = await poll_once(
						self.client,
						asset=str(self.cfg.asset),
						horizon_minutes=int(self.cfg.horizon_minutes),
						limit_markets=int(self.cfg.limit_markets),
						min_seconds_to_expiry=int(self.cfg.min_seconds_to_expiry),
					)

					# Momentum / volatility pause based on BTC spot movement.
					spots = [float(s.btc_spot_usd) for s in snaps if s.btc_spot_usd is not None]
					if spots:
						spot = float(sorted(spots)[len(spots) // 2])
						r1m = self._compute_return_1m_proxy(now_ts=float(now_ts), spot_usd=float(spot))
						if r1m is not None and abs(float(r1m)) >= float(self.cfg.momentum_pause_abs_return_1m):
							self._pause_until_ts = float(now_ts) + float(int(self.cfg.momentum_pause_seconds))
							self._last_pause_logged_until_ts = 0.0
							log.info(
								"MM_PAUSE_MOMENTUM r1m=%.5f thr=%.5f pause=%ss spot=%.2f",
								float(r1m),
								float(self.cfg.momentum_pause_abs_return_1m),
								str(int(self.cfg.momentum_pause_seconds)),
								float(spot),
							)
							await self.cancel_all_open_quotes()
							await asyncio.sleep(float(self.cfg.poll_interval_sec))
							continue

					self._last_polled_tickers = [str(s.ticker) for s in snaps]

					# poll_once() already returns open BTC15m markets; no snap.status field.
					tasks = [self.quote_market(s) for s in snaps]
					results = await asyncio.gather(*tasks, return_exceptions=True)
					for r in results:
						if isinstance(r, Exception):
							log.warning("MM_QUOTE_ERROR error=%s", r)
				except asyncio.CancelledError:
					raise
				except Exception as e:
					log.exception("MM_LOOP_ERROR %s", e)

				await asyncio.sleep(float(self.cfg.poll_interval_sec))
		finally:
			monitor_task.cancel()
			try:
				await monitor_task
			except asyncio.CancelledError:
				pass
			except Exception:
				pass


def _parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Model-driven BTC15m market maker")
	default_model_dir = _SRC_ROOT / "tradebot" / "data" / "btc15m_model_coinbase_80d_purged30"
	p.add_argument(
		"--model-dir",
		default=str(default_model_dir),
		help="Path to trained model dir (meta.json + model.json)",
	)
	p.add_argument("--dry-run", action="store_true", help="Simulate without placing orders")
	p.add_argument("--asset", type=str, default=_cfg_default("asset"), help="Asset (e.g. BTC)")
	p.add_argument(
		"--horizon-minutes",
		type=int,
		default=_cfg_default("horizon_minutes"),
		help="Only consider markets expiring within this horizon",
	)
	p.add_argument(
		"--min-seconds-to-expiry",
		type=int,
		default=_cfg_default("min_seconds_to_expiry"),
		help="Skip markets with less time-to-expiry than this",
	)
	p.add_argument("--poll", type=float, default=_cfg_default("poll_interval_sec"), help="Poll interval seconds")
	p.add_argument("--duration", type=float, default=_cfg_default("duration_sec"), help="Run duration seconds (0=forever)")
	p.add_argument(
		"--limit",
		"--limit-markets",
		dest="limit_markets",
		type=int,
		default=_cfg_default("limit_markets"),
		help="Max markets/tickers per poll",
	)
	p.add_argument(
		"--max-abs-pos",
		type=int,
		default=_cfg_default("max_abs_pos"),
		help="Max abs position per ticker",
	)
	p.add_argument(
		"--quote-size",
		type=int,
		default=_cfg_default("quote_size"),
		help="Order size per quote",
	)
	p.add_argument(
		"--min-spread",
		type=int,
		default=_cfg_default("min_spread_cents"),
		help="Minimum spread in cents",
	)
	p.add_argument(
		"--base-spread",
		type=int,
		default=_cfg_default("base_spread_cents"),
		help="Base spread in cents (before widening)",
	)
	p.add_argument(
		"--max-spread",
		type=int,
		default=_cfg_default("max_spread_cents"),
		help="Max spread in cents",
	)
	p.add_argument(
		"--one-sided-threshold",
		type=float,
		default=_cfg_default("one_sided_threshold"),
		help="If p_yes >= this, only quote BUY YES",
	)
	p.add_argument(
		"--one-sided-threshold-low",
		type=float,
		default=_cfg_default("one_sided_threshold_low"),
		help="If p_yes <= this, only quote SELL YES",
	)
	p.add_argument(
		"--deadzone-neutral-low",
		type=float,
		default=_cfg_default("deadzone_neutral_low"),
		help="If p_yes in [low,high], quote both sides",
	)
	p.add_argument(
		"--deadzone-neutral-high",
		type=float,
		default=_cfg_default("deadzone_neutral_high"),
		help="If p_yes in [low,high], quote both sides",
	)
	p.add_argument(
		"--momentum-pause-abs-return-1m",
		type=float,
		default=_cfg_default("momentum_pause_abs_return_1m"),
		help="If abs(1m return proxy) >= this, pause quoting",
	)
	p.add_argument(
		"--momentum-pause-seconds",
		type=int,
		default=_cfg_default("momentum_pause_seconds"),
		help="How long to pause quoting after momentum trigger",
	)
	p.add_argument(
		"--force-widen-tte-sec",
		type=int,
		default=_cfg_default("force_widen_tte_sec"),
		help="Widen spread when time-to-expiry below this",
	)
	p.add_argument(
		"--force-stop-tte-sec",
		type=int,
		default=_cfg_default("force_stop_tte_sec"),
		help="Stop quoting and cancel when time-to-expiry below this",
	)
	p.add_argument(
		"--skew-cents-per-contract",
		type=float,
		default=_cfg_default("inventory_skew_cents_per_contract"),
		help="Inventory skew (cents per contract) applied to fair price",
	)
	p.add_argument(
		"--inventory-soft-limit",
		type=int,
		default=_cfg_default("inventory_soft_limit"),
		help="Start widening/skewing more when abs(pos) >= this",
	)
	p.add_argument(
		"--inventory-hard-limit",
		type=int,
		default=_cfg_default("inventory_hard_limit"),
		help="Block the side that increases inventory when abs(pos) >= this",
	)
	p.add_argument(
		"--inventory-skew-quadratic",
		action="store_true",
		default=bool(_cfg_default("inventory_skew_quadratic")),
		help="Use quadratic pos^2 scaling for skew",
	)
	p.add_argument(
		"--min-quote-refresh-seconds",
		type=int,
		default=_cfg_default("min_quote_refresh_seconds"),
		help="Only refresh quotes if changed or older than this",
	)
	p.add_argument(
		"--no-cancel-before-replace",
		action="store_true",
		help="If set, do NOT cancel before placing replacements",
	)
	p.add_argument(
		"--monitor-interval-sec",
		type=float,
		default=_cfg_default("monitor_interval_sec"),
		help="How often to log fills/positions rollups",
	)
	p.add_argument(
		"--fills-limit",
		type=int,
		default=_cfg_default("fills_limit"),
		help="How many recent fills to poll each monitor tick",
	)
	return p.parse_args()


async def main() -> None:
	logging.basicConfig(
		level=logging.INFO,
		format="%(asctime)s %(levelname)s %(name)s: %(message)s",
	)

	# Silence noisy per-request HTTP logs (httpx/httpcore) while keeping our strategy logs.
	logging.getLogger("httpx").setLevel(logging.WARNING)
	logging.getLogger("httpcore").setLevel(logging.WARNING)

	args = _parse_args()
	# Give a clear error if the default model dir doesn't exist.
	if not Path(str(args.model_dir)).exists():
		raise SystemExit(
			f"Model dir not found: {args.model_dir}. Pass --model-dir to point at a folder containing meta.json + model.json."
		)
	settings = Settings.load()
	client = KalshiClient.from_settings(settings)

	cfg = MarketMakerConfig(
		model_dir=str(args.model_dir),
		dry_run=bool(args.dry_run),
		asset=str(args.asset),
		horizon_minutes=int(args.horizon_minutes),
		limit_markets=int(args.limit_markets),
		min_seconds_to_expiry=int(args.min_seconds_to_expiry),
		poll_interval_sec=float(args.poll),
		duration_sec=float(args.duration),
		base_spread_cents=int(args.base_spread),
		min_spread_cents=int(args.min_spread),
		max_spread_cents=int(args.max_spread),
		one_sided_threshold=float(args.one_sided_threshold),
		one_sided_threshold_low=float(args.one_sided_threshold_low),
		deadzone_neutral_low=float(args.deadzone_neutral_low),
		deadzone_neutral_high=float(args.deadzone_neutral_high),
		momentum_pause_abs_return_1m=float(args.momentum_pause_abs_return_1m),
		momentum_pause_seconds=int(args.momentum_pause_seconds),
		force_widen_tte_sec=int(args.force_widen_tte_sec),
		force_stop_tte_sec=int(args.force_stop_tte_sec),
		quote_size=int(args.quote_size),
		max_abs_pos=int(args.max_abs_pos),
		inventory_soft_limit=int(args.inventory_soft_limit),
		inventory_hard_limit=int(args.inventory_hard_limit),
		inventory_skew_cents_per_contract=float(args.skew_cents_per_contract),
		inventory_skew_quadratic=bool(args.inventory_skew_quadratic),
		min_quote_refresh_seconds=int(args.min_quote_refresh_seconds),
		cancel_before_replace=not bool(args.no_cancel_before_replace),
		monitor_interval_sec=float(args.monitor_interval_sec),
		fills_limit=int(args.fills_limit),
	)
	mm = KalshiMarketMaker(client=client, cfg=cfg)
	await mm.run()


if __name__ == "__main__":
	asyncio.run(main())

