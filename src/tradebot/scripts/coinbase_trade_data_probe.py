"""Coinbase trade (tick) data availability probe.

Purpose
- Verify we can pull historical BTC-USD trade data from Coinbase Advanced Trade
  (sub-minute resolution) and reconstruct a valid 60-second average price window.

Requirements (per request)
- Uses Coinbase Advanced Trade REST API endpoint:
    GET /api/v3/brokerage/products/BTC-USD/trades
- Auth via environment variables:
    COINBASE_PRIVATE_KEY_PATH
    COINBASE_KEY_ID
  using request signing (JWT ES256).

Notes
- This is a standalone probe script. It is not part of the trading bot.
- Coinbase market-data endpoints may be public, but this probe always attempts
  authenticated requests per spec.

Run
  python coinbase_trade_data_probe.py --minutes 15
"""

from __future__ import annotations

import argparse
import base64
import csv
import datetime as dt
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import requests

# We prefer using the already-installed cryptography package for ES256 signing.
# This avoids requiring OpenSSL on PATH.
try:  # pragma: no cover
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature

    _HAS_CRYPTOGRAPHY = True
except Exception:  # pragma: no cover
    _HAS_CRYPTOGRAPHY = False


API_HOST = "api.coinbase.com"
BASE_URL = f"https://{API_HOST}"
PRODUCT_ID = "BTC-USD"

# The user-requested endpoint (may not exist in current Coinbase docs/deployments).
TRADES_PATH = f"/api/v3/brokerage/products/{PRODUCT_ID}/trades"

# Documented endpoint that returns recent trades (ticks) plus best bid/ask.
TICKER_PATH = f"/api/v3/brokerage/products/{PRODUCT_ID}/ticker"


class ProbeError(RuntimeError):
    pass


def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64url_decode(s: str) -> bytes:
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode((s + pad).encode("ascii"))


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _format_yyyymmdd_hhmm(ts: dt.datetime) -> str:
    ts = ts.astimezone(dt.timezone.utc)
    return ts.strftime("%Y%m%d_%H%M")


def _read_env_required(name: str) -> str:
    val = (os.getenv(name) or "").strip()
    if not val:
        raise ProbeError(
            f"Missing required environment variable {name}. "
            f"Set it and retry."
        )
    return val


def _sign_es256_jwt_input(*, private_key_path: str, message: bytes) -> bytes:
    """Return raw 64-byte ES256 signature for JWT (r||s)."""

    if _HAS_CRYPTOGRAPHY:
        try:
            key_bytes = Path(private_key_path).read_bytes()
        except Exception as e:
            raise ProbeError(f"Failed to read private key file: {private_key_path}") from e

        # Some users store PEMs with literal "\n" sequences instead of actual newlines.
        # Detect and unescape that so cryptography can parse the PEM.
        if b"\\n" in key_bytes and b"\n" not in key_bytes:
            try:
                key_bytes = key_bytes.decode("utf-8", errors="strict").replace("\\n", "\n").encode("utf-8")
            except Exception:
                # Best-effort; fall through to the normal parser error.
                pass
        key_bytes = key_bytes.strip() + b"\n"

        try:
            key = serialization.load_pem_private_key(key_bytes, password=None)
        except Exception as e:
            raise ProbeError(
                "Failed to parse COINBASE_PRIVATE_KEY_PATH as a PEM private key. "
                "Coinbase Advanced Trade keys are typically EC keys in PEM format."
            ) from e

        if not isinstance(key, ec.EllipticCurvePrivateKey):
            raise ProbeError(
                "Private key is not an EC key. ES256 requires an elliptic curve private key (P-256)."
            )
        if key.curve.name not in {"secp256r1", "prime256v1"}:
            raise ProbeError(
                f"Unexpected EC curve {key.curve.name}. ES256 requires P-256 (secp256r1)."
            )

        sig_der = key.sign(message, ec.ECDSA(hashes.SHA256()))
        r, s = decode_dss_signature(sig_der)
        r_bytes = int(r).to_bytes(32, "big", signed=False)
        s_bytes = int(s).to_bytes(32, "big", signed=False)
        return r_bytes + s_bytes

    # Fallback path (requires OpenSSL on PATH). Keep it for portability.
    try:  # pragma: no cover
        import subprocess

        cmd = ["openssl", "dgst", "-sha256", "-sign", private_key_path]
        proc = subprocess.run(
            cmd,
            input=message,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if proc.returncode != 0:
            stderr = proc.stderr.decode("utf-8", errors="replace").strip()
            raise ProbeError(f"OpenSSL signing failed (rc={proc.returncode}): {stderr}")
        sig_der = proc.stdout
        return _ecdsa_der_to_raw_rs(sig_der, size=32)
    except FileNotFoundError as e:
        raise ProbeError(
            "Neither cryptography nor OpenSSL are available for ES256 signing. "
            "Install `cryptography` (preferred) or OpenSSL."
        ) from e


def _read_asn1_length(buf: bytes, i: int) -> tuple[int, int]:
    """Read DER length at buf[i], return (length, next_index)."""
    if i >= len(buf):
        raise ProbeError("Invalid DER: unexpected end while reading length")
    first = buf[i]
    i += 1
    if first < 0x80:
        return int(first), i
    n = first & 0x7F
    if n == 0 or n > 4 or i + n > len(buf):
        raise ProbeError("Invalid DER: bad long-form length")
    length = int.from_bytes(buf[i : i + n], "big")
    i += n
    return length, i


def _ecdsa_der_to_raw_rs(sig_der: bytes, size: int = 32) -> bytes:
    """Convert DER ECDSA signature to raw r||s bytes (JWT format).

    JWT ES256 expects 64-byte signature: r(32) || s(32).
    """
    if not sig_der:
        raise ProbeError("Invalid DER: empty signature")

    i = 0
    if sig_der[i] != 0x30:
        raise ProbeError("Invalid DER: expected SEQUENCE")
    i += 1
    seq_len, i = _read_asn1_length(sig_der, i)
    end = i + seq_len
    if end > len(sig_der):
        raise ProbeError("Invalid DER: sequence length out of bounds")

    def read_int() -> int:
        nonlocal i
        if i >= end or sig_der[i] != 0x02:
            raise ProbeError("Invalid DER: expected INTEGER")
        i += 1
        n, i2 = _read_asn1_length(sig_der, i)
        i = i2
        if i + n > end:
            raise ProbeError("Invalid DER: integer length out of bounds")
        raw = sig_der[i : i + n]
        i += n
        # Integers can be prefixed with 0x00 to force positive.
        if raw and raw[0] == 0x00:
            raw = raw[1:]
        return int.from_bytes(raw, "big")

    r = read_int()
    s = read_int()

    r_bytes = int(r).to_bytes(size, "big", signed=False)
    s_bytes = int(s).to_bytes(size, "big", signed=False)
    return r_bytes + s_bytes


def _build_jwt(
    *,
    key_id: str,
    private_key_path: str,
    method: str,
    path: str,
    uri_mode: str,
    ttl_seconds: int = 120,
) -> str:
    """Build a Coinbase Advanced Trade JWT (ES256).

    Coinbase documentation has used a JWT with a `uri` claim that binds method+path.
    Because doc formats can vary, we support two internal modes:
      - uri_mode="host_path": "GET api.coinbase.com/api/v3/..."
      - uri_mode="path_only": "GET /api/v3/..."

    The probe will try host_path first, then path_only if auth fails.
    """

    now = int(time.time())
    header = {"alg": "ES256", "kid": key_id, "typ": "JWT"}

    if uri_mode == "host_path":
        uri = f"{method.upper()} {API_HOST}{path}"
    elif uri_mode == "path_only":
        uri = f"{method.upper()} {path}"
    else:
        raise ProbeError(f"Unknown uri_mode={uri_mode}")

    payload = {
        "iss": "cdp",
        "sub": key_id,
        "nbf": now,
        "exp": now + int(ttl_seconds),
        "uri": uri,
    }

    signing_input = (
        f"{_b64url_encode(json.dumps(header, separators=(",", ":")).encode('utf-8'))}."
        f"{_b64url_encode(json.dumps(payload, separators=(",", ":")).encode('utf-8'))}"
    )

    sig_raw = _sign_es256_jwt_input(private_key_path=private_key_path, message=signing_input.encode("utf-8"))
    token = f"{signing_input}.{_b64url_encode(sig_raw)}"
    return token


def _request_with_jwt(
    *,
    session: requests.Session,
    method: str,
    url: str,
    params: dict[str, Any] | None,
    key_id: str,
    private_key_path: str,
    request_path: str,
) -> requests.Response:
    """Make an authenticated request; retries with alternate `uri` format on auth failure."""

    for uri_mode in ("host_path", "path_only"):
        jwt_token = _build_jwt(
            key_id=key_id,
            private_key_path=private_key_path,
            method=method,
            path=request_path,
            uri_mode=uri_mode,
        )
        headers = {
            "Authorization": f"Bearer {jwt_token}",
            "Accept": "application/json",
            "User-Agent": "coinbase-trade-data-probe/1.0",
        }
        resp = session.request(method=method, url=url, params=params, headers=headers, timeout=30)
        if resp.status_code not in (401, 403):
            return resp

        # If auth fails, try the other uri format once.
        # (Do not print sensitive response bodies.)
    return resp


def _parse_trades(payload: dict[str, Any]) -> list[dict[str, Any]]:
    trades = None
    for k in ("trades", "data", "results"):
        if isinstance(payload.get(k), list):
            trades = payload.get(k)
            break
    if trades is None:
        # Some responses nest under {"trades": {"trades": [...]}}
        t = payload.get("trades")
        if isinstance(t, dict) and isinstance(t.get("trades"), list):
            trades = t.get("trades")

    if not isinstance(trades, list):
        raise ProbeError("Unexpected response shape: could not find trades list")

    out: list[dict[str, Any]] = []
    for t in trades:
        if not isinstance(t, dict):
            continue

        trade_id = t.get("trade_id") or t.get("id") or t.get("tradeId")
        price = t.get("price") or t.get("price_in_quote")
        size = t.get("size") or t.get("size_in_quote") or t.get("qty")
        ts = t.get("time") or t.get("timestamp") or t.get("trade_time")

        if trade_id is None or price is None or size is None or ts is None:
            continue

        out.append({"trade_id": str(trade_id), "price": price, "size": size, "timestamp": str(ts)})

    if not out:
        raise ProbeError("No trades parsed from response")

    return out


def _to_utc_datetime(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, utc=True, errors="coerce")
    if parsed.isna().any():
        bad = int(parsed.isna().sum())
        raise ProbeError(f"Failed to parse {bad} trade timestamps")
    return parsed


def _warn_gaps(trade_times_utc: pd.Series) -> None:
    # trade_times_utc expected sorted DESC (newest -> oldest) during pagination.
    # Gap detection based on absolute diffs between consecutive trades.
    diffs = (trade_times_utc.iloc[:-1].to_numpy() - trade_times_utc.iloc[1:].to_numpy())
    # diffs are numpy timedelta64[ns]
    diffs_s = pd.to_timedelta(diffs).total_seconds()
    max_gap = float(diffs_s.max()) if len(diffs_s) else 0.0
    if max_gap > 2.0:
        print(f"WARNING: detected trade gap max={max_gap:.2f}s (> 2s)")


@dataclass(frozen=True)
class FetchResult:
    trades: list[dict[str, Any]]


def fetch_trades_last_n_minutes(
    *,
    minutes: int,
    max_pages: int = 500,
    window_seconds: int = 300,
    progress_every: int = 25,
) -> FetchResult:
    """Fetch BTC-USD trades going backwards via `before` pagination."""

    key_id = _read_env_required("COINBASE_KEY_ID")
    private_key_path = _read_env_required("COINBASE_PRIVATE_KEY_PATH")

    if not Path(private_key_path).exists():
        raise ProbeError(f"COINBASE_PRIVATE_KEY_PATH does not exist: {private_key_path}")

    end_time = _utcnow()
    cutoff = end_time - dt.timedelta(minutes=int(minutes))

    session = requests.Session()

    all_trades: list[dict[str, Any]] = []
    seen_trade_ids: set[str] = set()

    # Mode selection: try the requested /trades endpoint, but fall back to /ticker.
    mode: str = "trades_before"  # or: "ticker_time"
    before: str | None = None
    cursor_end: dt.datetime | None = None

    last_seen_ts: dt.datetime | None = None
    last_page_oldest_ts: dt.datetime | None = None

    # Conservative pagination loop.
    backoff = 0.5
    pages = 0
    while True:
        params: dict[str, Any] = {"limit": 1000}
        request_path = TRADES_PATH if mode == "trades_before" else TICKER_PATH

        if mode == "trades_before":
            if before:
                params["before"] = before
        else:
            # Time-based pagination using `end` and `start` UNIX seconds.
            # Use small windows so we can keep walking back without missing data.
            if cursor_end is None:
                cursor_end = end_time
            window_seconds_i = int(window_seconds)
            # Keep windows within a reasonable range so the endpoint responds quickly.
            window_seconds_i = max(60, min(3600, window_seconds_i))
            end_s = int(cursor_end.timestamp())
            start_s = int((cursor_end - dt.timedelta(seconds=window_seconds_i)).timestamp())
            params["end"] = str(end_s)
            params["start"] = str(start_s)

        url = f"{BASE_URL}{request_path}"
        resp = _request_with_jwt(
            session=session,
            method="GET",
            url=url,
            params=params,
            key_id=key_id,
            private_key_path=private_key_path,
            request_path=request_path,
        )

        if resp.status_code == 429:
            # Rate limit: back off and retry.
            retry_after = resp.headers.get("Retry-After")
            if retry_after:
                try:
                    backoff = max(backoff, float(retry_after))
                except Exception:
                    pass
            print(f"RATE_LIMIT: HTTP 429, sleeping {backoff:.2f}s", file=sys.stderr)
            time.sleep(backoff)
            backoff = min(10.0, backoff * 1.6)
            continue

        if resp.status_code in (401, 403):
            raise ProbeError(
                "Authentication failed (HTTP 401/403). Check COINBASE_KEY_ID and "
                "COINBASE_PRIVATE_KEY_PATH (Advanced Trade API key). Also ensure your JWT "
                "signing works (OpenSSL) and key permissions include market data."
            )

        # If the requested endpoint is not found, transparently fall back to /ticker.
        if resp.status_code == 404 and mode == "trades_before":
            mode = "ticker_time"
            before = None
            cursor_end = None
            continue

        if resp.status_code >= 500:
            raise ProbeError(f"Coinbase server error HTTP {resp.status_code}")

        if resp.status_code != 200:
            # Print a small, non-sensitive snippet.
            snippet = resp.text[:400].replace("\n", " ")
            raise ProbeError(f"HTTP {resp.status_code} from Coinbase: {snippet}")

        payload = resp.json()
        page_trades = _parse_trades(payload)

        # Coinbase should return newest->oldest. Assert strict decrease while paginating.
        # We parse timestamps as UTC for comparisons.
        page_df = pd.DataFrame(page_trades)
        page_df["ts"] = _to_utc_datetime(page_df["timestamp"])

        # Sanity checks:
        # - Within a page, timestamps should be non-increasing (newest -> oldest).
        # - Across pages, the *oldest* timestamp should strictly decrease, otherwise pagination isn't moving back.
        ts_list = page_df["ts"].tolist()
        for i in range(1, len(ts_list)):
            if ts_list[i] > ts_list[i - 1]:
                raise ProbeError(
                    "Timestamp ordering violation within page: timestamps increased (expected newest->oldest)."
                )

        page_oldest_ts = page_df["ts"].iloc[-1].to_pydatetime().replace(tzinfo=dt.timezone.utc)
        if last_page_oldest_ts is not None and not (page_oldest_ts < last_page_oldest_ts):
            raise ProbeError(
                "Pagination ordering violation: page-oldest timestamp did not strictly decrease across pages."
            )
        last_page_oldest_ts = page_oldest_ts

        # Track last seen timestamp (best-effort) for optional downstream diagnostics.
        if ts_list:
            last_seen_ts = ts_list[-1]

        # Append as-is (newest->oldest), deduping by trade_id.
        for t in page_trades:
            tid = str(t.get("trade_id"))
            if tid in seen_trade_ids:
                continue
            seen_trade_ids.add(tid)
            all_trades.append(t)
        pages += 1

        if progress_every and pages % int(progress_every) == 0:
            # Best-effort progress snapshot.
            try:
                oldest_p = page_df["ts"].iloc[-1].to_pydatetime().replace(tzinfo=dt.timezone.utc)
                newest_p = page_df["ts"].iloc[0].to_pydatetime().replace(tzinfo=dt.timezone.utc)
                print(
                    f"progress: mode={mode} pages={pages} trades={len(all_trades)} "
                    f"page_newest={newest_p.isoformat()} page_oldest={oldest_p.isoformat()}",
                    flush=True,
                )
            except Exception:
                pass

        # Update pagination cursor.
        if mode == "trades_before":
            oldest_trade_id = str(page_trades[-1]["trade_id"])
            before = oldest_trade_id
        else:
            # Move end cursor backward just before the oldest trade in this batch.
            oldest_ts_dt = page_df["ts"].iloc[-1].to_pydatetime().replace(tzinfo=dt.timezone.utc)
            cursor_end = oldest_ts_dt - dt.timedelta(seconds=1)

        # Stop when we have enough history.
        oldest_ts = page_df["ts"].iloc[-1].to_pydatetime().replace(tzinfo=dt.timezone.utc)
        newest_ts = page_df["ts"].iloc[0].to_pydatetime().replace(tzinfo=dt.timezone.utc)

        if oldest_ts <= cutoff and len(all_trades) > 0:
            break

        # Guard against pathological cases where the API is not moving backwards.
        if mode == "ticker_time" and cursor_end is not None and cursor_end >= newest_ts:
            raise ProbeError("Pagination stalled: end cursor did not move backwards")

        # Also stop if response seems to repeat.
        if pages >= int(max_pages):
            print(
                f"WARNING: reached max_pages={int(max_pages)} before covering requested minutes={minutes}. "
                "Proceeding with whatever history was fetched.",
                file=sys.stderr,
                flush=True,
            )
            break

        # Be polite.
        time.sleep(0.15)

    return FetchResult(trades=all_trades)


def save_trades_csv(*, trades: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["trade_id", "price", "size", "timestamp"])
        w.writeheader()
        for t in trades:
            w.writerow(
                {
                    "trade_id": t.get("trade_id"),
                    "price": t.get("price"),
                    "size": t.get("size"),
                    "timestamp": t.get("timestamp"),
                }
            )


def aggregation_check(*, trades: list[dict[str, Any]]) -> None:
    df = pd.DataFrame(trades)
    df["timestamp"] = _to_utc_datetime(df["timestamp"])

    # Coerce numeric
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["size"] = pd.to_numeric(df["size"], errors="coerce")

    df = df.dropna(subset=["timestamp", "price", "size"]).copy()
    if df.empty:
        raise ProbeError("No valid trades after parsing/coercion")

    # Data comes in newest->oldest; keep that for gap check + summary.
    df = df.sort_values("timestamp", ascending=False).reset_index(drop=True)

    total_trades = int(len(df))
    newest = df["timestamp"].iloc[0]
    oldest = df["timestamp"].iloc[-1]

    # Sanity: ensure we have >= 60 seconds of coverage.
    span_s = float((newest - oldest).total_seconds())
    if span_s < 60.0:
        raise ProbeError(f"Need at least 60 seconds of data; only have span={span_s:.1f}s")

    _warn_gaps(df["timestamp"])

    # Resample into 1-second buckets (ascending for resample).
    asc = df.sort_values("timestamp", ascending=True).set_index("timestamp")

    per_sec_mean = asc["price"].resample("1s").mean()
    roll_60 = per_sec_mean.rolling(window=60, min_periods=60).mean()

    last5 = roll_60.dropna().tail(5)
    if last5.empty:
        raise ProbeError("Rolling 60-second mean produced no values (insufficient contiguous seconds)")

    approx_tps = total_trades / max(1.0, span_s)

    print("\n=== Coinbase Trade Data Probe Summary ===")
    print(f"trades_fetched: {total_trades}")
    print(f"earliest_timestamp_utc: {oldest.isoformat()}")
    print(f"latest_timestamp_utc:   {newest.isoformat()}")
    print(f"span_seconds:           {span_s:.1f}")
    print(f"approx_trades_per_sec:  {approx_tps:.2f}")
    print("last_5_rolling_60s_mean_prices:")
    for ts, v in last5.items():
        print(f"  {ts.isoformat()}  {float(v):.2f}")

    example = float(last5.iloc[-1])
    print(f"example_60s_avg_price:  {example:.2f}")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Probe Coinbase Advanced Trade BTC-USD trades endpoint and compute a rolling 60s mean price."
        )
    )
    parser.add_argument("--minutes", type=int, default=10, help="How many minutes to fetch (default: 10)")
    parser.add_argument(
        "--max-pages",
        type=int,
        default=500,
        help="Safety cap on pagination requests (default: 500)",
    )
    parser.add_argument(
        "--window-seconds",
        type=int,
        default=300,
        help="Time window per request when using ticker pagination (default: 300)",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print progress every N pages (0 disables) (default: 25)",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)

    minutes = int(args.minutes)
    if minutes <= 0:
        raise SystemExit("--minutes must be > 0")

    try:
        result = fetch_trades_last_n_minutes(
            minutes=minutes,
            max_pages=int(args.max_pages),
            window_seconds=int(args.window_seconds),
            progress_every=int(args.progress_every),
        )

        project_root = Path(__file__).resolve().parents[3]  # .../tradebot
        out_dir = project_root / "data"
        out_path = out_dir / f"coinbase_trades_BTCUSD_{_format_yyyymmdd_hhmm(_utcnow())}.csv"

        save_trades_csv(trades=result.trades, out_path=out_path)
        print(f"saved_csv: {out_path}")

        aggregation_check(trades=result.trades)
        return 0

    except ProbeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Stopped", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
