from __future__ import annotations

import datetime as dt
from typing import Any


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def to_utc_datetime(value: Any) -> dt.datetime | None:
    """Best-effort parse into a timezone-aware UTC datetime.

    Accepts:
    - pandas Timestamp
    - datetime (naive assumed UTC)
    - unix seconds or milliseconds
    - ISO 8601 strings (with or without trailing 'Z')
    """

    if value is None:
        return None

    # Avoid importing pandas here; keep it pure.
    if hasattr(value, "to_pydatetime"):
        try:
            value = value.to_pydatetime()
        except Exception:
            pass

    if isinstance(value, dt.datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=dt.timezone.utc)
        return value.astimezone(dt.timezone.utc)

    if isinstance(value, (int, float)):
        v = float(value)
        if v <= 0:
            return None
        # heuristic ms
        if v > 1e12:
            v /= 1000.0
        try:
            return dt.datetime.fromtimestamp(v, tz=dt.timezone.utc)
        except Exception:
            return None

    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        # numeric string
        try:
            v = float(s)
            if v > 0:
                if v > 1e12:
                    v /= 1000.0
                return dt.datetime.fromtimestamp(v, tz=dt.timezone.utc)
        except Exception:
            pass

        # ISO string
        try:
            if s.endswith("Z"):
                s = s[:-1] + "+00:00"
            d = dt.datetime.fromisoformat(s)
            if d.tzinfo is None:
                d = d.replace(tzinfo=dt.timezone.utc)
            return d.astimezone(dt.timezone.utc)
        except Exception:
            return None

    return None


def date_range_to_utc_bounds(
    start_date: dt.date | None,
    end_date: dt.date | None,
) -> tuple[dt.datetime | None, dt.datetime | None]:
    """Convert date inputs to inclusive UTC datetime bounds.

    - start_date bound is at 00:00:00 UTC.
    - end_date bound is at 23:59:59.999999 UTC.
    """

    start_dt = None
    end_dt = None

    if start_date is not None:
        start_dt = dt.datetime.combine(start_date, dt.time.min, tzinfo=dt.timezone.utc)
    if end_date is not None:
        end_dt = dt.datetime.combine(end_date, dt.time.max, tzinfo=dt.timezone.utc)

    return start_dt, end_dt
