from __future__ import annotations

import csv
import datetime as dt
import json
import logging
import os
import threading
from pathlib import Path
from typing import Optional


class CsvLogHandler(logging.Handler):
    """Write Python logging records to a CSV file.

    Captures the same log events that appear in the terminal (via logging), but
    in a structured format that is easy to analyze later.

    Columns:
      - ts_utc_iso
      - ts_epoch
      - level
      - logger
      - event
      - msg_template
      - message
      - arg0..arg19
      - fields_json
      - exception
    """

    _MAX_ARGS = 20

    # Common structured fields we want as first-class columns when present.
    # Log sites can attach these via `extra={"csv_fields": {"ticker": "...", ...}}`.
    _FIELD_COLS: tuple[str, ...] = (
        "path",
        "ticker",
        "side",
        "action",
        "decision",
        "kelly_fraction",
        "qty",
        "price_cents",
        "price_dollars",
        "spot_usd",
        "strike_usd",
        "strike_src",
        "tte_s",
        "mode",
        "time_in_force",
        "fee_assumption",
        "fee_per_contract",
        "bankroll_usd",
        "portfolio_value_usd",
        "ev_after_fees",
        "p_yes",
        "p_no",
        "market_p_yes",
        "market_p_no",
        "reason",
        "order_id",
        "diff_pct",
        "max_pct",
        "error",
        "error_type",
    )

    def __init__(self, path: str) -> None:
        super().__init__()
        self._path = str(path)
        self._lock = threading.Lock()

        p = Path(self._path)
        p.parent.mkdir(parents=True, exist_ok=True)

        file_existed = p.exists()
        file_empty = (not file_existed) or (p.stat().st_size == 0)

        # If the file already exists and has a header, preserve it for backward
        # compatibility (do not silently change column order / count mid-file).
        self._header: list[str]
        if not file_empty:
            try:
                with open(p, "r", encoding="utf-8", newline="") as rfp:
                    reader = csv.reader(rfp)
                    first = next(reader, None)
                self._header = [str(x) for x in (first or [])]
            except Exception:
                self._header = []
        else:
            self._header = []

        if not self._header:
            self._header = [
                "ts_utc_iso",
                "ts_epoch",
                "level",
                "logger",
                "message",
                "exception",
            ]

        # newline='' is important on Windows for correct CSV rows.
        self._fp = open(p, "a", encoding="utf-8", newline="")
        self._writer = csv.writer(self._fp)

        if file_empty:
            self._writer.writerow([
                "ts_utc_iso",
                "ts_epoch",
                "level",
                "logger",
                "message",
                "exception",
            ])
            self._fp.flush()

    @staticmethod
    def _event_from_msg_template(msg_template: str) -> str:
        # Most of our logs begin with an uppercase event token (e.g., ORDER, ENTRY_BLOCK).
        first = (msg_template or "").strip().split(" ", 1)[0]
        if not first:
            return ""
        if all(c.isupper() or c.isdigit() or c == "_" for c in first):
            return first
        return ""

    def _normalize_args(self, args_obj: object) -> list[str]:
        if not args_obj:
            return []
        if isinstance(args_obj, dict):
            # Preserve mapping-shaped args as JSON instead of positional columns.
            try:
                return [json.dumps(args_obj, sort_keys=True, ensure_ascii=False)]
            except Exception:
                return [str(args_obj)]
        if isinstance(args_obj, (list, tuple)):
            out: list[str] = []
            for v in list(args_obj)[: self._MAX_ARGS]:
                try:
                    out.append(str(v))
                except Exception:
                    out.append("<unprintable>")
            return out
        return [str(args_obj)]

    @staticmethod
    def _extract_fields_from_record(record: logging.LogRecord) -> dict[str, object]:
        # Allow callers to attach structured fields via `extra={"csv_fields": {...}}`.
        v = getattr(record, "csv_fields", None)
        if isinstance(v, dict):
            return dict(v)
        return {}

    def emit(self, record: logging.LogRecord) -> None:
        try:
            created = float(getattr(record, "created", 0.0) or 0.0)
            ts_iso = dt.datetime.fromtimestamp(created, tz=dt.timezone.utc).isoformat()

            msg = record.getMessage()


            exc_text: str = ""
            if record.exc_info:
                try:
                    exc_text = self.formatException(record.exc_info)
                except Exception:
                    exc_text = "<exception_format_error>"


            row = [
                ts_iso,
                f"{created:.6f}",
                str(record.levelname),
                str(record.name),
                str(msg),
                str(exc_text),
            ]

            with self._lock:
                self._writer.writerow(row)
                self._fp.flush()
        except Exception:
            # Never let logging failures crash the bot.
            self.handleError(record)

    def close(self) -> None:
        try:
            with self._lock:
                try:
                    self._fp.flush()
                finally:
                    self._fp.close()
        finally:
            super().close()


def default_live_csv_log_path(*, prefix: str = "bot_logs") -> str:
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    return str(Path("runs") / f"{prefix}_{ts}.csv")


def install_csv_log_handler(
    *,
    path: str,
    level: int = logging.INFO,
    root_logger: Optional[logging.Logger] = None,
) -> CsvLogHandler:
    """Attach a CsvLogHandler to the root logger (or provided logger)."""

    logger = root_logger if root_logger is not None else logging.getLogger()
    handler = CsvLogHandler(str(path))
    handler.setLevel(int(level))
    logger.addHandler(handler)
    return handler
