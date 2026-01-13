from __future__ import annotations

import csv
import datetime as dt
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
      - message
      - exception
    """

    def __init__(self, path: str) -> None:
        super().__init__()
        self._path = str(path)
        self._lock = threading.Lock()

        p = Path(self._path)
        p.parent.mkdir(parents=True, exist_ok=True)

        file_existed = p.exists()
        file_empty = (not file_existed) or (p.stat().st_size == 0)

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
