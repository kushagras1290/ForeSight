"""Structured logging for Project FORESIGHT.

Two formats, selected by ``FORESIGHT_LOG_FORMAT``:

* ``json``    - one JSON object per line, for anywhere logs are collected.
* ``console`` - aligned, human-readable, for local development.

Both carry the same fields, so switching format never loses information. Extra
context is attached per call and appears as top-level JSON keys::

    log.info("pipeline finished", extra={"context": {"rows": 152_000}})

There is no ``print`` anywhere in ``src/``, ``service/``, or ``scripts/``.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import sys
from typing import Any

from foresight.config import get_settings

__all__ = ["configure_logging", "get_logger"]

_CONFIGURED = False

# Attributes present on every LogRecord. Anything outside this set was added by
# the caller and is worth emitting.
_RESERVED_ATTRS = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "message",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "taskName",
        "thread",
        "threadName",
    }
)

# Never let these reach a log sink, whatever a caller passes in `context`.
_SENSITIVE_KEYS = frozenset(
    {"password", "passwd", "secret", "token", "api_key", "apikey", "authorization", "cookie"}
)
_REDACTED = "***redacted***"


def _scrub(value: Any) -> Any:
    """Recursively redact values whose key looks like a credential."""
    if isinstance(value, dict):
        return {
            key: (_REDACTED if str(key).lower() in _SENSITIVE_KEYS else _scrub(inner))
            for key, inner in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_scrub(item) for item in value]
    return value


def _extract_context(record: logging.LogRecord) -> dict[str, Any]:
    """Pull caller-supplied fields off a record, scrubbed."""
    context: dict[str, Any] = {}
    explicit = getattr(record, "context", None)
    if isinstance(explicit, dict):
        context.update(explicit)
    for key, value in record.__dict__.items():
        if key not in _RESERVED_ATTRS and key != "context":
            context[key] = value
    return _scrub(context)


class JsonFormatter(logging.Formatter):
    """Render a record as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": dt.datetime.fromtimestamp(record.created, tz=dt.UTC).isoformat(
                timespec="milliseconds"
            ),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        context = _extract_context(record)
        if context:
            payload.update(context)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        # default=str keeps non-serialisable values (Path, Timestamp, numpy
        # scalars) from turning a log call into a crash.
        return json.dumps(payload, default=str, ensure_ascii=False)


class ConsoleFormatter(logging.Formatter):
    """Aligned, readable output for a terminal."""

    _FMT = "%(asctime)s  %(levelname)-8s  %(name)-28s  %(message)s"
    _DATEFMT = "%H:%M:%S"

    def __init__(self) -> None:
        super().__init__(fmt=self._FMT, datefmt=self._DATEFMT)

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        context = _extract_context(record)
        if context:
            rendered = " ".join(f"{key}={value}" for key, value in sorted(context.items()))
            base = f"{base}  |  {rendered}"
        return base


def configure_logging(*, force: bool = False) -> None:
    """Install the root handler. Idempotent unless ``force`` is set.

    Logs go to stderr so stdout stays clean for piped data.
    """
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    settings = get_settings()
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(JsonFormatter() if settings.log_format == "json" else ConsoleFormatter())

    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(settings.log_level)

    # These libraries are chatty at INFO and drown out the pipeline's own output.
    for noisy in ("matplotlib", "matplotlib.font_manager", "PIL", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger. Safe to call at module import."""
    configure_logging()
    return logging.getLogger(name)
