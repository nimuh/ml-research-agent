"""Structured JSONL logging with run/agent/phase correlation ids."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import ObservabilityConfig
from ..types import utcnow
from ..utils.io import append_jsonl, ensure_dir

LEVELS: dict[str, int] = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40}

_REDACTED = "***redacted***"


@dataclass(frozen=True)
class _Sink:
    """Where log records go. Replaced wholesale by :func:`configure_logging`."""

    level: int = LEVELS["INFO"]
    path: Path | None = None
    console: bool = False
    redact_keys: tuple[str, ...] = ("api_key", "authorization", "token", "secret", "password")


_sink = _Sink()
_lock = threading.Lock()


def configure_logging(config: ObservabilityConfig, *, log_dir: Path | None = None) -> None:
    """Install the process-wide log sink: JSONL file plus an optional console.

    Called once from the CLI. Everything else takes a logger and never touches
    this state, so a library user can silence us by simply not calling it.
    """
    global _sink
    directory = Path(log_dir) if log_dir is not None else Path("workspace") / "logs"
    ensure_dir(directory)
    _sink = _Sink(
        level=LEVELS.get(config.log_level, LEVELS["INFO"]),
        path=directory / "mra.jsonl",
        console=config.console,
        redact_keys=tuple(k.lower() for k in config.redact_keys),
    )


def active_redact_keys() -> tuple[str, ...]:
    """The keys the configured sink is currently masking.

    Exposed so other egress points (trace export, decision packets) mask the
    same set the logs do -- one configured list, not a second copy to drift.
    """
    return _sink.redact_keys


def redact(value: Any, redact_keys: tuple[str, ...] | list[str]) -> Any:
    """Recursively mask values whose key looks secret.

    Substring matching (not equality) so ``anthropic_api_key`` and
    ``x-api-key`` are both caught -- over-redacting a log line is free,
    leaking a key is not.
    """
    keys = tuple(k.lower() for k in redact_keys)
    return _redact(value, keys)


def _redact(value: Any, keys: tuple[str, ...]) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in value.items():
            name = str(k).lower()
            out[str(k)] = _REDACTED if any(key in name for key in keys) else _redact(v, keys)
        return out
    if isinstance(value, list | tuple):
        return [_redact(v, keys) for v in value]
    return value


@dataclass(frozen=True)
class StructuredLogger:
    """An immutable logger carrying correlation context.

    ``bind`` returns a *new* logger so a phase or agent can add its ids without
    mutating the logger its caller still holds.
    """

    name: str
    context: dict[str, Any] = field(default_factory=dict)

    def bind(self, **context: Any) -> StructuredLogger:
        merged = {**self.context, **{k: v for k, v in context.items() if v is not None}}
        return StructuredLogger(self.name, merged)

    def debug(self, event: str, **fields: Any) -> None:
        self.log("DEBUG", event, **fields)

    def info(self, event: str, **fields: Any) -> None:
        self.log("INFO", event, **fields)

    def warning(self, event: str, **fields: Any) -> None:
        self.log("WARNING", event, **fields)

    def error(self, event: str, **fields: Any) -> None:
        self.log("ERROR", event, **fields)

    def log(self, level: str, event: str, **fields: Any) -> None:
        sink = _sink
        severity = LEVELS.get(level, LEVELS["INFO"])
        if severity < sink.level:
            return
        record: dict[str, Any] = {
            "ts": utcnow().isoformat(),
            "level": level,
            "logger": self.name,
            "event": event,
            **self.context,
            **fields,
        }
        safe = _redact(record, sink.redact_keys)
        if sink.path is not None:
            with _lock:
                append_jsonl(sink.path, safe)
        if sink.console:
            _console_write(level, event, safe)


def get_logger(name: str, **context: Any) -> StructuredLogger:
    """The single entrypoint for obtaining a logger."""
    return StructuredLogger(name, {k: v for k, v in context.items() if v is not None})


_STYLES = {"DEBUG": "dim", "INFO": "cyan", "WARNING": "yellow", "ERROR": "bold red"}
_SKIP = frozenset({"ts", "level", "logger", "event"})


def _console_write(level: str, event: str, record: dict[str, Any]) -> None:
    detail = " ".join(f"{k}={v}" for k, v in record.items() if k not in _SKIP)
    try:
        from rich.console import Console

        console = Console(stderr=True)
        # markup=False: log fields are untrusted text and must not be parsed as rich markup.
        console.print(
            f"{level:<7} {event} {detail}".rstrip(),
            style=_STYLES.get(level, ""),
            markup=False,
            highlight=False,
        )
    except Exception:  # pragma: no cover - console output must never break a run
        import sys

        print(f"{level:<7} {event} {detail}", file=sys.stderr)


__all__ = [
    "LEVELS",
    "StructuredLogger",
    "active_redact_keys",
    "configure_logging",
    "get_logger",
    "redact",
]
