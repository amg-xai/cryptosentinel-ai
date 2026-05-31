import sys
from loguru import logger


def _inject_trace_context(record):
    """
    Loguru patcher: if an OpenTelemetry span is active, copy its
    trace_id (as a 32-hex string) into the log record's extra, so log
    lines correlate to the exact Jaeger trace. Falls back to '-' when no
    span is active (startup, background tasks, OTel not installed).
    """
    try:
        from opentelemetry import trace as _otel_trace
        span = _otel_trace.get_current_span()
        ctx = span.get_span_context() if span else None
        if ctx is not None and ctx.is_valid:
            record["extra"]["trace_id"] = format(ctx.trace_id, "032x")
            return
    except Exception:
        pass
    record["extra"].setdefault("trace_id", "-")


def setup_logging(log_level: str = "INFO") -> None:
    """Configure structured JSON logging for the entire application."""
    logger.remove()  # Remove default handler

    log_format = (
        "{{"
        '"timestamp": "{time:YYYY-MM-DDTHH:mm:ss.SSSZ}", '
        '"level": "{level}", '
        '"service": "cryptosentinel", '
        '"module": "{name}", '
        '"function": "{function}", '
        '"line": {line}, '
        '"event": "{message}", '
        '"trace_id": "{extra[trace_id]}"'
        "}}"
    )

    logger.configure(extra={"trace_id": "-"}, patcher=_inject_trace_context)
    logger.add(
        sys.stdout,
        format=log_format,
        level=log_level,
        colorize=False,
        serialize=False,
    )


def get_logger(module_name: str):
    """Get a module-scoped logger. Call this at the top of every src file."""
    return logger.bind(module=module_name)


# Initialize with defaults on import
setup_logging()
