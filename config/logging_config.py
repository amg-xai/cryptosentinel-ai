import sys

from loguru import logger


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

    logger.configure(extra={"trace_id": "-"})

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
