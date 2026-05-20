import structlog
import logging
import time
import uuid
from fastapi import Request, Response

from app.config import app_settings


SENSITIVE_KEYS = ("api_key", "authorization", "token", "secret", "password")


def _redact(value):
    if isinstance(value, dict):
        redacted = {}
        for key, item in value.items():
            if any(secret_key in str(key).lower() for secret_key in SENSITIVE_KEYS):
                redacted[key] = "[redacted]"
            else:
                redacted[key] = _redact(item)
        return redacted
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


def redact_sensitive_data(_, __, event_dict):
    return _redact(event_dict)


def setup_logging():
    log_level = getattr(logging, app_settings.LOG_LEVEL.upper(), logging.INFO)
    logging.basicConfig(level=log_level, format="%(message)s")

    renderer = (
        structlog.processors.JSONRenderer()
        if app_settings.LOG_FORMAT == "json"
        else structlog.dev.ConsoleRenderer(colors=False)
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            redact_sensitive_data,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        cache_logger_on_first_use=True,
    )

async def logging_middleware(request: Request, call_next):
    logger = structlog.get_logger()
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = request_id
    start = time.perf_counter()

    with structlog.contextvars.bound_contextvars(
        request_id=request_id,
        method=request.method,
        path=request.url.path,
    ):
        logger.info("request_started")
        try:
            response: Response = await call_next(request)
        except Exception:
            duration = round(time.perf_counter() - start, 4)
            logger.exception("request_failed", duration_seconds=duration)
            raise

        duration = round(time.perf_counter() - start, 4)
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "request_finished",
            status_code=response.status_code,
            duration_seconds=duration,
        )
        return response
