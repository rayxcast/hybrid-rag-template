import structlog
import logging
from fastapi import Request, Response

def setup_logging():
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        cache_logger_on_first_use=True,
    )

async def logging_middleware(request: Request, call_next):
    logger = structlog.get_logger()
    request_id = request.headers.get("X-Request-ID", "no-id")
    with structlog.contextvars.bound_contextvars(request_id=request_id):
        logger.info("request_started", method=request.method, path=request.url.path)
        response: Response = await call_next(request)
        logger.info("request_finished", status_code=response.status_code)
        return response