import time
from contextlib import contextmanager


@contextmanager
def stage_timer(stage_name: str, logger, trace_id: str, metrics: dict | None = None):
    start = time.perf_counter()
    yield
    duration = time.perf_counter() - start

    duration = round(duration, 4)

    logger.info(
        "stage_latency",
        trace_id=trace_id,
        stage=stage_name,
        duration_seconds=duration,
    )

    if metrics is not None:
        metrics[stage_name] = duration