from contextlib import asynccontextmanager
from pathlib import Path

import redis.asyncio as redis
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.endpoints import ingest, query, status as status_endpoint
from app.config import app_settings
from app.utils.cache import init_cache_index
from app.utils.logging import logging_middleware, setup_logging

setup_logging()

redis_client = redis.from_url(app_settings.REDIS_URL, decode_responses=True)  # if you need it here

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_cache_index()
    yield
    # Optional: await redis_client.aclose()

app = FastAPI(
    title=app_settings.APP_NAME,
    lifespan=lifespan,
    docs_url="/docs",           # keep Swagger
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

app.middleware("http")(logging_middleware)

static_dir = Path(__file__).resolve().parent / "static"

app.mount("/static", StaticFiles(directory=static_dir), name="static")

app.include_router(ingest.router)
app.include_router(query.router)
app.include_router(status_endpoint.router)


@app.get("/", include_in_schema=False)
async def demo_ui():
    return FileResponse(static_dir / "index.html")
