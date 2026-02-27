from contextlib import asynccontextmanager
from fastapi import FastAPI, status
from fastapi.responses import JSONResponse
from app.utils.logging import setup_logging, logging_middleware
from app.api.endpoints import ingest, query
from app.config import app_settings
from app.utils.cache import init_cache_index
import redis.asyncio as redis
from app.config import app_settings   # already imported, but for redis_client if needed

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

app.include_router(ingest.router)
app.include_router(query.router)