from fastapi import FastAPI
from app.utils.logging import setup_logging, logging_middleware
from app.api.endpoints import ingest  # Import your router
from app.config import app_settings
from app.api.endpoints import query

setup_logging()
app = FastAPI(title=app_settings.APP_NAME)

app.middleware("http")(logging_middleware)

app.include_router(ingest.router)

app.include_router(query.router)