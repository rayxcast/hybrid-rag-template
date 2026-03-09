from pydantic_settings import BaseSettings

class AppSettings(BaseSettings):
    
    # Configurable batching parameters (tune for production)
    QUEUE_SIZE: int = 2048
    MAX_BATCH_REQUESTS: int = 32
    INTERNAL_BATCH_SIZE: int = 64
    MAX_BATCH_PAIRS: int = 512
    BATCH_TIMEOUT: float = 0.150 # 50-200ms local dev | production GPU 5–20 ms | CPU production 20–50 ms

    # 1 worker per GPU | 2–4 workers per CPU
    WORKERS: int = 2

reranker_app_settings = AppSettings()