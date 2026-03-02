from app.config import app_settings
from .fastembed_reranker import FastEmbedReranker

def get_reranker():
    if app_settings.RERANKER_PROVIDER == "fastembed":
        return FastEmbedReranker()
    else:
        return None