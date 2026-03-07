from app.config import app_settings
from app.rag.reranker_providers.remote_reranker import RemoteReranker
from .fastembed_reranker import FastEmbedReranker

def get_reranker():
    if app_settings.RERANKER_PROVIDER == "fastembed":
        return FastEmbedReranker()
    elif app_settings.RERANKER_PROVIDER == "remote":
        return RemoteReranker()
    else:
        return None