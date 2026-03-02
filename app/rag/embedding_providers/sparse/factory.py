from .splade_provider import SparseEmbeddingProvider

def get_sparse_provider(provider:str = "fastembed"):
    if provider == "fastembed":
        return SparseEmbeddingProvider()
    
    raise ValueError(f"Unsupported sparse provider: {provider}")