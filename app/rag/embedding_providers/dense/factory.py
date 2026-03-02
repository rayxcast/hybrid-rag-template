from .openai_provider import OpenAIProvider

def get_dense_provider(provider:str = "openai"):
    if provider == "openai":
        return OpenAIProvider()
    
    raise ValueError(f"Unsupported dense provider: {provider}")