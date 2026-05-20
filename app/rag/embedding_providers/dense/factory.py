from .openai_provider import OpenAIProvider

def get_dense_provider(provider:str = "openai"):
    if provider == "openai":
        return OpenAIProvider()
    if provider == "google":
        from .google_provider import GoogleProvider

        return GoogleProvider()
    
    raise ValueError(f"Unsupported dense provider: {provider}")
