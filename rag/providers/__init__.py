import os
from dotenv import load_dotenv
from rag.providers.base import LLMProvider

load_dotenv()


def get_provider() -> LLMProvider:
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()

    if provider == "ollama":
        from rag.providers.ollama import OllamaProvider
        return OllamaProvider(
            embed_model=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
            generate_model=os.getenv("OLLAMA_GENERATE_MODEL", "gemma3:4b"),
        )

    elif provider == "gemini":
        from rag.providers.gemini import GeminiProvider
        return GeminiProvider(
            api_key=os.getenv("GEMINI_API_KEY"),
            embed_model=os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001"),
            generate_model=os.getenv("GEMINI_GENERATE_MODEL", "gemini-2.0-flash"),
        )

    else:
        raise ValueError(f"Unknown LLM_PROVIDER: '{provider}'. Choose 'ollama' or 'gemini'.")
