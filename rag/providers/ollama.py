import os
from langchain_ollama import OllamaEmbeddings, ChatOllama
from rag.providers.base import LLMProvider


class OllamaProvider(LLMProvider):

    def __init__(self, embed_model: str, generate_model: str):
        self.embed_model_name = embed_model
        self.generate_model = generate_model
        self.host = os.getenv("OLLAMA_HOST", "http://localhost:11434")

    def get_embeddings(self):
        return OllamaEmbeddings(model=self.embed_model_name, base_url=self.host)

    def get_llm(self):
        return ChatOllama(model=self.generate_model, base_url=self.host)
