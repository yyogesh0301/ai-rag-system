import os
import ollama
from rag.providers.base import LLMProvider


class OllamaProvider(LLMProvider):

    def __init__(self, embed_model: str, generate_model: str):
        self.embed_model = embed_model
        self.embed_model_name = embed_model
        self.generate_model = generate_model
        host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.client = ollama.Client(host=host)

    def embed(self, text: str) -> list[float]:
        response = self.client.embed(model=self.embed_model, input=text)
        return response.embeddings[0]

    def generate(self, prompt: str) -> str:
        response = self.client.generate(model=self.generate_model, prompt=prompt)
        return response.response
