from google import genai
from rag.providers.base import LLMProvider


class GeminiProvider(LLMProvider):

    def __init__(self, api_key: str, embed_model: str, generate_model: str):
        self.client = genai.Client(api_key=api_key)
        self.embed_model = embed_model
        self.embed_model_name = embed_model
        self.generate_model = generate_model

    def embed(self, text: str) -> list[float]:
        response = self.client.models.embed_content(
            model=self.embed_model,
            contents=[text]
        )
        return response.embeddings[0].values

    def generate(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self.generate_model,
            contents=prompt
        )
        return response.text
