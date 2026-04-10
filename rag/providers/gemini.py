from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from rag.providers.base import LLMProvider


class GeminiProvider(LLMProvider):

    def __init__(self, api_key: str, embed_model: str, generate_model: str):
        self.api_key = api_key
        self.embed_model_name = embed_model
        self.generate_model = generate_model

    def get_embeddings(self):
        return GoogleGenerativeAIEmbeddings(
            model=self.embed_model_name,
            google_api_key=self.api_key
        )

    def get_llm(self):
        return ChatGoogleGenerativeAI(
            model=self.generate_model,
            google_api_key=self.api_key
        )
