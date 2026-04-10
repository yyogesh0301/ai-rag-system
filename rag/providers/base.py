from abc import ABC, abstractmethod


class LLMProvider(ABC):

    embed_model_name: str

    @abstractmethod
    def get_embeddings(self):
        """Return a LangChain embeddings object."""
        pass

    @abstractmethod
    def get_llm(self):
        """Return a LangChain chat LLM object."""
        pass
