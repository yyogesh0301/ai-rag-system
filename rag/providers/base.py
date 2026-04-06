from abc import ABC, abstractmethod


class LLMProvider(ABC):

    embed_model_name: str  # set by each implementation

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Generate an embedding vector for the given text."""
        pass

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a text response for the given prompt."""
        pass
