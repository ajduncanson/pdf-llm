import os
from abc import ABC, abstractmethod
from typing import List


class BaseEmbedder(ABC):
    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        pass


class OpenAIEmbedder(BaseEmbedder):
    def __init__(self):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package required: pip install openai")

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        self.client = OpenAI(api_key=api_key)

    def embed(self, texts: List[str]) -> List[List[float]]:
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=texts,
        )
        return [item.embedding for item in response.data]


class GeminiEmbedder(BaseEmbedder):
    def __init__(self):
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError(
                "google-generativeai package required: pip install google-generativeai"
            )

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")

        genai.configure(api_key=api_key)
        self._genai = genai

    def embed(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            result = self._genai.embed_content(
                model="models/embedding-001",
                content=text,
                task_type="retrieval_document",
            )
            embeddings.append(result["embedding"])
        return embeddings


class LocalEmbedder(BaseEmbedder):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers package required: pip install sentence-transformers"
            )

        print(f"Loading local embedding model '{model_name}' (first run may download it)...")
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, convert_to_numpy=True).tolist()


def get_embedder(provider: str) -> BaseEmbedder:
    if provider == "openai":
        return OpenAIEmbedder()
    elif provider == "gemini":
        return GeminiEmbedder()
    elif provider == "anthropic":
        raise ValueError(
            "RAG is not supported with the anthropic provider. "
            "Use --provider openai or --provider gemini for RAG."
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")
