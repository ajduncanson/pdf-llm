from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple


class BaseProvider(ABC):
    @property
    @abstractmethod
    def default_model(self) -> str:
        pass

    @abstractmethod
    def query_with_metadata(
        self, prompt: str, context: str, model: str = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Returns (response_text, metadata_dict).
        metadata_dict keys: model_id, prompt_tokens, completion_tokens, total_tokens
        """
        pass

    def query(self, prompt: str, context: str, model: str = None) -> str:
        """Convenience wrapper — returns response text only."""
        text, _ = self.query_with_metadata(prompt, context, model)
        return text
