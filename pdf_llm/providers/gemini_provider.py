import os
from typing import Any, Dict, Tuple

from .base import BaseProvider


class GeminiProvider(BaseProvider):
    default_model = "gemini-1.5-pro"

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

    def query_with_metadata(
        self, prompt: str, context: str, model: str = None
    ) -> Tuple[str, Dict[str, Any]]:
        model_name = model or self.default_model
        model_obj = self._genai.GenerativeModel(model_name)
        response = model_obj.generate_content(
            f"Here are the documents:\n\n{context}\n\n{prompt}"
        )
        text = response.text
        usage = response.usage_metadata
        prompt_tokens = getattr(usage, "prompt_token_count", None)
        completion_tokens = getattr(usage, "candidates_token_count", None)
        total_tokens = getattr(usage, "total_token_count", None)
        metadata = {
            "model_id": model_name,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }
        return text, metadata
