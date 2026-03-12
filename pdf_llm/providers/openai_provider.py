import os
from typing import Any, Dict, Tuple

from .base import BaseProvider


class OpenAIProvider(BaseProvider):
    default_model = "gpt-4o"

    def __init__(self):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package required: pip install openai")

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        self.client = OpenAI(api_key=api_key)

    def query_with_metadata(
        self, prompt: str, context: str, model: str = None
    ) -> Tuple[str, Dict[str, Any]]:
        model = model or self.default_model
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": f"Here are the documents:\n\n{context}\n\n{prompt}",
                }
            ],
        )
        text = response.choices[0].message.content
        metadata = {
            "model_id": response.model,
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }
        return text, metadata
