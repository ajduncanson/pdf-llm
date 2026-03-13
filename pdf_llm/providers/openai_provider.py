import os
from typing import Any, Dict, Tuple

from .base import BaseProvider


class OpenAIProvider(BaseProvider):
    default_model = "gpt-5-mini-2025-08-07"

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
        import openai

        model = model or self.default_model
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": f"Here are the documents:\n\n{context}\n\n{prompt}",
                    }
                ],
            )
        except openai.AuthenticationError:
            raise RuntimeError("OpenAI authentication failed — check your OPENAI_API_KEY.")
        except openai.RateLimitError as e:
            raise RuntimeError(
                f"OpenAI rate limit or quota exceeded. "
                f"Check your usage at https://platform.openai.com/usage\nDetail: {e}"
            )
        except openai.BadRequestError as e:
            raise RuntimeError(f"OpenAI rejected the request (bad request): {e}")
        except openai.APIConnectionError as e:
            raise RuntimeError(f"Could not connect to OpenAI API: {e}")
        except openai.APIStatusError as e:
            raise RuntimeError(f"OpenAI API error {e.status_code}: {e.message}")

        text = response.choices[0].message.content
        metadata = {
            "model_id": response.model,
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }
        return text, metadata
