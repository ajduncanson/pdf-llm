import os
from typing import Any, Dict, Tuple

from .base import BaseProvider


class AnthropicProvider(BaseProvider):
    default_model = "claude-sonnet-4-20250514"

    def __init__(self):
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic package required: pip install anthropic")

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")

        self.client = anthropic.Anthropic(api_key=api_key)

    def query_with_metadata(
        self, prompt: str, context: str, model: str = None
    ) -> Tuple[str, Dict[str, Any]]:
        import anthropic

        model = model or self.default_model
        try:
            response = self.client.messages.create(
                model=model,
                max_tokens=4096,
                messages=[
                    {
                        "role": "user",
                        "content": f"Here are the documents:\n\n{context}\n\n{prompt}",
                    }
                ],
            )
        except anthropic.AuthenticationError:
            raise RuntimeError("Anthropic authentication failed — check your ANTHROPIC_API_KEY.")
        except anthropic.RateLimitError as e:
            raise RuntimeError(
                f"Anthropic rate limit exceeded. Wait and retry, or check your plan.\nDetail: {e}"
            )
        except anthropic.BadRequestError as e:
            raise RuntimeError(f"Anthropic rejected the request (bad request): {e}")
        except anthropic.APIConnectionError as e:
            raise RuntimeError(f"Could not connect to Anthropic API: {e}")
        except anthropic.APIStatusError as e:
            raise RuntimeError(f"Anthropic API error {e.status_code}: {e.message}")

        text = response.content[0].text
        metadata = {
            "model_id": response.model,
            "prompt_tokens": response.usage.input_tokens,
            "completion_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
        }
        return text, metadata
