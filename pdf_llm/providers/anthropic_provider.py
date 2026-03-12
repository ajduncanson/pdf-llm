import os

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

    def query(self, prompt: str, context: str, model: str = None) -> str:
        model = model or self.default_model
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
        return response.content[0].text
