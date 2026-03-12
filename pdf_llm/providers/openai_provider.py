import os

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

    def query(self, prompt: str, context: str, model: str = None) -> str:
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
        return response.choices[0].message.content
