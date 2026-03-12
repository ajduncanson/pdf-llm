import os

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

    def query(self, prompt: str, context: str, model: str = None) -> str:
        model_name = model or self.default_model
        model = self._genai.GenerativeModel(model_name)
        response = model.generate_content(
            f"Here are the documents:\n\n{context}\n\n{prompt}"
        )
        return response.text
