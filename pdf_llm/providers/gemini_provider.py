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
        try:
            from google.api_core.exceptions import (
                GoogleAPICallError,
                InvalidArgument,
                PermissionDenied,
                ResourceExhausted,
                ServiceUnavailable,
            )
        except ImportError:
            GoogleAPICallError = InvalidArgument = PermissionDenied = Exception
            ResourceExhausted = ServiceUnavailable = Exception

        model_name = model or self.default_model
        model_obj = self._genai.GenerativeModel(model_name)
        try:
            response = model_obj.generate_content(
                f"Here are the documents:\n\n{context}\n\n{prompt}"
            )
        except PermissionDenied:
            raise RuntimeError("Gemini authentication failed — check your GEMINI_API_KEY.")
        except ResourceExhausted as e:
            raise RuntimeError(
                f"Gemini quota or rate limit exceeded. Check your usage in Google AI Studio.\nDetail: {e}"
            )
        except InvalidArgument as e:
            raise RuntimeError(f"Gemini rejected the request (invalid argument): {e}")
        except ServiceUnavailable as e:
            raise RuntimeError(f"Gemini service unavailable: {e}")
        except GoogleAPICallError as e:
            raise RuntimeError(f"Gemini API error: {e}")

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
