from pathlib import Path
from typing import List

try:
    from pypdf import PdfReader
except ImportError:
    raise ImportError("pypdf is required: pip install pypdf")

# Approximate token limits, leaving headroom for the prompt and response
CONTEXT_LIMITS = {
    "openai": 120_000,
    "anthropic": 180_000,
    "gemini": 900_000,
}

CHARS_PER_TOKEN = 4  # rough approximation


def extract_text_from_pdf(path: str) -> str:
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            pages.append(f"[Page {i + 1}]\n{text}")
    return "\n\n".join(pages)


def load_pdfs(paths: List[str]) -> str:
    parts = []
    for path in paths:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"PDF not found: {path}")
        if p.suffix.lower() != ".pdf":
            raise ValueError(f"Not a PDF file: {path}")
        text = extract_text_from_pdf(str(p))
        parts.append(f"=== Document: {p.name} ===\n{text}")
    return "\n\n".join(parts)


def check_context_length(text: str, provider: str) -> str:
    estimated_tokens = len(text) // CHARS_PER_TOKEN
    limit = CONTEXT_LIMITS.get(provider, 120_000)

    if estimated_tokens > limit:
        print(
            f"Warning: estimated {estimated_tokens:,} tokens exceeds {provider} "
            f"limit of {limit:,}. Truncating."
        )
        text = text[: limit * CHARS_PER_TOKEN]
        text += "\n\n[... content truncated due to context limit ...]"
    elif estimated_tokens > limit * 0.8:
        print(
            f"Warning: estimated {estimated_tokens:,} tokens is close to "
            f"{provider} limit of {limit:,}."
        )

    return text
