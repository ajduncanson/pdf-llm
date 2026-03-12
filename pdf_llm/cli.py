import argparse
import sys

from dotenv import load_dotenv

from .core import load_pdfs, check_context_length
from .providers import PROVIDERS

load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Query PDF documents using an LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python main.py --pdf report.pdf --prompt "Summarize this document" --provider anthropic
  python main.py --pdf doc1.pdf doc2.pdf --prompt "Compare these documents" --provider openai
  python main.py --pdf paper.pdf --prompt "What are the key findings?" --provider gemini --model gemini-1.5-flash
        """,
    )
    parser.add_argument(
        "--pdf", nargs="+", required=True, metavar="FILE",
        help="One or more PDF files to process",
    )
    parser.add_argument(
        "--prompt", required=True,
        help="The question or instruction for the LLM",
    )
    parser.add_argument(
        "--provider", required=True, choices=list(PROVIDERS.keys()),
        help="LLM provider to use",
    )
    parser.add_argument(
        "--model", default=None,
        help="Specific model to use (overrides the provider default)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading {len(args.pdf)} PDF(s)...")
    try:
        context = load_pdfs(args.pdf)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    context = check_context_length(context, args.provider)

    model_label = args.model or "default model"
    print(f"Querying {args.provider} ({model_label})...")
    try:
        provider = PROVIDERS[args.provider]()
        response = provider.query(args.prompt, context, args.model)
    except (ValueError, ImportError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print("\n" + "=" * 60)
    print(response)
