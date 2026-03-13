import argparse
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from .core import load_pdfs, check_context_length
from .providers import PROVIDERS

load_dotenv()

GOVERNANCE_CONFIG_PATH = Path(__file__).parent.parent / "governance_config.yaml"


def _load_governance_logger():
    """Load config and return a GovernanceLogger, or None if unavailable."""
    try:
        import yaml
        from .governance_logger import GovernanceLogger

        if not GOVERNANCE_CONFIG_PATH.exists():
            print("[governance] Warning: governance_config.yaml not found — logging disabled.")
            return None

        with open(GOVERNANCE_CONFIG_PATH) as f:
            config = yaml.safe_load(f)
        return GovernanceLogger(config)
    except Exception as e:
        print(f"[governance] Warning: could not initialise logger — {e}")
        return None


def _build_source_docs(pdf_paths, logger):
    from pypdf import PdfReader
    source_docs = []
    for path in pdf_paths:
        p = Path(path)
        try:
            page_count = len(PdfReader(str(p)).pages)
        except Exception:
            page_count = None
        source_docs.append({
            "filename": p.name,
            "sha256_hash": logger.compute_document_hash(str(p)),
            "page_count": page_count,
            "ingestion_timestamp": None,
        })
    return source_docs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Query PDF documents using an LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python main.py --pdf report.pdf --prompt "Summarize this document" --provider anthropic
  python main.py --pdf doc1.pdf doc2.pdf --prompt "Compare these documents" --provider openai
  python main.py --pdf paper.pdf --prompt "What are the key findings?" --provider gemini --model gemini-1.5-flash
  python main.py --pdf report.pdf --prompt "Key risks?" --provider openai --rag
  python main.py --pdf report.pdf --prompt "Key risks?" --provider openai --rag --chunk-size 300 --top-k 8
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
    parser.add_argument(
        "--rag", action="store_true",
        help="Use RAG pipeline (chunk, embed, retrieve) instead of sending full text",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=500, metavar="N",
        help="Words per chunk when using --rag (default: 500)",
    )
    parser.add_argument(
        "--top-k", type=int, default=5, metavar="N",
        help="Number of chunks to retrieve when using --rag (default: 5)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.rag and args.provider == "anthropic":
        print(
            "Error: --rag is not supported with --provider anthropic.\n"
            "Use --provider openai or --provider gemini for RAG, or use\n"
            "--provider anthropic without --rag for full-context mode.",
            file=sys.stderr,
        )
        sys.exit(1)

    logger = _load_governance_logger()

    try:
        if args.rag:
            from .rag import run_rag
            response, trace_id = run_rag(
                pdf_paths=args.pdf,
                prompt=args.prompt,
                provider_name=args.provider,
                model=args.model,
                chunk_size=args.chunk_size,
                top_k=args.top_k,
                logger=logger,
            )
        else:
            pipeline_start = time.monotonic()

            print(f"Loading {len(args.pdf)} PDF(s)...")
            context = load_pdfs(args.pdf)
            context = check_context_length(context, args.provider)

            # Build log entry before the LLM call so it exists on failure
            entry = None
            trace_id = None
            provider = PROVIDERS[args.provider]()
            if logger:
                try:
                    source_docs = _build_source_docs(args.pdf, logger)
                    entry = logger.build_log_entry(
                        prompt=args.prompt,
                        source_documents=source_docs,
                    )
                    trace_id = entry["trace_id"]
                    entry["provider"] = args.provider
                    entry["model_id"] = args.model or provider.default_model
                    entry["max_tokens"] = 4096
                except Exception as e:
                    print(f"[governance] Warning: failed to build log entry — {e}")

            model_label = args.model or "default model"
            print(f"Querying {args.provider} ({model_label})...")

            llm_start = time.monotonic()
            try:
                response, llm_meta = provider.query_with_metadata(args.prompt, context, args.model)
            except RuntimeError as e:
                total_ms = int((time.monotonic() - pipeline_start) * 1000)
                if logger and entry:
                    try:
                        entry["pipeline_status"] = "failed"
                        entry["error"] = str(e)
                        entry["total_latency_ms"] = total_ms
                        entry["flagged_for_review"] = True
                        entry["flag_reasons"] = [f"API error: {e}"]
                        logger.write(entry)
                    except Exception as log_err:
                        print(f"[governance] Warning: failed to log error entry — {log_err}")
                raise

            llm_ms = int((time.monotonic() - llm_start) * 1000)
            total_ms = int((time.monotonic() - pipeline_start) * 1000)

            if logger and entry:
                try:
                    model_id = llm_meta.get("model_id") or args.model or provider.default_model
                    entry["model_id"] = model_id
                    entry["prompt_tokens_used"] = llm_meta.get("prompt_tokens")
                    entry["completion_tokens_used"] = llm_meta.get("completion_tokens")
                    entry["total_tokens_used"] = llm_meta.get("total_tokens")
                    entry["llm_latency_ms"] = llm_ms
                    entry["total_latency_ms"] = total_ms
                    if llm_meta.get("total_tokens"):
                        entry["estimated_cost_usd"] = logger._estimate_cost(
                            model_id, llm_meta["total_tokens"]
                        )
                    logger.populate_response_fields(entry, response, model_id)
                    entry = logger.check_flags(entry)
                    logger.write(entry)
                except Exception as e:
                    print(f"[governance] Warning: logging failed — {e}")

    except (FileNotFoundError, ValueError, ImportError, RuntimeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if trace_id:
        print(f"\n[trace_id: {trace_id}]")
    print("\n" + "=" * 60)
    print(response)
