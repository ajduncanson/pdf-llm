import time
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple

from .chunker import chunk_text
from .core import load_pdfs
from .embedder import get_embedder
from .providers import PROVIDERS
from .vector_store import VectorStore

if TYPE_CHECKING:
    from .governance_logger import GovernanceLogger


def run_rag(
    pdf_paths: List[str],
    prompt: str,
    provider_name: str,
    model: str = None,
    chunk_size: int = 300,
    overlap: int = 50,
    top_k: int = 10,
    logger: Optional["GovernanceLogger"] = None,
    session_id: Optional[str] = None,
) -> Tuple[str, Optional[str]]:
    """
    Returns (response_text, trace_id).
    trace_id is None if logger is not provided.
    """
    pipeline_start = time.monotonic()
    entry = None
    trace_id = None

    # --- Governance: build entry and collect document metadata ---
    if logger:
        try:
            source_docs = []
            for path in pdf_paths:
                p = Path(path)
                try:
                    from pypdf import PdfReader
                    page_count = len(PdfReader(str(p)).pages)
                except Exception:
                    page_count = None
                source_docs.append({
                    "filename": p.name,
                    "sha256_hash": logger.compute_document_hash(str(p)),
                    "page_count": page_count,
                    "ingestion_timestamp": None,
                })
            entry = logger.build_log_entry(
                prompt=prompt,
                source_documents=source_docs,
                session_id=session_id,
            )
            trace_id = entry["trace_id"]
        except Exception as e:
            print(f"[governance] Warning: failed to build log entry — {e}")

    # 1. Extract text from all PDFs
    print(f"Loading {len(pdf_paths)} PDF(s)...")
    full_text = load_pdfs(pdf_paths)

    # 2. Chunk
    chunks = chunk_text(full_text, chunk_size=chunk_size, overlap=overlap)
    print(f"Split into {len(chunks)} chunks.")

    if entry:
        try:
            entry["chunking_strategy"] = "fixed"
            entry["chunk_size_tokens"] = chunk_size
            entry["chunk_overlap_tokens"] = overlap
        except Exception as e:
            print(f"[governance] Warning: failed to populate chunking fields — {e}")

    # 3. Embed chunks
    print("Embedding chunks...")
    embedder = get_embedder(provider_name)
    embedding_model = getattr(embedder, "model_name", type(embedder).__name__)

    if entry:
        try:
            entry["embedding_model"] = embedding_model
        except Exception as e:
            print(f"[governance] Warning: failed to populate embedding fields — {e}")

    chunk_embeddings = embedder.embed(chunks)

    # 4. Store in vector store
    store = VectorStore()
    store.add(chunks, chunk_embeddings)

    # 5. Embed the prompt and retrieve relevant chunks
    print("Retrieving relevant chunks...")
    retrieval_start = time.monotonic()
    prompt_embedding = embedder.embed([prompt])[0]
    relevant_chunks, similarities = store.query(prompt_embedding, top_k=top_k)
    retrieval_ms = int((time.monotonic() - retrieval_start) * 1000)

    similarity_threshold = 0.5
    if logger:
        try:
            similarity_threshold = logger.thresholds.get("top_chunk_similarity_min", 0.5)
        except Exception:
            pass

    if entry:
        try:
            top_sim = max(similarities) if similarities else None
            chunks_retrieved_meta = [
                {
                    "chunk_id": str(i),
                    "source_doc": None,
                    "page_number": None,
                    "similarity_score": round(sim, 4),
                    "rank_position": i + 1,
                    "chunk_text_preview": chunk[:120],
                    "chunk_text": chunk,  # full text retained for deferred RAGAS scoring
                }
                for i, (chunk, sim) in enumerate(zip(relevant_chunks, similarities))
            ]
            context_tokens = sum(
                int(len(c.split()) * 1.3) for c in relevant_chunks
            )
            entry["k_chunks_requested"] = top_k
            entry["k_chunks_returned"] = len(relevant_chunks)
            entry["search_type"] = "semantic"
            entry["reranker_used"] = False
            entry["chunks_retrieved"] = chunks_retrieved_meta
            entry["total_context_tokens"] = context_tokens
            entry["retrieval_latency_ms"] = retrieval_ms
            entry["top_chunk_similarity"] = top_sim
            entry["low_retrieval_confidence"] = (
                top_sim is not None and top_sim < similarity_threshold
            )
        except Exception as e:
            print(f"[governance] Warning: failed to populate retrieval fields — {e}")

    # 6. Query the LLM
    context = "\n\n---\n\n".join(relevant_chunks)
    print(f"Querying {provider_name} ({model or 'default model'}) with {len(relevant_chunks)} chunks...")

    llm_start = time.monotonic()
    provider = PROVIDERS[provider_name]()
    try:
        response_text, llm_meta = provider.query_with_metadata(prompt, context, model)
    except RuntimeError as e:
        total_ms = int((time.monotonic() - pipeline_start) * 1000)
        if logger and entry:
            try:
                entry["pipeline_status"] = "failed"
                entry["error"] = str(e)
                entry["provider"] = provider_name
                entry["model_id"] = model or provider.default_model
                entry["total_latency_ms"] = total_ms
                entry["flagged_for_review"] = True
                entry["flag_reasons"] = [f"API error: {e}"]
                logger.write(entry)
            except Exception as log_err:
                print(f"[governance] Warning: failed to log error entry — {log_err}")
        raise
    llm_ms = int((time.monotonic() - llm_start) * 1000)
    total_ms = int((time.monotonic() - pipeline_start) * 1000)

    if entry:
        try:
            model_id = llm_meta.get("model_id") or model or provider.default_model
            entry["provider"] = provider_name
            entry["model_id"] = model_id
            entry["max_tokens"] = 4096
            entry["prompt_tokens_used"] = llm_meta.get("prompt_tokens")
            entry["completion_tokens_used"] = llm_meta.get("completion_tokens")
            entry["total_tokens_used"] = llm_meta.get("total_tokens")
            entry["llm_latency_ms"] = llm_ms
            entry["total_latency_ms"] = total_ms
            if llm_meta.get("total_tokens") and logger:
                entry["estimated_cost_usd"] = logger._estimate_cost(
                    model_id, llm_meta["total_tokens"]
                )
            logger.populate_response_fields(entry, response_text, model_id)
            entry = logger.check_flags(entry)
            logger.write(entry)
        except Exception as e:
            print(f"[governance] Warning: failed to write log entry — {e}")

    return response_text, trace_id
