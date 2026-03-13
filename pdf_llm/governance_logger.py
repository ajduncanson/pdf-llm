import hashlib
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


class GovernanceLogger:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.config_hash = self._hash_dict(config)
        self.log_dir = Path(config.get("logging", {}).get("log_dir", "./logs"))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.thresholds = config.get("thresholds", {})
        self.cost_table = config.get("cost_per_million_tokens", {})

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_log_entry(
        self,
        prompt: str,
        source_documents: Optional[List[Dict]] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        return {
            # Trace
            "trace_id": str(uuid.uuid4()),
            "session_id": session_id or str(uuid.uuid4()),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "environment": self.config.get("environment", "dev"),
            "pipeline_version": self.config.get("pipeline_version", "unknown"),
            "config_hash": self.config_hash,
            # Input
            "raw_prompt": prompt,
            "prompt_token_count": self._estimate_tokens(prompt),
            "prompt_language": self._detect_language(prompt),
            "source_documents": source_documents or [],
            "pii_detected_in_prompt": False,
            "prompt_injection_flag": False,
            # Chunking
            "chunking_strategy": None,
            "chunk_size_tokens": None,
            "chunk_overlap_tokens": None,
            "embedding_model": None,
            "embedding_model_version": None,
            # Retrieval
            "k_chunks_requested": None,
            "k_chunks_returned": None,
            "search_type": None,
            "reranker_used": False,
            "chunks_retrieved": [],
            "total_context_tokens": None,
            "retrieval_latency_ms": None,
            "top_chunk_similarity": None,
            "low_retrieval_confidence": False,
            # Generation
            "provider": None,
            "model_id": None,
            "model_version": None,
            "temperature": None,
            "max_tokens": None,
            "system_prompt_hash": None,
            "prompt_tokens_used": None,
            "completion_tokens_used": None,
            "total_tokens_used": None,
            "estimated_cost_usd": None,
            "llm_latency_ms": None,
            "total_latency_ms": None,
            # Pipeline status
            "pipeline_status": "success",
            "error": None,
            # Output
            "response_text": None,
            "response_token_count": None,
            "citations_included": None,
            # Quality metrics
            "faithfulness_score": None,
            "answer_relevance_score": None,
            "context_precision_score": None,
            "context_recall_score": None,
            "hallucination_score": None,
            "hallucination_flag": None,
            "toxicity_score": None,
            "toxicity_flag": None,
            "summarisation_coverage_score": None,
            "qa_groundedness_score": None,
            "scorer_method": None,
            "scorer_model_used": None,
            # Safety
            "prompt_injection_detected": False,
            "jailbreak_attempt_detected": False,
            "sensitive_topic_flag": False,
            "refusal_triggered": False,
            "output_pii_detected": False,
            "document_sensitivity_level": "internal",
            # Human oversight
            "flagged_for_review": False,
            "flag_reasons": [],
            "reviewer_id": None,
            "review_outcome": None,
            "user_feedback_rating": None,
            # Governance metadata
            "data_classification": self.config.get("data_classification", "internal"),
            "retention_policy_days": self.config.get("retention_policy_days", 365),
            "jurisdiction": self.config.get("jurisdiction", "AU"),
        }

    def score(self, entry: Dict[str, Any], scores_dict: Dict[str, float]) -> Dict[str, Any]:
        entry.update(scores_dict)
        return entry

    def check_flags(
        self, entry: Dict[str, Any], thresholds: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        t = thresholds or self.thresholds
        reasons = []

        hallucination_max = t.get("hallucination_score_max", 0.3)
        faithfulness_min = t.get("faithfulness_score_min", 0.6)
        relevance_min = t.get("answer_relevance_score_min", 0.6)
        toxicity_max = t.get("toxicity_score_max", 0.05)
        latency_max = t.get("total_latency_ms_max", 10000)

        h = entry.get("hallucination_score")
        if h is not None and h > hallucination_max:
            reasons.append(f"hallucination_score {h:.3f} > {hallucination_max}")

        f = entry.get("faithfulness_score")
        if f is not None and f < faithfulness_min:
            reasons.append(f"faithfulness_score {f:.3f} < {faithfulness_min}")

        r = entry.get("answer_relevance_score")
        if r is not None and r < relevance_min:
            reasons.append(f"answer_relevance_score {r:.3f} < {relevance_min}")

        tx = entry.get("toxicity_score")
        if tx is not None and tx > toxicity_max:
            reasons.append(f"toxicity_score {tx:.3f} > {toxicity_max}")

        if entry.get("toxicity_flag"):
            reasons.append("toxicity_flag is True")

        if entry.get("prompt_injection_detected"):
            reasons.append("prompt_injection_detected")

        if entry.get("output_pii_detected"):
            reasons.append("output_pii_detected")

        if entry.get("low_retrieval_confidence"):
            reasons.append("low_retrieval_confidence (top chunk similarity below threshold)")

        lat = entry.get("total_latency_ms")
        if lat is not None and lat > latency_max:
            reasons.append(f"total_latency_ms {lat} > {latency_max}")

        if reasons:
            entry["flagged_for_review"] = True
            entry["flag_reasons"] = reasons

        return entry

    def write(self, entry: Dict[str, Any]) -> None:
        log_path = self.log_dir / f"governance_{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.jsonl"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=self._json_serialiser) + "\n")

        # One-line stdout summary
        ts = entry.get("timestamp_utc", "")[:19]
        trace = entry.get("trace_id", "")[:8]
        flagged = entry.get("flagged_for_review", False)
        score_parts = []
        for key in ("faithfulness_score", "answer_relevance_score", "hallucination_score"):
            v = entry.get(key)
            if v is not None:
                score_parts.append(f"{key.replace('_score','')[:8]}={v:.2f}")
        scores_str = " | ".join(score_parts) if score_parts else "no scores yet"
        print(f"[governance] {ts} trace={trace}... flagged={flagged} {scores_str}")

        if flagged:
            reasons = ", ".join(entry.get("flag_reasons", []))
            print(f"WARNING: trace {trace}... flagged for review — {reasons}", file=sys.stderr)

    def populate_response_fields(
        self, entry: Dict[str, Any], response_text: str, model_id: str
    ) -> Dict[str, Any]:
        """Convenience method to fill output fields from a plain string response."""
        import re
        entry["response_text"] = response_text
        entry["response_token_count"] = self._estimate_tokens(response_text)
        citation_pattern = re.compile(r"\[.+?\]|Source:", re.IGNORECASE)
        entry["citations_included"] = bool(citation_pattern.search(response_text))
        if model_id and entry.get("total_tokens_used"):
            entry["estimated_cost_usd"] = self._estimate_cost(
                model_id, entry["total_tokens_used"]
            )
        return entry

    @staticmethod
    def compute_document_hash(filepath: str) -> str:
        h = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _hash_dict(self, d: Dict) -> str:
        serialised = json.dumps(d, sort_keys=True, default=str)
        return hashlib.sha256(serialised.encode()).hexdigest()

    def _estimate_tokens(self, text: str) -> int:
        return int(len(text.split()) * 1.3)

    def _detect_language(self, text: str) -> str:
        try:
            from langdetect import detect
            return detect(text)
        except Exception:
            return "en"

    def _estimate_cost(self, model_id: str, total_tokens: int) -> float:
        rate = self.cost_table.get(model_id, 0.0)
        return round((total_tokens / 1_000_000) * rate, 6)

    @staticmethod
    def _json_serialiser(obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")
