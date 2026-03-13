"""
RAGAS-based quality scorer for governance log entries.

Install RAGAS before using:
  pip install ragas

Called by score_logs.py for deferred (batch) evaluation — not invoked inline
during pipeline execution.
"""

import os
from typing import Any, Dict, List, Optional


# Maps governance log field names ↔ RAGAS metric names
_RAGAS_TO_LOG_KEY = {
    "faithfulness": "faithfulness_score",
    "answer_relevancy": "answer_relevance_score",
    "context_precision": "context_precision_score",
    "context_recall": "context_recall_score",
}


class RAGASScorer:
    def __init__(self, config: Dict[str, Any]):
        scoring = config.get("scoring", {})
        self.evaluator_provider = scoring.get("evaluator_provider", "openai")
        self.evaluator_model = scoring.get("evaluator_model", "gpt-4o-mini")
        self.enabled_metrics = scoring.get(
            "metrics", ["faithfulness", "answer_relevancy", "context_precision"]
        )

    def score_entry(self, entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Score a single log entry dict. Returns a scores dict ready to pass to
        GovernanceLogger.score(), or None if the entry lacks required fields.

        Required entry fields: raw_prompt, response_text
        Optional but recommended: chunks_retrieved[].chunk_text (for context metrics)
        """
        try:
            from ragas import evaluate
            from ragas.evaluation import EvaluationDataset, SingleTurnSample
        except ImportError:
            raise ImportError(
                "ragas package required for scoring: pip install ragas"
            )

        prompt = entry.get("raw_prompt")
        response = entry.get("response_text")
        if not prompt or not response:
            return None

        # Extract full chunk text from log; fall back to preview if full text absent
        contexts = [
            c.get("chunk_text") or c.get("chunk_text_preview", "")
            for c in entry.get("chunks_retrieved", [])
            if c.get("chunk_text") or c.get("chunk_text_preview")
        ]

        # Metrics that require a ground-truth reference — excluded unless one is provided
        REQUIRES_REFERENCE = {"context_precision", "context_recall"}

        metrics_to_run = [
            m for m in self.enabled_metrics if m not in REQUIRES_REFERENCE
        ]
        # Without contexts, only answer_relevancy is computable
        if not contexts:
            metrics_to_run = [m for m in metrics_to_run if m == "answer_relevancy"]
        if not metrics_to_run:
            return None

        sample = SingleTurnSample(
            user_input=prompt,
            response=response,
            retrieved_contexts=contexts or None,
        )
        dataset = EvaluationDataset(samples=[sample])

        metric_objects = self._build_metrics(metrics_to_run)
        evaluator_llm = self._build_evaluator_llm()

        kwargs: Dict[str, Any] = {
            "dataset": dataset,
            "metrics": metric_objects,
            "raise_exceptions": False,
        }
        if evaluator_llm is not None:
            kwargs["llm"] = evaluator_llm

        result = evaluate(**kwargs)

        # result.scores is a list of dicts, one per sample (we always have one sample)
        sample_scores = result.scores[0] if result.scores else {}

        scores: Dict[str, Any] = {}
        for ragas_key, log_key in _RAGAS_TO_LOG_KEY.items():
            if ragas_key not in metrics_to_run:
                continue
            val = sample_scores.get(ragas_key)
            if val is not None:
                try:
                    fval = float(val)
                    import math
                    scores[log_key] = round(fval, 4) if not math.isnan(fval) else None
                except (TypeError, ValueError):
                    scores[log_key] = None

        # Derive hallucination as the complement of faithfulness
        faith = scores.get("faithfulness_score")
        if faith is not None:
            scores["hallucination_score"] = round(1.0 - faith, 4)
            scores["hallucination_flag"] = scores["hallucination_score"] > 0.3

        scores["scorer_method"] = "ragas"
        scores["scorer_model_used"] = self.evaluator_model
        return scores

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_metrics(self, metric_names: List[str]):
        from ragas import metrics as ragas_metrics

        class_map = {
            "faithfulness": ragas_metrics.Faithfulness,
            "answer_relevancy": ragas_metrics.AnswerRelevancy,
            "context_precision": ragas_metrics.ContextPrecision,
            "context_recall": ragas_metrics.ContextRecall,
        }
        return [class_map[n]() for n in metric_names if n in class_map]

    def _build_evaluator_llm(self):
        """
        Returns a RAGAS-compatible LLM evaluator via LangChain wrappers
        (langchain is a core RAGAS dependency), or None to use the RAGAS
        default (OpenAI, requires OPENAI_API_KEY in env).
        """
        from ragas.llms import LangchainLLMWrapper

        if self.evaluator_provider == "openai":
            from langchain_openai import ChatOpenAI
            return LangchainLLMWrapper(ChatOpenAI(model=self.evaluator_model))

        if self.evaluator_provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            return LangchainLLMWrapper(
                ChatAnthropic(
                    model=self.evaluator_model,
                    api_key=os.getenv("ANTHROPIC_API_KEY"),
                    max_tokens=4096,
                )
            )

        if self.evaluator_provider == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI
            return LangchainLLMWrapper(
                ChatGoogleGenerativeAI(
                    model=self.evaluator_model,
                    google_api_key=os.getenv("GEMINI_API_KEY"),
                )
            )

        return None
