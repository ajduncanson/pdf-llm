# Governance Logging

## What is logged and why

Every pipeline run writes a structured log entry to a JSONL file. The entry captures the full lifecycle of a request for auditability, cost tracking, and quality monitoring:

- **Input** — prompt, estimated token count, source document metadata (filename, SHA256 hash, page count)
- **Chunking** — strategy, chunk size, overlap
- **Retrieval** — chunks retrieved, similarity scores, latency, confidence signal
- **Generation** — provider, model, token usage, estimated cost, latency
- **Output** — response text, citation detection
- **Quality scores** — faithfulness, relevance, hallucination, toxicity (populated externally via `score()`)
- **Safety flags** — prompt injection, jailbreak, PII, sensitive topics
- **Human oversight** — flagged-for-review status, flag reasons

**Raw PII is never stored.** Only boolean flags (`pii_detected_in_prompt`, `output_pii_detected`) are recorded.

---

## Where log files are stored

```
./logs/governance_YYYY-MM-DD.jsonl
```

One file per day, with one JSON object per line (JSONL format). Controlled by `logging.log_dir` in `governance_config.yaml`.

---

## Reading logs

```bash
# Today's log — full table
python log_reader.py

# Specific date
python log_reader.py --date 2025-06-01

# Flagged entries only (with reasons and scores)
python log_reader.py --flagged-only

# Aggregate stats (counts, averages, cost, model breakdown)
python log_reader.py --summary
```

---

## Thresholds that trigger real-time flags

A `flagged_for_review = True` flag is written to the log entry, and a WARNING is printed to stderr, if any of the following are breached:

| Condition | Default threshold |
|---|---|
| `hallucination_score` too high | > 0.3 |
| `faithfulness_score` too low | < 0.6 |
| `answer_relevance_score` too low | < 0.6 |
| `toxicity_score` too high | > 0.05 |
| `toxicity_flag` is True | — |
| `prompt_injection_detected` is True | — |
| `output_pii_detected` is True | — |
| `low_retrieval_confidence` is True | top chunk similarity < 0.5 |
| `total_latency_ms` too high | > 10,000 ms |

---

## Adjusting thresholds

Edit `governance_config.yaml`:

```yaml
thresholds:
  hallucination_score_max: 0.3
  faithfulness_score_min: 0.6
  answer_relevance_score_min: 0.6
  toxicity_score_max: 0.05
  top_chunk_similarity_min: 0.5
  total_latency_ms_max: 10000
```

Cost rates (per million tokens) can also be updated here as provider pricing changes.

---

## Note on PII

The logger records **only flags**, never content. `pii_detected_in_prompt` and `output_pii_detected` are booleans set by external detection logic — no personal data is extracted or stored in the log.
