# pdf-llm

Query PDF documents using LLMs from OpenAI, Anthropic, or Gemini. Supports full-context mode (entire PDF sent to the model) and a RAG pipeline (chunk, embed, retrieve) for large documents.

---

## Table of contents

- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Running with Docker](#running-with-docker)
- [Running locally (without Docker)](#running-locally-without-docker)
- [Usage reference](#usage-reference)
- [Governance logging](#governance-logging)
- [Quality scoring (RAGAS)](#quality-scoring-ragas)

---

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running
- API keys for whichever provider(s) you intend to use:
  - OpenAI: https://platform.openai.com/api-keys
  - Anthropic: https://console.anthropic.com/settings/keys
  - Gemini: https://aistudio.google.com/app/apikey

---

## Setup

These steps are required once on any new machine.

**1. Clone the repository**

```bash
git clone <repo-url>
cd pdf-llm
```

**2. Create your environment file**

```bash
cp .env.example .env
```

Open `.env` and fill in the API keys for the providers you want to use. You only need the key(s) for the provider(s) you plan to call — leave the others blank.

```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=AIza...
```

**3. Create local folders**

The container writes governance logs to `./logs/` and reads PDFs from wherever you point it. Create the logs folder so Docker can mount it:

```bash
mkdir -p logs docs
```

**4. Build the Docker image**

```bash
docker compose build
```

This only needs to be re-run when you change `requirements.txt` or the application code.

---

## Running with Docker

The general form of the command is:

```bash
docker compose run --rm pdf-llm --pdf docs/<filename>.pdf --prompt "<your question>" --provider <provider>
```

The `docker-compose.yml` file handles the boilerplate — volumes and env file are configured once so you don't repeat them on every command:

| Your machine | Inside container | Purpose |
|---|---|---|
| `./docs/` | `/app/docs/` | Where you place input PDFs |
| `./logs/` | `/app/logs/` | Where governance logs are written |

Files written to `/app/logs/` inside the container appear immediately in `./logs/` on your machine, and survive after the container exits.

### Examples

**Full-context mode (Anthropic)** — send the entire PDF to the model:

```bash
docker compose run --rm pdf-llm \
  --pdf docs/report.pdf --prompt "Summarise the key findings" --provider anthropic
```

**Full-context mode (OpenAI)** — use a specific model:

```bash
docker compose run --rm pdf-llm \
  --pdf docs/report.pdf --prompt "What are the risks?" --provider openai --model gpt-4o
```

**RAG mode (OpenAI)** — chunk and retrieve before querying; better for large PDFs:

```bash
docker compose run --rm pdf-llm \
  --pdf docs/report.pdf --prompt "What does section 3 say about compliance?" \
  --provider openai --rag
```

**RAG with tuned retrieval:**

```bash
docker compose run --rm pdf-llm \
  --pdf docs/report.pdf --prompt "Key obligations under clause 5" \
  --provider gemini --rag --chunk-size 300 --top-k 8
```

**Multiple PDFs:**

```bash
docker compose run --rm pdf-llm \
  --pdf docs/doc1.pdf docs/doc2.pdf \
  --prompt "Compare the approaches described in these two documents" \
  --provider anthropic
```

> **Note:** RAG mode is supported for `openai` and `gemini` only. Using `--rag` with `--provider anthropic` will return an error — use full-context mode instead.

---

## Running locally (without Docker)

**1. Create and activate a virtual environment**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Run**

```bash
python main.py --pdf path/to/doc.pdf --prompt "Your question" --provider anthropic
```

---

## Usage reference

```
python main.py --pdf FILE [FILE ...] --prompt TEXT --provider PROVIDER [options]

Required:
  --pdf FILE [FILE ...]   One or more PDF files to process
  --prompt TEXT           The question or instruction for the LLM
  --provider PROVIDER     LLM provider: openai | anthropic | gemini

Options:
  --model MODEL           Override the provider's default model
  --rag                   Use RAG pipeline instead of full-context mode
                          (openai and gemini only)
  --chunk-size N          Words per chunk when using --rag (default: 500)
  --top-k N               Chunks to retrieve when using --rag (default: 5)
```

### Provider defaults

| Provider | Default model |
|---|---|
| `anthropic` | `claude-sonnet-4-20250514` |
| `openai` | `gpt-5-mini-2025-08-07` |
| `gemini` | `gemini-1.5-pro` |

---

## Governance logging

Every run writes a structured log entry to `./logs/governance_YYYY-MM-DD.jsonl`. Each entry records:

- Prompt, response, provider, model, token counts
- Source document metadata (filename, SHA-256 hash, page count)
- Latency (pipeline and LLM separately)
- Estimated cost in USD
- Quality flags (hallucination, low retrieval confidence, high latency, etc.)
- RAG retrieval details (chunks, similarity scores) when `--rag` is used

**Read today's log:**

```bash
python log_reader.py
```

**Read a specific date, flagged entries only:**

```bash
python log_reader.py --date 2025-06-01 --flagged-only
```

**Summary statistics:**

```bash
python log_reader.py --summary
```

---

## Quality scoring (RAGAS)

Deferred quality scoring uses [RAGAS](https://github.com/explodinggradients/ragas) to evaluate faithfulness and answer relevancy after the fact, without adding latency to the main pipeline.

**Install the optional dependency:**

```bash
pip install ragas langchain-anthropic
```

**Score today's log:**

```bash
python score_logs.py
```

**Score a specific date, skipping already-scored entries:**

```bash
python score_logs.py --date 2025-06-01 --unscored-only
```

**Dry run (show what would be scored without calling the API):**

```bash
python score_logs.py --dry-run
```

Scoring is configured in `governance_config.yaml` under the `scoring` key. Set `enabled: true` to activate it, and choose your evaluator provider and model. The default uses `claude-haiku-4-5` via Anthropic, which is fast and inexpensive for evaluation.

> RAGAS scoring is not available inside the Docker container — it is intended to be run locally against the log files that the container writes to `./logs/`.
