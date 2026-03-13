#!/usr/bin/env python3
"""
Deferred RAGAS scoring for governance logs.

Reads a JSONL log file, scores entries using RAGAS, updates scores and
re-evaluates flags, then writes results back to the same file.

Requires: pip install ragas
Must be run from the pdf-llm project root with the venv active.

Usage:
  python score_logs.py                       # score today's log
  python score_logs.py --date 2025-06-01     # specific date
  python score_logs.py --unscored-only       # skip already-scored entries
  python score_logs.py --dry-run             # show what would be scored
  python score_logs.py --limit 10            # score at most N entries
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

LOG_DIR = Path("./logs")
CONFIG_PATH = Path("./governance_config.yaml")


def load_config():
    try:
        import yaml
    except ImportError:
        print("Error: pyyaml required — pip install pyyaml", file=sys.stderr)
        sys.exit(1)
    if not CONFIG_PATH.exists():
        print(f"Error: {CONFIG_PATH} not found.", file=sys.stderr)
        sys.exit(1)
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def is_scorable(entry: dict) -> bool:
    return bool(
        entry.get("pipeline_status") != "failed"
        and entry.get("response_text")
        and entry.get("raw_prompt")
    )


def is_already_scored(entry: dict) -> bool:
    return any(
        entry.get(k) is not None
        for k in ("faithfulness_score", "answer_relevance_score", "hallucination_score")
    )


def fmt(val) -> str:
    return f"{val:.3f}" if val is not None else "—"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Deferred RAGAS scoring for governance logs"
    )
    parser.add_argument(
        "--date",
        default=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        help="Date of log file to score (YYYY-MM-DD, default: today)",
    )
    parser.add_argument(
        "--unscored-only",
        action="store_true",
        help="Skip entries that already have scores",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show which entries would be scored without running RAGAS",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Maximum number of entries to score (useful for testing)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config()

    if not config.get("scoring", {}).get("enabled", False):
        print(
            "Scoring is disabled (scoring.enabled: false in governance_config.yaml).\n"
            "Set it to true to enable RAGAS scoring."
        )
        sys.exit(0)

    log_path = LOG_DIR / f"governance_{args.date}.jsonl"
    if not log_path.exists():
        print(f"No log file found: {log_path}")
        sys.exit(0)

    # Load all entries
    entries = []
    with open(log_path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: skipping line {i} — {e}", file=sys.stderr)

    # Determine candidates
    candidates = []
    for entry in entries:
        if not is_scorable(entry):
            continue
        if args.unscored_only and is_already_scored(entry):
            continue
        candidates.append(entry)

    if args.limit:
        candidates = candidates[: args.limit]

    print(
        f"Log : {log_path.name}\n"
        f"Total entries : {len(entries)}\n"
        f"To score      : {len(candidates)}\n"
    )

    if args.dry_run:
        print("Dry run — entries that would be scored:")
        for e in candidates:
            trace = (e.get("trace_id") or "")[:8]
            ts = (e.get("timestamp_utc") or "")[:19]
            n_chunks = len(e.get("chunks_retrieved") or [])
            already = " (already scored)" if is_already_scored(e) else ""
            print(f"  {trace}...  {ts}  chunks={n_chunks}{already}")
        return

    if not candidates:
        print("Nothing to score.")
        return

    try:
        from pdf_llm.governance_logger import GovernanceLogger
        from pdf_llm.scorer import RAGASScorer
    except ImportError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    scorer = RAGASScorer(config)
    logger = GovernanceLogger(config)

    # Index entries by trace_id for in-place updates
    index = {e.get("trace_id"): i for i, e in enumerate(entries)}

    scored = failed = skipped = 0

    for entry in candidates:
        trace = (entry.get("trace_id") or "")[:8]
        print(f"  {trace}...", end="  ", flush=True)

        try:
            scores = scorer.score_entry(entry)
            if scores is None:
                print("skipped (insufficient data)")
                skipped += 1
                continue

            updated = logger.score(entry, scores)
            updated = logger.check_flags(updated)

            idx = index.get(entry.get("trace_id"))
            if idx is not None:
                entries[idx] = updated

            faith = fmt(scores.get("faithfulness_score"))
            rel = fmt(scores.get("answer_relevance_score"))
            halluc = fmt(scores.get("hallucination_score"))
            flagged = "FLAGGED" if updated.get("flagged_for_review") else "ok"
            print(f"faith={faith}  relevance={rel}  halluc={halluc}  [{flagged}]")
            scored += 1

        except Exception as e:
            print(f"FAILED — {e}")
            failed += 1

    # Write all entries back (scored + unscored) preserving original order
    with open(log_path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, default=str) + "\n")

    print(
        f"\nDone.  scored={scored}  skipped={skipped}  failed={failed}\n"
        f"Written to: {log_path}"
    )


if __name__ == "__main__":
    main()
