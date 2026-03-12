#!/usr/bin/env python3
"""
Read and summarise governance logs.

Usage:
  python log_reader.py                        # today's log, full table
  python log_reader.py --date 2025-06-01      # specific date
  python log_reader.py --flagged-only         # flagged entries only
  python log_reader.py --summary              # aggregate stats
"""

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


LOG_DIR = Path("./logs")


def parse_args():
    parser = argparse.ArgumentParser(description="Read governance logs")
    parser.add_argument(
        "--date",
        default=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        help="Date to read (YYYY-MM-DD, default: today)",
    )
    parser.add_argument(
        "--flagged-only",
        action="store_true",
        help="Show only entries flagged for review",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print aggregate stats instead of individual entries",
    )
    return parser.parse_args()


def load_entries(date: str):
    log_path = LOG_DIR / f"governance_{date}.jsonl"
    if not log_path.exists():
        print(f"No log file found for {date} ({log_path})")
        sys.exit(0)
    entries = []
    with open(log_path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: could not parse line {i} — {e}", file=sys.stderr)
    return entries


def _score_str(entry):
    parts = []
    for key, short in [
        ("faithfulness_score", "faith"),
        ("answer_relevance_score", "relevance"),
        ("hallucination_score", "halluc"),
    ]:
        v = entry.get(key)
        if v is not None:
            parts.append(f"{short}={v:.2f}")
    return " | ".join(parts) if parts else "—"


def print_table(entries):
    if not entries:
        print("No entries.")
        return
    header = f"{'TRACE_ID':<38}  {'TIMESTAMP':<20}  {'MODEL':<30}  {'LATENCY':>8}  {'FLAGGED'}"
    print(header)
    print("-" * len(header))
    for e in entries:
        trace = e.get("trace_id", "")
        ts = (e.get("timestamp_utc") or "")[:19]
        model = e.get("model_id") or "—"
        latency = e.get("total_latency_ms")
        latency_str = f"{latency:,}ms" if latency is not None else "—"
        flagged = "YES ⚠" if e.get("flagged_for_review") else "no"
        print(f"{trace:<38}  {ts:<20}  {model:<30}  {latency_str:>8}  {flagged}")


def print_flagged(entries):
    flagged = [e for e in entries if e.get("flagged_for_review")]
    if not flagged:
        print("No flagged entries.")
        return
    print(f"{len(flagged)} flagged entry/entries:\n")
    for e in flagged:
        trace = e.get("trace_id", "")
        ts = (e.get("timestamp_utc") or "")[:19]
        reasons = e.get("flag_reasons") or []
        print(f"  trace_id  : {trace}")
        print(f"  timestamp : {ts}")
        print(f"  reasons   : {'; '.join(reasons)}")
        print(f"  scores    : {_score_str(e)}")
        print()


def print_summary(entries):
    if not entries:
        print("No entries.")
        return

    total = len(entries)
    flagged = sum(1 for e in entries if e.get("flagged_for_review"))
    total_cost = sum(e.get("estimated_cost_usd") or 0.0 for e in entries)

    def avg(vals):
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None

    def fmt(v):
        return f"{v:.3f}" if v is not None else "—"

    latencies = [e.get("total_latency_ms") for e in entries]
    faith = [e.get("faithfulness_score") for e in entries]
    relevance = [e.get("answer_relevance_score") for e in entries]
    halluc = [e.get("hallucination_score") for e in entries]

    model_counts = defaultdict(int)
    for e in entries:
        m = e.get("model_id") or "unknown"
        model_counts[m] += 1

    print(f"{'='*50}")
    print(f"  Total runs         : {total}")
    print(f"  Flagged            : {flagged} ({100*flagged/total:.1f}%)")
    print(f"  Avg latency        : {fmt(avg(latencies))}ms")
    print(f"  Avg faithfulness   : {fmt(avg(faith))}")
    print(f"  Avg relevance      : {fmt(avg(relevance))}")
    print(f"  Avg hallucination  : {fmt(avg(halluc))}")
    print(f"  Total est. cost    : ${total_cost:.4f} USD")
    print(f"\n  Model usage:")
    for model, count in sorted(model_counts.items(), key=lambda x: -x[1]):
        print(f"    {model:<35} {count} run(s)")
    print(f"{'='*50}")


def main():
    args = parse_args()
    entries = load_entries(args.date)

    print(f"Log: governance_{args.date}.jsonl  ({len(entries)} entries)\n")

    if args.summary:
        print_summary(entries)
    elif args.flagged_only:
        print_flagged(entries)
    else:
        print_table(entries)


if __name__ == "__main__":
    main()
