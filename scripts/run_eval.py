#!/usr/bin/env python3
"""Run the ClarifyCodeBench interactive evaluation for one model.

Example:
    export CLARIFY_API_KEY=sk-...           # your endpoint key
    export CLARIFY_BASE_URL=https://api.openai.com/v1
    python scripts/run_eval.py \\
        --data data/ClarifyCodeBench.jsonl \\
        --model gpt-4o \\
        --out results/gpt-4o.jsonl

Reasoning ("thinking") models: add --thinking and raise --max-tokens (e.g. 8192).
The judge model defaults to gpt-4o; override with --judge-model.
"""

from __future__ import annotations

import argparse

from clarifycodebench.interact import load_tasks, run_dataset
from clarifycodebench.llm import make_client


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/ClarifyCodeBench.jsonl", help="benchmark JSONL")
    ap.add_argument("--model", required=True, help="evaluated model id (as your endpoint names it)")
    ap.add_argument("--out", required=True, help="output run JSONL")
    ap.add_argument("--judge-model", default="gpt-4o", help="LLM-as-judge model id")
    ap.add_argument("--max-turns", type=int, default=6, help="max interaction rounds")
    ap.add_argument("--consistency", type=int, default=3, help="judge samples for majority vote")
    ap.add_argument("--max-tokens", type=int, default=1024, help="generation token budget")
    ap.add_argument("--thinking", action="store_true", help="enable reasoning mode")
    ap.add_argument("--max-retries", type=int, default=10, help="API retries per call")
    ap.add_argument("--no-resume", action="store_true", help="do not skip already-completed task_ids")
    ap.add_argument("--api-key", default=None, help="override CLARIFY_API_KEY")
    ap.add_argument("--base-url", default=None, help="override CLARIFY_BASE_URL")
    ap.add_argument("--limit", type=int, default=None, help="only run the first N tasks (debug)")
    args = ap.parse_args()

    client = make_client(api_key=args.api_key, base_url=args.base_url)
    tasks = load_tasks(args.data)
    if args.limit is not None:
        tasks = tasks[: args.limit]

    run_dataset(
        client,
        args.model,
        tasks,
        args.out,
        resume=not args.no_resume,
        judge_model=args.judge_model,
        max_turns=args.max_turns,
        consistency=args.consistency,
        max_tokens=args.max_tokens,
        thinking=args.thinking,
        max_retries=args.max_retries,
    )
    print(f"done -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
