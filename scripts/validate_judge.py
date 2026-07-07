#!/usr/bin/env python3
"""Validate the LLM-as-judge against a TF-IDF lexical baseline.

Reports agreement accuracy, Cohen's kappa, a confusion matrix, and a threshold
sensitivity sweep between the judge's yes/no labels and a TF-IDF cosine
heuristic over the annotated key questions.

Example:
    python scripts/validate_judge.py \\
        --pred results/gpt-4o.jsonl \\
        --data data/ClarifyCodeBench.jsonl \\
        --out results/judge_eval/summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from clarifycodebench.judge_validation import validate


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True, help="run-output JSONL")
    ap.add_argument("--data", default="data/ClarifyCodeBench.jsonl", help="benchmark JSONL")
    ap.add_argument("--threshold", type=float, default=0.40, help="TF-IDF cosine threshold")
    ap.add_argument("--out", default=None, help="optional path to write the summary JSON")
    args = ap.parse_args()

    summary = validate(args.pred, args.data, threshold=args.threshold)
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
        print(f"\nwritten -> {out}", flush=True)


if __name__ == "__main__":
    main()
