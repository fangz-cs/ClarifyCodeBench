#!/usr/bin/env python3
"""Compute TKQR / ORA / turns / Diff from ClarifyCodeBench run outputs.

Example:
    python scripts/compute_metrics.py results/gpt-4o.jsonl
    python scripts/compute_metrics.py results/            # aggregate a directory
    python scripts/compute_metrics.py results/*.jsonl --json

TKQR and ORA are averaged over all task records. ``avg_diff`` is the mean gap
between clarification rounds asked and the number of key questions (negative =
under-asking). No API calls are made; this reads only the run files.
"""

from __future__ import annotations

import argparse
import json

from clarifycodebench.metrics import aggregate


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="run JSONL file(s) or directory(ies)")
    ap.add_argument("--json", action="store_true", help="emit a JSON object instead of a table")
    args = ap.parse_args()

    result = aggregate(args.paths)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    print(f"tasks scored : {result['num_tasks']} (from {result['num_files']} file(s))")
    print(f"TKQR         : {result['TKQR']:.4f}")
    print(f"ORA          : {result['ORA']:.4f}")
    print(f"avg turns    : {result['avg_turns']:.4f}")
    print(f"avg diff     : {result['avg_diff']:+.4f}")


if __name__ == "__main__":
    main()
