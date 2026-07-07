#!/usr/bin/env python3
"""Extract final code from a run output into LiveCodeBench ``code.jsonl`` format.

pass@1 is computed by the external LiveCodeBench evaluator against the hidden
tests of each problem (keyed by ``question_id``). This script produces the
``code.jsonl`` the evaluator expects; see the README for how to run it.

Example:
    python scripts/prepare_pass1.py --run results/gpt-4o.jsonl --out pass1/gpt-4o/code.jsonl
"""

from __future__ import annotations

import argparse

from clarifycodebench.functional import build_code_file


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run-output JSONL")
    ap.add_argument("--out", required=True, help="output code.jsonl (LiveCodeBench format)")
    args = ap.parse_args()

    n = build_code_file(args.run, args.out)
    print(f"wrote {n} code entries -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
