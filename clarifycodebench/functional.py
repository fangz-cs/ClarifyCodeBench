"""Prepare run outputs for pass@1 evaluation with LiveCodeBench.

Functional correctness (pass@1) is measured by executing the model's final code
against the hidden tests of the *original* LiveCodeBench problem (keyed by
``question_id``). ClarifyCodeBench does **not** re-implement or redistribute the
LiveCodeBench test suite; instead this module extracts the generated code into
the input format expected by the LiveCodeBench code-generation evaluator.

Workflow
--------
1. Run the interactive protocol -> run-output JSONL (see :mod:`clarifycodebench.interact`).
2. ``build_code_file`` -> ``code.jsonl``: a list of ``{"question_id", "code_list"}``.
3. Feed ``code.jsonl`` (plus the LiveCodeBench problems for the same
   ``question_id``s) to the LiveCodeBench evaluator to obtain pass@1.

See the project README for how to obtain LiveCodeBench and run its evaluator.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .extract import extract_code


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _final_code(record: dict[str, Any]) -> str:
    """Best-effort extraction of final code from a run record (any schema)."""
    if record.get("final_code"):
        return record["final_code"]
    contents = record.get("response_contents") or []
    if contents:
        return extract_code(contents[-1])
    msgs = record.get("messages") or record.get("response_history") or []
    for m in reversed(msgs):
        if m.get("role") == "assistant":
            return extract_code(m.get("content", ""))
    return ""


def build_code_file(run_path: str | Path, out_path: str | Path) -> int:
    """Write a LiveCodeBench-style ``code.jsonl`` from a run-output JSONL.

    Output is a single JSON array of ``{"question_id", "code_list": [code]}``
    entries, matching the LiveCodeBench code-generation evaluator's input.
    Returns the number of entries written.
    """
    run_path, out_path = Path(run_path), Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    entries = []
    for rec in _read_jsonl(run_path):
        qid = rec.get("id") or rec.get("question_id")
        entries.append({"question_id": qid, "code_list": [_final_code(rec)]})

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False)
    return len(entries)
