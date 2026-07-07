"""Clarification-quality metrics: TKQR and ORA.

This module contains the two metrics introduced in the paper plus the
aggregation logic that turns a directory of run outputs into the reported
averages (TKQR, ORA, average interaction turns, and Diff).

Definitions
-----------
TKQR (Turn-discounted Key Question Rate): a normalized DCG over the per-turn
hit sequence. It rewards asking annotated key questions *early*.

    DCG_n  = sum_{i=1..n} h_i / log2(i + 1)
    IDCG_n = sum_{i=1..min(n,K)} 1 / log2(i + 1)
    TKQR   = DCG_n / IDCG_n            (in [0, 1])

ORA (Optimal Round Adherence): a Gaussian penalty on the number of
question-asking rounds ``n`` relative to the number of key questions ``K``.

    ORA(n, K, sigma) = exp( -(n - K)^2 / (2 * sigma^2) )   (in (0, 1])

with ``sigma`` chosen so ORA = 0.5 when ``|n - K| = 0.5*K``, i.e.
``sigma = 0.5*K / sqrt(2 ln 2) ≈ 0.425*K`` (and a small constant when K = 0).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable


# --------------------------------------------------------------------------- #
# Core metric functions
# --------------------------------------------------------------------------- #
def calculate_tkqr(hit_sequence: list[int], total_key_questions: int) -> float:
    """Turn-discounted Key Question Rate for a single task.

    ``hit_sequence[i] == 1`` iff the model asked a *previously uncovered* key
    question at turn ``i`` (repeated questions score 0). Returns a value in
    ``[0, 1]``; 0.0 if no key question is hit or ``total_key_questions <= 0``.
    """
    n = len(hit_sequence)
    if n == 0 or total_key_questions <= 0:
        return 0.0

    dcg = sum(hit / math.log2(i + 2) for i, hit in enumerate(hit_sequence))
    ideal_len = min(n, total_key_questions)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_len))
    return dcg / idcg if idcg else 0.0


def ora_sigma(K: int) -> float:
    """Default Gaussian scale for ORA.

    The paper derives ``sigma = 0.5*K / sqrt(2 ln 2) ≈ 0.425*K`` (so that
    ORA = 0.5 when ``|n - K| = 0.5*K``). We use the ``0.425*K`` constant exactly,
    which is what produced the reported results; ``0.3`` is used when ``K == 0``.
    """
    if K <= 0:
        return 0.3
    return 0.425 * K


def calculate_ora(n: int, K: int, sigma: float | None = None) -> float:
    """Optimal Round Adherence for a single task.

    ``n`` is the number of rounds in which the model asked a question; ``K`` is
    the number of key questions. Highest (1.0) when ``n == K``. ``sigma``
    defaults to :func:`ora_sigma`.
    """
    if n < 0 or K < 0:
        raise ValueError("n and K must be non-negative")
    if sigma is None:
        sigma = ora_sigma(K)
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    return math.exp(-((n - K) ** 2) / (2.0 * sigma**2))


# --------------------------------------------------------------------------- #
# Per-run helpers
# --------------------------------------------------------------------------- #
def hit_sequence_from_record(record: dict[str, Any]) -> list[int]:
    """Reconstruct the padded hit sequence for one run record.

    ``hit_dict`` maps a turn index (as a string) to 0/1 for each turn in which
    the model asked a question. The sequence is read in turn order and padded
    with zeros up to the total number of model responses (``turns``), so the
    final code-generation turn contributes a 0.
    """
    hit_dict = record.get("hit_dict", {}) or {}
    turns = _num_responses(record)
    seq = [int(hit_dict[str(i)]) for i in range(len(hit_dict)) if str(i) in hit_dict]
    if turns > len(seq):
        seq = seq + [0] * (turns - len(seq))
    return seq


def _num_responses(record: dict[str, Any]) -> int:
    """Total number of model responses (question turns + the final code turn)."""
    if "turns" in record and record["turns"] is not None:
        return int(record["turns"])
    if "response_contents" in record and record["response_contents"] is not None:
        return len(record["response_contents"])
    if "messages" in record:
        return sum(1 for m in record["messages"] if m.get("role") == "assistant")
    return len(record.get("hit_dict", {}) or {})


def key_question_num(record: dict[str, Any]) -> int:
    """Number of annotated key questions K for this task."""
    if record.get("key_question_num") is not None:
        return int(record["key_question_num"])
    return len(record.get("qa_pairs", []) or [])


def score_record(record: dict[str, Any]) -> dict[str, float]:
    """Compute TKQR / ORA / turns / diff for a single run record.

    Reproduces the reference aggregation:

    - ``turns_total``     = number of model responses (incl. final code turn)
    - ``n_questions``     = turns_total - 1  (clarification rounds asked)
    - ``K``               = number of key questions
    - TKQR uses the padded hit sequence; the special case (K == 0 and the model
      coded immediately in a single turn) scores 1.0.
    - ORA and Diff use ``n_questions`` vs ``K``.
    """
    K = key_question_num(record)
    turns_total = _num_responses(record)
    hit_seq = hit_sequence_from_record(record)

    tkqr = calculate_tkqr(hit_seq, K)
    if K == 0 and turns_total == 1:
        tkqr = 1.0

    n_questions = turns_total - 1
    ora = calculate_ora(n_questions, K)
    return {
        "tkqr": tkqr,
        "ora": ora,
        "turns": float(n_questions),
        "diff": float(n_questions - K),
    }


# --------------------------------------------------------------------------- #
# Aggregation over a set of run files
# --------------------------------------------------------------------------- #
def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def aggregate(paths: Iterable[str | Path]) -> dict[str, Any]:
    """Average TKQR / ORA / turns / diff over every record in ``paths``.

    ``paths`` may be individual ``.jsonl`` files or directories (searched
    recursively for ``.jsonl``). Records missing ``key_question_num`` fall back
    to ``len(qa_pairs)`` if present.
    """
    files: list[Path] = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            files.extend(sorted(f for f in p.rglob("*.jsonl") if f.is_file()))
        elif p.is_file():
            files.append(p)

    n = 0
    tot = {"tkqr": 0.0, "ora": 0.0, "turns": 0.0, "diff": 0.0}
    for f in files:
        for rec in _iter_jsonl(f):
            s = score_record(rec)
            for k in tot:
                tot[k] += s[k]
            n += 1

    if n == 0:
        raise ValueError(f"No run records found under: {list(map(str, paths))}")

    return {
        "num_tasks": n,
        "num_files": len(files),
        "TKQR": tot["tkqr"] / n,
        "ORA": tot["ora"] / n,
        "avg_turns": tot["turns"] / n,
        "avg_diff": tot["diff"] / n,
    }
