"""Validate the LLM-as-judge against a lexical TF-IDF baseline.

This reproduces the threat-to-validity check for the judge: for every
clarification question the model asked, compare the judge's yes/no label
(stored in ``hit_dict``) against a cheap TF-IDF cosine-similarity heuristic
over the annotated key questions, and report agreement (accuracy, Cohen's
kappa, confusion matrix) plus a threshold sensitivity sweep.

Human validation in the paper reports Cohen's kappa 0.89 between the judge and
human labels; this module provides the automatic lexical cross-check.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import cohen_kappa_score

from .extract import extract_question


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _normalize(s: str) -> str:
    return " ".join(s.strip().split())


def load_key_questions(data_path: Path) -> dict[str, list[str]]:
    """Map task_id -> list of annotated key questions."""
    by_task: dict[str, list[str]] = {}
    for row in _read_jsonl(data_path):
        tid = row.get("task_id")
        if not tid:
            continue
        by_task[tid] = [
            qa["question"].strip()
            for qa in row.get("qa_pairs", []) or []
            if isinstance(qa.get("question"), str) and qa["question"].strip()
        ]
    return by_task


def _model_questions(record: dict[str, Any]) -> list[str]:
    """Extract, in order, the clarification questions the model asked."""
    turns = record.get("messages") or record.get("response_history") or []
    out: list[str] = []
    for m in turns:
        if m.get("role") != "assistant":
            continue
        q = extract_question(m.get("content", ""))
        if q:
            out.append(q)
    return out


@dataclass
class TurnRecord:
    task_id: str
    turn_idx: int
    gpt_label: int
    max_sim: float
    best_key_question: str


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


def _best_similarity(vecs, question: str, key_questions: list[str]) -> tuple[float, str]:
    word_vec, char_vec = vecs
    if not key_questions:
        return 0.0, ""
    texts = [question] + key_questions
    Xw = word_vec.transform(texts)
    Xc = char_vec.transform(texts)
    pqw, pqc = Xw[0].toarray().ravel(), Xc[0].toarray().ravel()
    best, best_q = -1.0, ""
    for i, kq in enumerate(key_questions, start=1):
        sim = max(_cosine(pqw, Xw[i].toarray().ravel()), _cosine(pqc, Xc[i].toarray().ravel()))
        if sim > best:
            best, best_q = sim, kq
    return max(0.0, best), best_q


def validate(
    pred_path: str | Path,
    data_path: str | Path,
    threshold: float = 0.40,
    threshold_grid: Iterable[float] = (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60),
) -> dict[str, Any]:
    """Compare judge labels vs TF-IDF labels; return a summary dict.

    ``pred_path`` is a run-output JSONL; ``data_path`` is the benchmark JSONL.
    """
    pred_path, data_path = Path(pred_path), Path(data_path)
    by_task = load_key_questions(data_path)

    # Fit TF-IDF on all model questions + key questions.
    corpus: list[str] = []
    for row in _read_jsonl(pred_path):
        corpus.extend(_normalize(q) for q in _model_questions(row))
        corpus.extend(_normalize(x) for x in by_task.get(row.get("task_id"), []))
    corpus = corpus or ["dummy"]
    word_vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1, max_df=0.95).fit(corpus)
    char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, max_df=0.98).fit(corpus)
    vecs = (word_vec, char_vec)

    records: list[TurnRecord] = []
    for row in _read_jsonl(pred_path):
        tid = row.get("task_id")
        if tid not in by_task:
            continue
        hit_dict = row.get("hit_dict", {}) or {}
        for t, q in enumerate(_model_questions(row)):
            gpt_label = 1 if int(hit_dict.get(str(t), 0)) == 1 else 0
            sim, best_kq = _best_similarity(vecs, _normalize(q), by_task[tid])
            records.append(TurnRecord(tid, t, gpt_label, sim, best_kq))

    if not records:
        raise ValueError("No clarification turns parsed; check the input schema.")

    y_gpt = np.array([r.gpt_label for r in records])
    y_ours = np.array([1 if r.max_sim >= threshold else 0 for r in records])
    tp = int(((y_gpt == 1) & (y_ours == 1)).sum())
    tn = int(((y_gpt == 0) & (y_ours == 0)).sum())
    fp = int(((y_gpt == 0) & (y_ours == 1)).sum())
    fn = int(((y_gpt == 1) & (y_ours == 0)).sum())

    grid = []
    for thr in threshold_grid:
        y = np.array([1 if r.max_sim >= thr else 0 for r in records])
        grid.append(
            {
                "threshold": float(thr),
                "accuracy": float((y_gpt == y).mean()),
                "kappa": float(cohen_kappa_score(y_gpt, y)),
                "ours_positive_rate": float(y.mean()),
            }
        )

    return {
        "pred_path": str(pred_path),
        "data_path": str(data_path),
        "n_turns": len(records),
        "threshold": threshold,
        "agreement_accuracy": float((y_gpt == y_ours).mean()),
        "cohen_kappa": float(cohen_kappa_score(y_gpt, y_ours)),
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "gpt_positive_rate": float(y_gpt.mean()),
        "threshold_sensitivity": grid,
    }
