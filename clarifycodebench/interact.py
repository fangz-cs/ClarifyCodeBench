"""Interactive evaluation protocol for ClarifyCodeBench.

For each task the evaluated model sees the *ambiguous* requirement and then, at
each turn, must either ask exactly one clarification question or emit final
code. When it asks a question, an LLM-as-judge matches it against the annotated
key questions; on a match the model receives the ground-truth answer, otherwise
a "not relevant" reply. The loop runs until the model emits code or reaches
``max_turns``.

The per-run output record (one JSON object per task, JSONL) is:

    {
      "id": <question_id>,            # LiveCodeBench problem id (for pass@1)
      "task_id": <task_id>,           # ClarifyCodeBench task id
      "key_question_num": <int>,      # K = number of annotated key questions
      "prompt": <ambiguous requirement>,
      "messages": [{"role","content"}, ...],   # full dialogue
      "response_contents": [<assistant content per turn>, ...],
      "hit_dict": {"0": 1, "1": 0, ...},        # per-turn key-question hit
      "hit_questions": [<matched key question>, ...],
      "turns": <int>,                 # number of model responses (incl. code turn)
      "final_code": <extracted code>,
      "finished": <bool>              # False if a turn failed after all retries
    }

This record feeds both metric computation (:mod:`clarifycodebench.metrics`) and
pass@1 preparation (:mod:`clarifycodebench.functional`).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

from openai import OpenAI

from .extract import extract_code, extract_question, is_code
from .llm import generate, judge_match
from .prompts import JUDGE_PROMPT, SYSTEM_PROMPT

FOLLOWUP = "\nBased on our current conversation, please ask a clarifying question or generate the code."
NOT_RELEVANT = "This question is not relevant to the current task."
ALREADY_ASKED = "This question has already been answered."


def _judge_answer(
    client: OpenAI,
    model_question: str,
    qa_pairs: list[dict[str, str]],
    hit_questions: set[str],
    judge_model: str,
    consistency: int,
    max_retries: int,
) -> tuple[int, str, str | None]:
    """Judge one model question against every key question.

    Returns ``(hit, reply, matched_key_question)`` where ``hit`` is 1 iff the
    question matches a *previously uncovered* key question. ``matched_key_question``
    is set only on a fresh hit; ``reply`` is the text returned to the model.
    Returns ``hit == -1`` to signal an unrecoverable judge failure.
    """
    for qa in qa_pairs:
        key_q = qa["question"]
        prompt = JUDGE_PROMPT.format(key_question=key_q, model_question=model_question)
        verdict = judge_match(client, prompt, judge_model, consistency, max_retries)
        if verdict is None:
            return -1, "", None
        if verdict:
            if key_q in hit_questions:
                return 0, ALREADY_ASKED, None
            return 1, qa["answer"], key_q
    return 0, NOT_RELEVANT, None


def run_task(
    client: OpenAI,
    model: str,
    task: dict[str, Any],
    *,
    judge_model: str = "gpt-4o",
    max_turns: int = 6,
    consistency: int = 3,
    max_tokens: int = 8192,
    thinking: bool = False,
    max_retries: int = 10,
) -> dict[str, Any] | None:
    """Run the interactive protocol on one task and return its result record.

    Returns ``None`` only if the very first generation fails.
    """
    qa_pairs = task.get("qa_pairs", []) or []
    question = task["modified_content"]

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    response_contents: list[str] = []
    hit_questions: set[str] = set()
    hit_dict: dict[str, int] = {}
    finished = True

    for turn in range(max_turns):
        response = generate(client, model, messages, max_tokens, thinking, max_retries)
        if response is None:
            finished = False
            break
        content = response.choices[0].message.content or ""
        messages.append({"role": "assistant", "content": content})
        response_contents.append(content)

        if is_code(content):
            break

        model_question = extract_question(content) or content
        hit, reply, matched = _judge_answer(
            client, model_question, qa_pairs, hit_questions, judge_model, consistency, max_retries
        )
        if hit == -1:  # judge failed after all retries
            finished = False
            break
        hit_dict[str(turn)] = hit
        if matched is not None:
            hit_questions.add(matched)
        messages.append({"role": "user", "content": reply + FOLLOWUP})

    final_content = response_contents[-1] if response_contents else ""
    return {
        "id": task.get("question_id"),
        "task_id": task.get("task_id"),
        "key_question_num": len(qa_pairs),
        "prompt": question,
        "messages": messages,
        "response_contents": response_contents,
        "hit_dict": hit_dict,
        "hit_questions": sorted(hit_questions),
        "turns": len(response_contents),
        "final_code": extract_code(final_content),
        "finished": finished,
    }


def load_tasks(path: str | Path) -> list[dict[str, Any]]:
    """Load the benchmark JSONL into a list of task dicts."""
    tasks = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    return tasks


def _completed_task_ids(save_path: Path) -> set[str]:
    done: set[str] = set()
    if save_path.exists():
        with save_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    done.add(json.loads(line).get("task_id"))
    return done


def run_dataset(
    client: OpenAI,
    model: str,
    tasks: Iterable[dict[str, Any]],
    save_path: str | Path,
    *,
    resume: bool = True,
    **run_kwargs: Any,
) -> None:
    """Run the protocol over a dataset, appending one JSON record per task.

    Results are streamed to ``save_path`` (JSONL). With ``resume=True`` any
    ``task_id`` already present in ``save_path`` is skipped, so interrupted runs
    can be continued safely.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    done = _completed_task_ids(save_path) if resume else set()

    for task in tasks:
        tid = task.get("task_id")
        if tid in done:
            print(f"skip {tid} (already done)", flush=True)
            continue
        print(f"task {tid} ({task.get('question_id')})", flush=True)
        record = run_task(client, model, task, **run_kwargs)
        if record is None or not record["finished"]:
            print(f"  unfinished: {tid}", flush=True)
            if record is None:
                continue
        with save_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
