# ClarifyCodeBench — Dataset Card

`ClarifyCodeBench.jsonl` contains **419** interactive code-generation tasks with
*intentionally underspecified* requirements, plus annotated key clarification
questions, ground-truth answers, and fine-grained underspecification types.

The tasks are built on top of [LiveCodeBench](https://livecodebench.github.io/)
(v6) by **manual deletion-only editing**: annotators remove a small amount of the
information needed to determine the intended behavior, keeping the task natural
and the underlying solution unchanged. Every task is human-annotated (no LLM is
used in construction). Each edited task is underspecified initially but becomes
fully specified once the annotated clarification answers are given.

## Schema (one JSON object per line)

| Field | Type | Description |
|-------|------|-------------|
| `task_id` | str | ClarifyCodeBench task id, e.g. `task_1`. |
| `question_id` | str | Original **LiveCodeBench** problem id, e.g. `abc390_b`. Use this to recover the full statement and hidden tests for pass@1. |
| `modified_content` | str | The **underspecified** requirement shown to the model. |
| `qa_pairs` | list | Annotated key clarification questions and their ground-truth answers. `len(qa_pairs)` = **K**, the number of underspecification points. |
| `qa_pairs[].question` | str | A key clarification question the model should ask. |
| `qa_pairs[].answer` | str | Ground-truth answer, taken from the deleted information. |
| `keywords` | list[str] | Underspecification type label(s) in English (see the taxonomy below). |

### Example

```json
{
  "task_id": "task_1",
  "question_id": "abc390_b",
  "keywords": ["Terminology"],
  "modified_content": "You are given a length-N sequence A=(A_1,...,A_N).\nDetermine whether A is a geometric progression. ...",
  "qa_pairs": [
    {
      "question": "What kinds of numbers can the sequence contain? Are zero or negative numbers allowed?",
      "answer": "All elements of the sequence are positive integers"
    }
  ]
}
```

> Note: the **full** (fully specified) problem statement and the **hidden test
> cases** are **not** redistributed here — they belong to LiveCodeBench. Fetch
> them from LiveCodeBench keyed by `question_id` (see the top-level README).

## Statistics

- **419** tasks, by number of underspecification points **K**:
  - K=1: **199** · K=2: **169** · K=3: **51**  (671 underspecification points total)

Underspecification-type distribution (over all annotated points):

| Type | Count | Share |
|------|------:|------:|
| Output Format | 185 | 27.6% |
| Terminology | 163 | 24.3% |
| Edge Cases | 81 | 12.1% |
| Behavior | 76 | 11.3% |
| Indices & Ranges | 56 | 8.3% |
| Collection Semantics | 48 | 7.2% |
| Comparison Rules | 27 | 4.0% |
| Ordering & Atomicity | 17 | 2.5% |
| Units | 9 | 1.3% |
| Numerical Precision | 5 | 0.7% |
| String & Localization | 4 | 0.6% |

## Underspecification taxonomy

Types are grounded in requirement-quality principles (clarity, completeness,
verifiability) and instantiated for code generation. See
`clarifycodebench/taxonomy.py` for per-type definitions.

## License

The **annotations** (underspecified rewrites, key questions, answers, types) are
released under the repository's MIT license. The underlying problems and hidden
tests are governed by **LiveCodeBench's** license — please cite and comply with
it when using pass@1.
