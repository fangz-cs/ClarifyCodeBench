# ClarifyCodeBench

[![arXiv](https://img.shields.io/badge/arXiv-2607.00711-b31b1b.svg)](https://arxiv.org/abs/2607.00711)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](pyproject.toml)

**Evaluating LLMs on clarifying ambiguous requirements for code generation.**

📄 **Paper:** [arXiv:2607.00711](https://arxiv.org/abs/2607.00711)

Real-world coding requirements are frequently ambiguous, incomplete, or
underspecified. ClarifyCodeBench is an *interactive* benchmark that measures
whether an LLM can **detect** ambiguity, **ask** the right clarification
questions, and **use** the answers to generate correct code — rather than
silently guessing intent.

Each of the **419 tasks** provides an underspecified requirement, annotated key
clarification questions with ground-truth answers, a fine-grained ambiguity
type, and (via LiveCodeBench) an executable hidden-test suite for the final
code.

<p align="center"><em>Given an ambiguous requirement, the model either asks one
clarification question or emits code. A question is matched against the
annotated key questions by an LLM-as-judge; on a match the model receives the
ground-truth answer. The loop repeats until the model emits code.</em></p>

## Metrics

Beyond functional correctness (**pass@1**), ClarifyCodeBench introduces two
metrics for *interaction quality*:

- **TKQR** — Turn-discounted Key Question Rate. A normalized DCG over the
  per-turn hit sequence; rewards asking annotated key questions **early**.
  Range `[0, 1]`.
- **ORA** — Optimal Round Adherence. A Gaussian penalty on the number of
  clarification rounds `n` relative to the number of key questions `K`; highest
  when the model asks exactly the needed number of questions. Range `(0, 1]`.

Both are defined in [`clarifycodebench/metrics.py`](clarifycodebench/metrics.py).

## Install

```bash
git clone <your-repo-url> clarifycodebench && cd clarifycodebench
python -m venv .venv && source .venv/bin/activate
pip install -e .          # or: pip install -r requirements.txt
```

Set credentials for any OpenAI-compatible endpoint (official API, a proxy, or a
gateway that multiplexes GPT / Claude / Gemini / DeepSeek / Qwen):

```bash
cp .env.example .env      # then edit, or just export:
export CLARIFY_API_KEY=sk-...
export CLARIFY_BASE_URL=https://api.openai.com/v1
```

## Quickstart

**1. Run the interactive evaluation** for a model:

```bash
python scripts/run_eval.py \
    --data data/ClarifyCodeBench.jsonl \
    --model gpt-4o \
    --out results/gpt-4o.jsonl
# reasoning models: add --thinking --max-tokens 8192
```

This writes one JSON record per task (resumable — re-running skips finished
tasks). The judge defaults to `gpt-4o` (temperature 1.0, 3-sample majority
vote); override with `--judge-model`.

**2. Compute clarification metrics** (no API calls):

```bash
python scripts/compute_metrics.py results/gpt-4o.jsonl
# TKQR / ORA / avg turns / avg diff
```

Try it right now on the bundled example (no API key needed):

```bash
python scripts/compute_metrics.py examples/sample_results.jsonl
```

**3. Functional correctness (pass@1)** — extract code, then evaluate with
LiveCodeBench:

```bash
python scripts/prepare_pass1.py --run results/gpt-4o.jsonl --out pass1/gpt-4o/code.jsonl
```

`code.jsonl` is a list of `{"question_id", "code_list"}` in LiveCodeBench's
input format. Run it through the
[LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench) code-generation
evaluator against the hidden tests for the same `question_id`s to obtain pass@1.
The hidden tests are **not** redistributed here (they belong to LiveCodeBench).

**4. (Optional) Validate the LLM-as-judge** against a TF-IDF baseline:

```bash
python scripts/validate_judge.py --pred results/gpt-4o.jsonl --data data/ClarifyCodeBench.jsonl
# agreement accuracy, Cohen's kappa, confusion matrix, threshold sweep
```

## Repository layout

```
clarifycodebench/          # importable package
  prompts.py               # system / judge / full-req prompts
  llm.py                   # OpenAI-compatible client (env keys) + generate/judge
  extract.py               # parse [QUESTION] / [CODE] from responses
  interact.py              # the interactive evaluation protocol
  metrics.py               # TKQR, ORA, and aggregation over run outputs
  judge_validation.py      # judge vs TF-IDF agreement (Cohen's kappa)
  functional.py            # prepare code.jsonl for LiveCodeBench pass@1
  taxonomy.py              # ambiguity type names + definitions
scripts/                   # run_eval, compute_metrics, prepare_pass1, validate_judge
configs/models.yaml        # per-model API params used in the paper
data/ClarifyCodeBench.jsonl  # the 419-task benchmark (+ data card)
examples/sample_results.jsonl
```

## Run-output schema

`run_eval.py` writes JSONL; each record has `id` (LiveCodeBench
`question_id`), `task_id`, `key_question_num` (K), `prompt`, full `messages`,
`hit_dict` (per-turn key-question hit), `hit_questions`, `turns`, `final_code`,
and `finished`. This feeds both `compute_metrics.py` and `prepare_pass1.py`.

## Dataset construction

The benchmark is built from LiveCodeBench by **manual, deletion-only ambiguity
injection**: annotators remove a small amount of the information needed to
determine intended behavior, then write the key clarification questions,
ground-truth answers, and ambiguity types. Every task is human-annotated — no
LLM is used to construct the benchmark. See the [data card](data/README.md).

## Citation

```bibtex
@article{fang2026clarifycodebench,
  title         = {ClarifyCodeBench: Evaluating LLMs on Clarifying Ambiguous Requirements for Code Generation},
  author        = {Fang, Zheng and Jin, Dongming and Dong, Yihong and Li, Yongmin and Zhang, Kechi and Jin, Zhi and Li, Ge},
  year          = {2026},
  journal       = {arXiv preprint arXiv:2607.00711},
  eprint        = {2607.00711},
  archivePrefix = {arXiv},
  primaryClass  = {cs.SE},
  url           = {https://arxiv.org/abs/2607.00711}
}
```

## License

Code and annotations: **MIT** (see [LICENSE](LICENSE)). The underlying problems
and hidden tests are governed by the **LiveCodeBench** license; please cite and
comply with it when computing pass@1.
