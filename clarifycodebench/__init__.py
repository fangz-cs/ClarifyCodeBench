"""ClarifyCodeBench: evaluating LLMs on clarifying underspecified code requirements.

Only lightweight, dependency-free modules are imported eagerly so that offline
analysis (metric computation, response parsing, taxonomy lookup) works without
the LLM SDK installed. Modules that need extra dependencies are imported
explicitly by the caller:

    from clarifycodebench.llm import make_client            # needs `openai`
    from clarifycodebench.interact import run_dataset       # needs `openai`
    from clarifycodebench.judge_validation import validate  # needs `scikit-learn`
    from clarifycodebench.functional import build_code_file
"""

from . import extract, metrics, taxonomy
from .metrics import aggregate, calculate_ora, calculate_tkqr, score_record

__version__ = "0.1.0"

__all__ = [
    "extract",
    "metrics",
    "taxonomy",
    "aggregate",
    "calculate_ora",
    "calculate_tkqr",
    "score_record",
]
