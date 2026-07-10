"""Ambiguity taxonomy for ClarifyCodeBench.

Every annotated ambiguity point has exactly one type. A task's ``keywords``
field is the *deduplicated set* of the types of its points, stored as the
English type names used in the paper, so ``len(keywords) <= len(qa_pairs)``:
a task whose two ambiguity points share a type carries a single keyword.
This module documents that taxonomy with a short definition per type and
provides small helpers to normalize and validate labels.

The taxonomy is grounded in requirement-quality principles (clarity,
completeness, verifiability) and instantiated for code-generation tasks.
"""

from __future__ import annotations

# Short, human-readable definitions per type (mirrors Table 2 of the paper).
TYPE_DEFINITIONS: dict[str, str] = {
    "Terminology": "A domain term, action, or state is undefined, overloaded, or open to multiple interpretations.",
    "Behavior": "The required function, objective, or side effect is underspecified, so the intended behavior is not uniquely determined.",
    "Edge Cases": "Boundary or exceptional conditions are not specified, leaving behavior unclear for special inputs.",
    "Indices & Ranges": "Index bases, interval boundaries, or inclusion rules are underspecified.",
    "Ordering & Atomicity": "Temporal order, simultaneity, or indivisible execution assumptions are unclear.",
    "Output Format": "The required output structure, layout, or presentation rule is missing or unclear.",
    "Comparison Rules": "The comparison key, tie-breaking rule, or stability requirement is not specified.",
    "Units": "A quantity is specified without a clear unit, scale, prefix, or dimensional convention.",
    "Collection Semantics": "A collection or state object is mentioned, but its membership, update rule, or access semantics are underspecified.",
    "Numerical Precision": "Precision, rounding, tolerance, or error-handling requirements are unclear.",
    "String & Localization": "String encoding, casing, or localization conventions are left unspecified.",
}

# The canonical type names, in the order used throughout the paper.
TYPES: list[str] = list(TYPE_DEFINITIONS)


def normalize(keyword: str) -> str:
    """Normalize an ambiguity keyword (trims surrounding whitespace)."""
    return keyword.strip()


def normalize_types(keywords: list[str]) -> list[str]:
    """Normalize a list of keywords (order preserved)."""
    return [normalize(k) for k in keywords]


def is_known(keyword: str) -> bool:
    """Return True if ``keyword`` is one of the documented taxonomy types."""
    return normalize(keyword) in TYPE_DEFINITIONS
