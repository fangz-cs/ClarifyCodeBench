"""Parsing helpers for model responses.

The evaluated model must answer in one of two tagged formats:

    [QUESTION] ... [/QUESTION]      (ask one clarification question)
    [CODE] ... [/CODE]             (emit final code)

``extract_question`` and ``extract_code`` recover the payload of each. The code
extractor also tolerates Markdown-fenced ```python blocks as a fallback, since
models occasionally ignore the tag format.
"""

from __future__ import annotations

import re

_QUESTION_RE = re.compile(r"\[QUESTION\](.*?)(?:\[/QUESTION\]|$)", re.DOTALL)
_CODE_TAG_RE = re.compile(r"\[CODE\](.*?)(?:\[/CODE\]|$)", re.DOTALL)
_CODE_FENCE_RE = re.compile(
    r"```[ \t]*(?P<lang>[A-Za-z0-9_+-]*)[ \t]*\n(?P<code>.*?)(?:\n```|```|$)",
    re.DOTALL,
)


def is_code(text: str) -> bool:
    """Whether a response should be treated as final code (contains ``[CODE]``)."""
    return "[CODE]" in (text or "")


def extract_question(text: str) -> str | None:
    """Return the clarification question inside ``[QUESTION]...[/QUESTION]``.

    Returns ``None`` when no ``[QUESTION]`` tag is present.
    """
    if not text:
        return None
    m = _QUESTION_RE.search(text)
    return m.group(1).strip() if m else None


def extract_code(text: str) -> str:
    """Extract the first Python code block from ``text``.

    Priority: ``[CODE]...[/CODE]`` tag, then the first ```python (or unlabelled)
    fenced block, whichever appears earlier. If neither is present the whole
    text is treated as code. A trailing ``if __name__ == "__main__"`` guard is
    replaced by a bare ``main()`` call so the extracted snippet runs under the
    stdin/stdout harness. Returns ``""`` only for empty input.
    """
    if not text:
        return ""

    m_tag = _CODE_TAG_RE.search(text)
    m_fence = None
    for m in _CODE_FENCE_RE.finditer(text):
        lang = (m.group("lang") or "").strip().lower()
        if lang in ("", "python", "py"):
            m_fence = m
            break

    if not m_tag and not m_fence:
        code = text
    elif m_tag and (not m_fence or m_tag.start() <= m_fence.start()):
        code = m_tag.group(1).strip()
    else:
        code = (m_fence.group("code") or "").strip()

    lines: list[str] = []
    for line in code.splitlines():
        if "if __name__" in line:
            lines.append("main()")
            break
        lines.append(line)
    return "\n".join(lines).strip()
