"""Thin OpenAI-compatible LLM client used for evaluation and judging.

All providers in the paper are reached through an OpenAI-compatible endpoint
(the ``openai`` SDK). Credentials and the base URL come from **environment
variables** — never hard-code keys:

    CLARIFY_API_KEY     API key for the evaluated / judge model endpoint
    CLARIFY_BASE_URL    Base URL of the OpenAI-compatible endpoint
                        (e.g. https://api.openai.com/v1, or a proxy/gateway)

You can point these at any gateway that multiplexes the different model
families (GPT, Claude, Gemini, DeepSeek, Qwen), which is how the original
experiments were run.
"""

from __future__ import annotations

import os
import time
from typing import Any

try:
    import httpx
except ImportError:  # httpx is an optional transport dependency
    httpx = None

from openai import OpenAI


def make_client(
    api_key: str | None = None,
    base_url: str | None = None,
    trust_env: bool = False,
) -> OpenAI:
    """Build an OpenAI-compatible client from args or environment variables.

    Raises ``RuntimeError`` if no API key is available, so misconfiguration
    fails loudly instead of sending an empty credential.
    """
    api_key = api_key or os.environ.get("CLARIFY_API_KEY")
    base_url = base_url or os.environ.get("CLARIFY_BASE_URL")
    if not api_key:
        raise RuntimeError(
            "No API key. Set CLARIFY_API_KEY (and CLARIFY_BASE_URL) or pass api_key=."
        )
    kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    if httpx is not None:
        # trust_env=False avoids picking up ambient HTTP(S)_PROXY settings.
        kwargs["http_client"] = httpx.Client(trust_env=trust_env)
    return OpenAI(**kwargs)


def _completion_kwargs(model: str, thinking: bool, max_tokens: int) -> dict[str, Any]:
    """Per-family generation kwargs (token-limit / thinking-toggle differences).

    Families are matched by substring on the model id, mirroring the endpoints
    used in the paper. Unknown models fall back to a plain request.
    """
    kw: dict[str, Any] = {"temperature": 0.0, "stream": False}
    m = model.lower()
    if "gpt" in m:
        # GPT-5-style reasoning effort; "minimal" for non-thinking runs.
        kw["max_completion_tokens"] = max_tokens
        kw["reasoning_effort"] = "low" if thinking else "minimal"
    elif "qwen" in m:
        kw["max_tokens"] = max_tokens
        kw["extra_body"] = {"enable_thinking": bool(thinking)}
    else:
        # deepseek / claude / gemini and generic OpenAI-compatible models.
        kw["max_tokens"] = max_tokens
    return kw


def generate(
    client: OpenAI,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int = 8192,
    thinking: bool = False,
    max_retries: int = 10,
) -> Any:
    """Call the chat completion endpoint with exponential-backoff retries.

    Returns the raw response object, or ``None`` if all retries fail.
    """
    kwargs = _completion_kwargs(model, thinking, max_tokens)
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(model=model, messages=messages, **kwargs)
        except Exception as exc:  # noqa: BLE001 - surface then back off
            print(f"[generate] attempt {attempt + 1}/{max_retries} failed: {exc}", flush=True)
            time.sleep(min(2**attempt, 30))
    return None


def judge_match(
    client: OpenAI,
    prompt: str,
    judge_model: str = "gpt-4o",
    consistency: int = 3,
    max_retries: int = 10,
) -> bool | None:
    """LLM-as-judge: does a model question match an annotated key question?

    Samples the judge ``consistency`` times (temperature 1.0) and returns the
    majority ``yes``/``no`` verdict as a bool. Returns ``None`` on total failure.
    """
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=judge_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=1.0,
                top_p=0.95,
                n=consistency,
                stream=False,
            )
            yes = no = 0
            for choice in resp.choices:
                verdict = (choice.message.content or "").strip().lower()
                if "yes" in verdict:
                    yes += 1
                elif "no" in verdict:
                    no += 1
            return yes > no
        except Exception as exc:  # noqa: BLE001
            print(f"[judge] attempt {attempt + 1}/{max_retries} failed: {exc}", flush=True)
            time.sleep(min(2**attempt, 30))
    return None
