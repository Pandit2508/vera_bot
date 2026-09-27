"""
llm.py
======
Thin LLM client. Tries providers in this order, based on whichever API key is
set in the environment: Anthropic -> Groq -> (deterministic fallback if
neither is configured). This lets the rest of the pipeline (validation, HTTP
server, submission generation) run and be tested without any credentials, and
upgrades automatically to real LLM-composed messages the moment a key is set,
with zero other code changes.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.request
import urllib.error
from typing import Optional

ANTHROPIC_MODEL = os.environ.get("VERA_MODEL", "claude-sonnet-4-6")
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"

# Groq's API is OpenAI-compatible. As of writing, llama-3.3-70b-versatile and
# llama-3.1-8b-instant have moved to Enterprise-only access on Groq, so
# openai/gpt-oss-120b is the strongest model actually available on free/
# developer-tier Groq accounts for this kind of structured-JSON composition
# task. Override with GROQ_MODEL if Groq's lineup changes again by the time
# you run this -- check https://console.groq.com/docs/models for the current
# list under "Production Models".
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

TIMEOUT_S = 25


class LLMError(RuntimeError):
    pass


def _strip_json_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    return text


def call_anthropic(system: str, user: str, api_key: str, model: str = ANTHROPIC_MODEL) -> str:
    body = json.dumps({
        "model": model,
        "max_tokens": 600,
        "temperature": 0,  # submission requires determinism
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }).encode("utf-8")
    req = urllib.request.Request(
        ANTHROPIC_URL,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise LLMError(f"Anthropic API error {e.code}: {e.read().decode('utf-8', 'ignore')}") from e
    text_blocks = [b["text"] for b in data.get("content", []) if b.get("type") == "text"]
    return "".join(text_blocks)


def call_groq(system: str, user: str, api_key: str, model: str = GROQ_MODEL,
               max_retries: int = 3, max_wait_s: float = 15.0) -> str:
    # gpt-oss models are reasoning models: they spend part of the completion-token
    # budget "thinking" before writing the final JSON, so the budget needs real
    # headroom -- 600 was enough for a plain chat model but truncates gpt-oss mid-output.
    # response_format=json_object is intentionally NOT set here: combined with a
    # reasoning model it was producing empty/invalid "failed_generation" 400s; we
    # instead rely on the existing fence-stripping + regex salvage below, same as
    # the Anthropic path.
    body = json.dumps({
        "model": model,
        "temperature": 0,  # submission requires determinism
        "max_completion_tokens": 1500,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }).encode("utf-8")
    req = urllib.request.Request(
        GROQ_URL,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            # Groq's API sits behind Cloudflare, which sometimes flags urllib's
            # default (near-absent) User-Agent as bot traffic (Cloudflare error
            # 1010). A normal-looking User-Agent avoids that false positive.
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
        },
    )
    # Retry budget is caller-supplied: the offline submission generator can afford
    # to wait out Groq's rolling 60s TPM window (see generate_submission.py's
    # patient params), but a live /v1/tick or /v1/reply call has a 30s SLA
    # (challenge-testing-brief.md) and must fail fast to the deterministic
    # fallback instead of hanging.
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", "ignore")
            if e.code == 429 and attempt < max_retries - 1:
                # Free-tier Groq caps tokens/minute; the error message tells us how
                # long to wait, either as "try again in 14.1s" or "...in 300ms".
                wait_s = 8.0
                m = re.search(r"try again in ([\d.]+)(ms|s)\b", err_body)
                if m:
                    val, unit = float(m.group(1)), m.group(2)
                    wait_s = val / 1000.0 if unit == "ms" else val
                wait_s = min(wait_s, max_wait_s)
                time.sleep(wait_s + 0.5)  # small buffer past the suggested wait
                continue
            raise LLMError(f"Groq API error {e.code}: {err_body}") from e
    raise LLMError("Groq API: exhausted retries on rate limiting")


def compose_via_llm(system: str, user: str, patient: bool = False) -> Optional[dict]:
    """Returns parsed dict, or None if no API key is configured (caller should fall back).

    `patient=False` (default, used by the live bot.py server): fail fast — small
    retry budget so a single call can't blow the 30s /v1/tick or /v1/reply SLA.
    `patient=True` (used by generate_submission.py, which has no real-time SLA):
    a much larger retry budget so a transient rate-limit gets ridden out instead
    of silently degrading that test pair to the deterministic fallback.
    """
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    groq_key = os.environ.get("GROQ_API_KEY", "")

    if anthropic_key:
        raw = call_anthropic(system, user, anthropic_key)
    elif groq_key:
        if patient:
            raw = call_groq(system, user, groq_key, max_retries=6, max_wait_s=65.0)
        else:
            raw = call_groq(system, user, groq_key, max_retries=2, max_wait_s=10.0)
    else:
        return None

    raw = _strip_json_fences(raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # try to salvage the first {...} block
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise LLMError(f"Could not parse LLM JSON output: {raw[:300]}")
