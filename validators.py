"""
validators.py
==============
Post-composition checks against the challenge's own anti-pattern list
(challenge-brief.md §11). Returns a list of violation strings (empty = clean).
The composer uses this to decide whether to re-prompt once before giving up.
"""
from __future__ import annotations

import re

MULTI_CTA_PATTERNS = [
    re.compile(r"reply\s+\w+\s+for\s+.*(,|\bor\b).*(reply|for)\s", re.IGNORECASE),
]

PROMO_HYPE_WORDS = ["amazing deal", "huge discount", "unbelievable", "don't miss out!!!", "limited time only!!!"]

REQUIRED_KEYS = {"body", "cta", "send_as"}


def validate_output(
    action: dict,
    category: dict,
    merchant: dict,
    customer: dict | None,
) -> list[str]:
    problems: list[str] = []

    # 1. required fields present & non-empty
    present = {k for k in action if action.get(k) not in (None, "")}
    missing = REQUIRED_KEYS - present
    if missing:
        problems.append(f"missing/empty required fields: {sorted(missing)}")

    body = action.get("body", "") or ""
    if not body.strip():
        problems.append("empty body")
        return problems  # nothing else to check

    # 2. send_as correctness
    if customer is None and action.get("send_as") != "vera":
        problems.append("send_as should be 'vera' when there is no customer context")
    if customer is not None and action.get("send_as") not in ("vera", "merchant_on_behalf"):
        problems.append("send_as must be 'vera' or 'merchant_on_behalf'")

    # 3. multi-CTA anti-pattern
    reply_count = len(re.findall(r"\breply\s+\w+\b", body, re.IGNORECASE))
    if reply_count > 1:
        problems.append("multiple CTA options detected (anti-pattern: stacked 'reply X' choices)")

    # 4. promotional hype tone
    lowered = body.lower()
    for phrase in PROMO_HYPE_WORDS:
        if phrase in lowered:
            problems.append(f"promotional hype language detected: '{phrase}'")

    # 5. taboo vocabulary from category voice
    taboo = (category.get("voice", {}) or {}).get("vocab_taboo", [])
    for word in taboo:
        if word.lower() in lowered:
            problems.append(f"taboo vocabulary used: '{word}'")

    # 6. anti-repetition vs conversation history
    history = merchant.get("conversation_history", []) or []
    for turn in history:
        if turn.get("from") == "vera" and turn.get("body", "").strip() == body.strip():
            problems.append("body repeats a message already sent verbatim (anti-repetition)")
            break

    # 7. re-introduction after first message
    if history and re.search(r"\b(hi|hello)[,!]?\s+(i'?m|this is)\s+vera\b", lowered):
        problems.append("re-introducing 'I'm Vera' after conversation already has history")

    # 8. long preamble
    if re.match(r"^\s*i hope you'?re doing well", lowered):
        problems.append("long/generic preamble detected")

    # 9. raw JSON field/key name leaked into the message (e.g. "(seasonal_beats)",
    # "delta_pct") -- a snake_case token is essentially never legitimate prose in
    # English or Hindi, so this is a cheap, reliable tell that the model echoed an
    # internal field name instead of translating the fact into natural language.
    leaked_keys = re.findall(r"\b[a-z]+(?:_[a-z]+){1,}\b", body)
    if leaked_keys:
        problems.append(f"raw field/key name(s) leaked into body: {sorted(set(leaked_keys))}")

    return problems
