"""
composer.py
===========
The core `compose()` — this is the "brain" referenced in challenge-brief.md §5.
Same function is used by:
  - the offline submission.jsonl generator (generate_submission.py)
  - the live HTTP server's /v1/tick and /v1/reply handlers (bot.py)

Pipeline: build prompt -> call LLM (or fallback) -> validate -> one repair
re-prompt if validation fails -> return ComposedMessage dict.
"""
from __future__ import annotations

import logging
from typing import Optional

from prompts import SYSTEM_PROMPT, build_user_prompt
from llm import compose_via_llm, LLMError
from fallback_composer import compose_fallback
from validators import validate_output

log = logging.getLogger("vera.composer")


def compose(
    category: dict,
    merchant: dict,
    trigger: dict,
    customer: Optional[dict] = None,
    resolved_digest_item: Optional[dict] = None,
    conversation_so_far: Optional[list[dict]] = None,
    patient: bool = False,
) -> dict:
    """
    Returns a dict with keys: body, cta, send_as, suppression_key, rationale.
    Deterministic (temperature=0) when using the LLM path.

    `patient`: pass True from offline/batch callers (e.g. generate_submission.py)
    to allow a much larger retry budget on rate limits. Leave False (default) for
    anything on the live bot.py server's request path, which has a 30s SLA per
    challenge-testing-brief.md and must fail fast to the deterministic fallback
    rather than risk hanging on a long retry wait.
    """
    suppression_key = trigger.get("suppression_key", f"{trigger.get('kind','trg')}:{merchant.get('merchant_id','?')}")

    result = _compose_once(category, merchant, trigger, customer, resolved_digest_item, conversation_so_far, patient)
    problems = validate_output(result, category, merchant, customer)

    if problems:
        log.warning("validation issues on first attempt: %s", problems)
        repaired = _repair_attempt(category, merchant, trigger, customer, resolved_digest_item,
                                    conversation_so_far, result, problems, patient)
        if repaired is not None:
            repaired_problems = validate_output(repaired, category, merchant, customer)
            if len(repaired_problems) < len(problems):
                repaired["used_fallback"] = False  # repair only ever runs via the LLM path
                result = repaired
                problems = repaired_problems

    result.setdefault("cta", "open_ended")
    result.setdefault("send_as", "merchant_on_behalf" if customer else "vera")
    result["suppression_key"] = suppression_key
    result.setdefault("rationale", "composed message")
    result.setdefault("used_fallback", False)
    if problems:
        result["rationale"] = (result.get("rationale") or "") + f" [unresolved validation flags: {problems}]"
    return result


def _compose_once(category, merchant, trigger, customer, resolved_digest_item, conversation_so_far, patient=False) -> dict:
    user_prompt = build_user_prompt(category, merchant, trigger, customer, resolved_digest_item, conversation_so_far)
    try:
        out = compose_via_llm(SYSTEM_PROMPT, user_prompt, patient=patient)
    except LLMError as e:
        log.error("LLM call failed, using fallback composer: %s", e)
        out = None
    if out is None:
        out = compose_fallback(category, merchant, trigger, customer, resolved_digest_item)
        out["used_fallback"] = True
    else:
        out["used_fallback"] = False
    return out


def _repair_attempt(category, merchant, trigger, customer, resolved_digest_item,
                     conversation_so_far, prior_result: dict, problems: list[str], patient: bool = False) -> Optional[dict]:
    """One re-prompt asking the LLM to specifically fix the flagged issues."""
    import json

    repair_prompt = (
        build_user_prompt(category, merchant, trigger, customer, resolved_digest_item, conversation_so_far)
        + "\n\nYour previous attempt had problems, fix them:\n"
        + f"Previous output: {json.dumps(prior_result, ensure_ascii=False)}\n"
        + f"Problems found: {problems}\n"
        + "Return a corrected JSON object with the same schema."
    )
    try:
        return compose_via_llm(SYSTEM_PROMPT, repair_prompt, patient=patient)
    except LLMError as e:
        log.error("repair attempt failed: %s", e)
        return None
