"""
bot.py
======
The live HTTP server per challenge-testing-brief.md §2. Implements the 5
required endpoints:
  POST /v1/context   - receive versioned context pushes
  POST /v1/tick       - periodic wake-up; bot may initiate proactive sends
  POST /v1/reply      - respond to a merchant/customer reply, synchronously
  GET  /v1/healthz    - liveness probe
  GET  /v1/metadata   - bot identity

Run: uvicorn bot:app --host 0.0.0.0 --port 8080
"""
from __future__ import annotations

import logging
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from context_store import ContextStore
from composer import compose
from conversation_handlers import ConversationState, respond, nudge_unanswered
from llm import ANTHROPIC_MODEL, GROQ_MODEL

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("vera.bot")

app = FastAPI(title="Vera Challenge Bot")
START = time.time()

store = ContextStore()
conversations: dict[str, ConversationState] = {}
# Dedup: don't fire the same suppression_key to the same merchant twice within a test run.
sent_suppression_keys: set[tuple[str, str]] = set()
MAX_UNANSWERED_NUDGES = 3
MAX_ACTIONS_PER_TICK = 20


def _active_model_label() -> str:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return f"Anthropic {ANTHROPIC_MODEL}"
    if os.environ.get("GROQ_API_KEY"):
        return f"Groq {GROQ_MODEL}"
    return "no LLM key set -- running on deterministic fallback templates"


TEAM_METADATA = {
    "team_name": "Team Vera Rebuild",
    "team_members": ["Naman Pandit"],
    "model": _active_model_label(),
    "approach": "single-prompt composer over the 4-context bundle, with a post-generation "
                "validator + one repair re-prompt, plus a rule-based multi-turn layer for "
                "auto-reply detection / intent transitions / graceful exit.",
    "contact_email": "namanpandit2502@gmail.com",
    "version": "0.1.0",
    "submitted_at": datetime.now(timezone.utc).isoformat(),
}


# ---------------------------------------------------------------------------
# GET /v1/healthz
# ---------------------------------------------------------------------------
@app.get("/v1/healthz")
async def healthz():
    counts = store.counts()
    for scope in ("category", "merchant", "customer", "trigger"):
        counts.setdefault(scope, 0)
    return {"status": "ok", "uptime_seconds": int(time.time() - START), "contexts_loaded": counts}


# ---------------------------------------------------------------------------
# GET /v1/metadata
# ---------------------------------------------------------------------------
@app.get("/v1/metadata")
async def metadata():
    return TEAM_METADATA


# ---------------------------------------------------------------------------
# POST /v1/context
# ---------------------------------------------------------------------------
class CtxBody(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: dict[str, Any]
    delivered_at: Optional[str] = None


VALID_SCOPES = {"category", "merchant", "customer", "trigger"}


@app.post("/v1/context")
async def push_context(body: CtxBody):
    if body.scope not in VALID_SCOPES:
        return {"accepted": False, "reason": "invalid_scope", "details": f"scope must be one of {sorted(VALID_SCOPES)}"}
    result = store.push(body.scope, body.context_id, body.version, body.payload)
    if not result["accepted"]:
        return result
    return {
        "accepted": True,
        "ack_id": f"ack_{body.context_id}_v{body.version}",
        "stored_at": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# POST /v1/tick
# ---------------------------------------------------------------------------
class TickBody(BaseModel):
    now: str
    available_triggers: list[str] = []


@app.post("/v1/tick")
async def tick(body: TickBody):
    actions = []
    for trg_id in body.available_triggers:
        if len(actions) >= MAX_ACTIONS_PER_TICK:
            break
        trigger = store.get("trigger", trg_id)
        if not trigger:
            continue

        merchant_id = trigger.get("merchant_id")
        customer_id = trigger.get("customer_id")
        bundle = store.resolve_bundle(merchant_id, trigger_id=trg_id, customer_id=customer_id)
        merchant, category = bundle["merchant"], bundle["category"]
        if not (merchant and category):
            continue  # don't have enough context yet — skip, don't guess

        suppression_key = trigger.get("suppression_key", f"{trg_id}:{merchant_id}")
        dedup_key = (merchant_id, suppression_key)
        if dedup_key in sent_suppression_keys:
            continue  # already sent this; restraint over spam
        # Skip if this exact trigger already exists verbatim in the merchant's history
        if any(t.get("body") for t in (merchant.get("conversation_history") or [])
               if trigger.get("suppression_key") and trigger["suppression_key"] in str(t)):
            continue

        try:
            result = compose(
                category, merchant, trigger, bundle["customer"], bundle["resolved_digest_item"]
            )
        except Exception as e:  # never let one bad compose kill the whole tick
            log.exception("compose failed for trigger %s: %s", trg_id, e)
            continue

        conversation_id = f"conv_{merchant_id}_{trg_id}_{uuid.uuid4().hex[:6]}"
        conversations[conversation_id] = ConversationState(
            conversation_id=conversation_id, merchant_id=merchant_id, customer_id=customer_id
        )
        conversations[conversation_id].add_turn("vera", result["body"])
        sent_suppression_keys.add(dedup_key)

        actions.append({
            "conversation_id": conversation_id,
            "merchant_id": merchant_id,
            "customer_id": customer_id,
            "send_as": result.get("send_as", "vera"),
            "trigger_id": trg_id,
            "template_name": f"vera_{trigger.get('kind','generic')}_v1",
            "template_params": [merchant.get("identity", {}).get("name", "")],
            "body": result["body"],
            "cta": result.get("cta", "open_ended"),
            "suppression_key": suppression_key,
            "rationale": result.get("rationale", ""),
        })

    return {"actions": actions}


# ---------------------------------------------------------------------------
# POST /v1/reply
# ---------------------------------------------------------------------------
class ReplyBody(BaseModel):
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    from_role: str
    message: str
    received_at: Optional[str] = None
    turn_number: Optional[int] = None


@app.post("/v1/reply")
async def reply(body: ReplyBody):
    state = conversations.get(body.conversation_id)
    if state is None:
        # judge replied to a conversation we don't remember (shouldn't happen, but be defensive)
        state = ConversationState(
            conversation_id=body.conversation_id,
            merchant_id=body.merchant_id or "",
            customer_id=body.customer_id,
        )
        conversations[body.conversation_id] = state

    merchant_id = body.merchant_id or state.merchant_id
    customer_id = body.customer_id or state.customer_id
    merchant = store.get("merchant", merchant_id)
    if not merchant:
        return {"action": "end", "rationale": f"no merchant context stored for {merchant_id}"}
    category = store.get("category", merchant.get("category_slug"))
    customer = store.get("customer", customer_id) if customer_id else None

    # Best-effort: find the trigger this conversation started from. Our own /v1/tick mints
    # conversation_id as conv_{merchant_id}_{trigger_id}_{rand}, so the trigger_id is embedded;
    # fall back to a synthetic minimal trigger if we can't recover it (e.g. judge-issued ids).
    trigger = None
    cid = body.conversation_id
    if cid.startswith("conv_"):
        for trg_id in store.trigger_ids():
            if trg_id in cid:
                trigger = store.get("trigger", trg_id)
                break
    if trigger is None:
        trigger = {"kind": "conversation_continuation", "scope": "merchant", "source": "internal",
                   "payload": {}, "suppression_key": body.conversation_id}

    resolved_digest_item = None
    if category and trigger.get("payload"):
        item_id = trigger["payload"].get("top_item_id") or trigger["payload"].get("digest_item_id")
        if item_id:
            resolved_digest_item = store.resolve_digest_item(category["slug"], item_id)

    outcome = respond(state, body.message, category or {}, merchant, trigger, customer, resolved_digest_item)
    return outcome


# ---------------------------------------------------------------------------
# Optional teardown hook (testing-brief §11)
# ---------------------------------------------------------------------------
@app.post("/v1/teardown")
async def teardown():
    global store, conversations, sent_suppression_keys
    store = ContextStore()
    conversations = {}
    sent_suppression_keys = set()
    return {"status": "wiped"}
