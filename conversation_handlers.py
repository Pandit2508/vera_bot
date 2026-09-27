"""
conversation_handlers.py
=========================
Implements the "Open challenges" from challenge-brief.md §12 and the Phase 4
replay scenarios from challenge-testing-brief.md §4:
  1. Detect auto-replies (same text 3+ times) vs real merchant replies.
  2. Handle intent transitions ("let's do it") -> switch to action mode immediately.
  3. Know when to stop (explicit not-interested, or after repeated unanswered nudges).
  4. Stay on-mission when the merchant goes off-topic or hostile.

`respond()` is the optional-deliverable entry point from challenge-brief.md §7.4,
and is also what bot.py's /v1/reply handler calls under the hood.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from composer import compose
from llm import compose_via_llm, LLMError
from prompts import SYSTEM_PROMPT

# ---- intent / signal detection -----------------------------------------

INTENT_YES_PATTERNS = [
    r"\blet'?s do (it|this)\b", r"\bgo ahead\b", r"\bok(ay)? (start|let'?s go|do it)\b",
    r"\byes,? (please )?(start|go|do it|send)\b", r"\bi want to (join|start|do this)\b",
    r"\bhaan\b.*\b(karo|kar do|shuru)\b", r"\bmujhe (join|karna) (hai|hoon)\b",
    r"\bsounds good,? (do it|go ahead)\b", r"\bsure,? go ahead\b",
]

NOT_INTERESTED_PATTERNS = [
    r"\bnot interested\b", r"\bstop\b", r"\bplease stop\b", r"\bremove me\b",
    r"\bnahi chahiye\b", r"\bmat karo\b", r"\bno thanks\b", r"\bunsubscribe\b",
    r"\bdon'?t (contact|message) me\b",
]

ABUSE_MARKERS = [r"\bfuck\b", r"\bidiot\b", r"\bstupid\b", r"\bbakwas\b", r"\bbekar\b", r"\bnonsense\b"]

# Requests clearly outside Vera's mission (marketing/GBP/campaigns/offers) that a merchant
# might still throw at the bot — the brief's own Phase-4 example is exactly this ("can you
# also help me file my GST?"). Keyword-based since we can't enumerate every off-topic ask,
# but these cover the realistic ones an Indian small-business owner would raise.
OFF_TOPIC_MARKERS = [
    r"\bgst\b", r"\bincome tax\b", r"\bitr\b", r"\bloan\b", r"\binsurance\b",
    r"\bvisa\b", r"\bpassport\b", r"\blegal advice\b", r"\bmedical advice\b",
    r"\baadhaar\b", r"\bpan card\b", r"\bcompany registration\b",
]

# Common WhatsApp Business canned auto-reply phrasing (English + Hindi). Matching content,
# not just repetition, lets us flag a *likely* auto-reply on the very first occurrence and
# confirm it on the second — this is the "better/faster detection" opportunity called out in
# challenge-brief.md §3.1 (production Vera burns 2-3 turns before it acts).
AUTO_REPLY_CONTENT_MARKERS = [
    r"thank you for contacting", r"we will get back to you", r"automated (assistant|response|reply)",
    r"currently (unavailable|closed|away)", r"business hours", r"aapki jaankari ke liye.*shukriya",
    r"team tak pahuncha", r"main ek automated", r"hum jald hi (aapse )?sampark karenge",
]


def looks_like_auto_reply_content(message: str) -> bool:
    m = _norm(message)
    return any(re.search(p, m) for p in AUTO_REPLY_CONTENT_MARKERS)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def detect_intent_yes(message: str) -> bool:
    m = _norm(message)
    return any(re.search(p, m) for p in INTENT_YES_PATTERNS)


def detect_not_interested(message: str) -> bool:
    m = _norm(message)
    return any(re.search(p, m) for p in NOT_INTERESTED_PATTERNS)


def detect_abuse_or_offtopic(message: str) -> bool:
    m = _norm(message)
    return any(re.search(p, m) for p in ABUSE_MARKERS) or any(re.search(p, m) for p in OFF_TOPIC_MARKERS)


def is_auto_reply(history: list[dict], new_message: str, threshold: int = 3) -> bool:
    """Auto-reply heuristic per challenge-brief.md §12 hint: same message verbatim 3+ times."""
    norm_new = _norm(new_message)
    same_count = 1  # counting this message itself
    for turn in history:
        if turn.get("from") == "merchant" and _norm(turn.get("message", turn.get("body", ""))) == norm_new:
            same_count += 1
    return same_count >= threshold


# ---- conversation state ---------------------------------------------------

@dataclass
class ConversationState:
    conversation_id: str
    merchant_id: str
    customer_id: Optional[str] = None
    turns: list[dict] = field(default_factory=list)     # [{"from": "vera"/"merchant", "message": str}, ...]
    intent_mode: str = "pitch"                            # "pitch" -> "action" once intent detected
    auto_reply_strikes: int = 0
    unanswered_nudges: int = 0
    ended: bool = False

    def add_turn(self, frm: str, message: str):
        self.turns.append({"from": frm, "message": message})


def _thread_reference(trigger: dict, resolved_digest_item: Optional[dict]) -> str:
    """A short phrase pointing back at the actual live thread, so an off-topic/hostile
    redirect doesn't drop the conversation into a generic void (api-call-examples.md
    Ex 2.7: 'Coming back to the JIDA piece — want me to draft the patient post first...')."""
    kind = trigger.get("kind", "")
    if resolved_digest_item:
        title = resolved_digest_item.get("title", "")
        short = title if len(title) <= 40 else title[:40].rsplit(" ", 1)[0] + "..."
        return f"Coming back to the {short} piece — want me to go ahead?"
    topic = kind.replace("_", " ") if kind else "what we were discussing"
    return f"Coming back to the {topic} — want me to go ahead, or was there something specific you needed there?"


# ---- the reply handler --------------------------------------------------

def respond(
    state: ConversationState,
    merchant_message: str,
    category: dict,
    merchant: dict,
    trigger: dict,
    customer: Optional[dict] = None,
    resolved_digest_item: Optional[dict] = None,
) -> dict:
    """
    Given the conversation so far + the merchant's latest message, produce the
    next move: {"action": "send"|"wait"|"end", "body"?, "cta"?, "rationale"}.
    """
    if state.ended:
        return {"action": "end", "rationale": "conversation already closed"}

    # 1. Auto-reply detection — content match flags it as early as the FIRST occurrence
    #    (faster than pure repetition-counting), repetition confirms it. Either signals it.
    # 3-strike pattern per the harness's own worked example (api-call-examples.md Ex 4.1):
    #   strike 1 -> one nudge (send); strike 2 -> back off (wait); strike 3+ -> exit (end).
    likely_auto_reply = looks_like_auto_reply_content(merchant_message) or is_auto_reply(
        state.turns, merchant_message, threshold=2
    )
    if likely_auto_reply:
        state.auto_reply_strikes += 1
        state.add_turn("merchant", merchant_message)
        if state.auto_reply_strikes == 1:
            body = ("Samajh gayi. Team tak pahunchane se pehle, kya aap khud 2 minute mein "
                    "dekhna chahenge ki exactly kya karna hai? Chalega?")
            state.add_turn("vera", body)
            return {"action": "send", "body": body, "cta": "binary",
                    "rationale": "Detected likely auto-reply; one gentle nudge to reach a human before backing off."}
        if state.auto_reply_strikes == 2:
            return {"action": "wait", "wait_seconds": 14400,
                    "rationale": "Same auto-reply pattern again; owner likely not at phone. Backing off 4h before retrying rather than burning another turn."}
        state.ended = True
        body = "Koi baat nahi, samajh gayi. Main directly owner/manager se connect kar lungi. Best wishes! 🙂"
        state.add_turn("vera", body)
        return {"action": "end", "body": body,
                "rationale": "Auto-reply confirmed 3x with no real engagement signal; exiting rather than continuing to burn turns."}

    state.add_turn("merchant", merchant_message)

    # 2. Explicit not-interested -> graceful exit, no further nudging
    if detect_not_interested(merchant_message):
        state.ended = True
        return {"action": "end",
                "rationale": "Merchant explicitly signaled not interested; exiting without pushing further."}

    # 3. Intent transition -> switch to action mode, route immediately (don't re-qualify)
    if detect_intent_yes(merchant_message):
        state.intent_mode = "action"
        body = _action_mode_message(merchant, trigger, resolved_digest_item)
        state.add_turn("vera", body)
        return {"action": "send", "body": body, "cta": "open_ended",
                "rationale": "Merchant gave explicit go-ahead; routing straight to action, no re-qualification."}

    # 4. Abusive -> absorb politely, redirect once; Off-topic -> decline scope politely, redirect once
    is_abusive = any(re.search(p, _norm(merchant_message)) for p in ABUSE_MARKERS)
    is_offtopic = any(re.search(p, _norm(merchant_message)) for p in OFF_TOPIC_MARKERS)
    if is_abusive or is_offtopic:
        thread_ref = _thread_reference(trigger, resolved_digest_item)
        if is_abusive:
            body = ("Samajh sakti hoon. Main sirf aapke magicpin/Google profile aur marketing mein "
                    f"madad kar sakti hoon — baaki cheezon ke liye aapko sahi jagah point kar deti hoon. "
                    f"{thread_ref}")
            rationale = "Acknowledged hostile message politely, redirected back to the original trigger thread without engaging the hostility."
        else:
            body = (f"That's outside what I can help with directly. {thread_ref}")
            rationale = "Off-topic request outside Vera's mission; declined scope politely and picked the original thread back up rather than a generic redirect."
        state.add_turn("vera", body)
        return {"action": "send", "body": body, "cta": "open_ended", "rationale": rationale}

    # 5. Otherwise — compose a genuine contextual reply via the core composer,
    #    passing the live thread so the model doesn't repeat itself or re-introduce Vera.
    result = compose(
        category, merchant, trigger, customer, resolved_digest_item,
        conversation_so_far=state.turns,
    )
    prior_bodies = {_norm(t["message"]) for t in state.turns if t["from"] == "vera"}
    if _norm(result.get("body", "")) in prior_bodies:
        # Safety net for the deterministic fallback composer, which doesn't read
        # conversation history: never resend a message verbatim (anti-repetition, §11).
        name = merchant.get("identity", {}).get("owner_first_name") or merchant.get("identity", {}).get("name", "")
        result = {
            "body": f"{name}, just circling back on this — happy to answer anything specific, or I can go ahead if you're good with it.",
            "cta": "open_ended",
            "rationale": "Avoided resending an identical prior message; offered a distinct low-friction follow-up instead.",
        }
    state.add_turn("vera", result["body"])
    return {"action": "send", "body": result["body"], "cta": result.get("cta", "open_ended"),
            "rationale": result.get("rationale", "continued conversation")}


def _action_mode_message(merchant: dict, trigger: dict, resolved_digest_item: Optional[dict]) -> str:
    """Fast, low-friction action-mode message once intent is confirmed. Deliberately template-first
    (not another LLM round-trip) so we never re-ask a qualifying question after a clear 'yes'."""
    name = merchant.get("identity", {}).get("owner_first_name") or merchant.get("identity", {}).get("name", "")
    if trigger.get("kind") == "research_digest" and resolved_digest_item:
        return (f"Great, {name} — sending the abstract now and drafting a 2-line patient-ed WhatsApp "
                f"you can review and reshare. Give me a sec.")
    return f"Great, {name} — starting now, I'll confirm here once it's done."


def nudge_unanswered(state: ConversationState, max_nudges: int = 3) -> bool:
    """Called by the tick loop when a sent message got no reply. Returns True if the
    conversation should now be abandoned (per §12 point 5 — know when to stop)."""
    state.unanswered_nudges += 1
    if state.unanswered_nudges >= max_nudges:
        state.ended = True
        return True
    return False
