"""
fallback_composer.py
=====================
A deterministic, template-based composer used ONLY when no LLM key is
configured (ANTHROPIC_API_KEY unset). It keeps the pipeline runnable offline
(so you can exercise the HTTP server, validators, and judge_simulator.py
without spending API credits) with genuinely specific, rubric-shaped output
rather than one generic filler line.

Design notes:
- Every trigger `kind` the dataset can produce (see dataset/triggers_seed.json
  and the `additional_kinds` list in dataset/generate_dataset.py) has its own
  template below, keyed in KIND_HANDLERS.
- The generator's non-seed ("generated") triggers carry a placeholder payload
  (`{"placeholder": True, "metric_or_topic": kind}`) instead of real fields —
  a template must never print those placeholder values verbatim. Each handler
  therefore checks `_is_placeholder(trigger)` and, when true, falls back to
  whatever REAL data is available on the merchant/customer/category objects
  (performance, offers, signals, review_themes, subscription, relationship,
  etc.) instead of the empty trigger payload. That real data is always
  present because merchants/customers are fully populated even when their
  triggers aren't.
- Swap in a real ANTHROPIC_API_KEY for actual submissions — this file is a
  safety net, not the rubric-winning path.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional


# --------------------------------------------------------------------------
# small shared helpers
# --------------------------------------------------------------------------

def _first_name(merchant: dict) -> str:
    return merchant.get("identity", {}).get("owner_first_name") or merchant.get("identity", {}).get("name", "there")


def _cust_name(customer: dict) -> str:
    name = (customer or {}).get("identity", {}).get("name", "there")
    # strip "(parent: X)" style annotations for direct address
    return name.split(" (")[0] if " (" in name else name


def _is_hindi_friendly(merchant: dict, customer: Optional[dict]) -> bool:
    if customer and "hi" in (customer.get("identity", {}).get("language_pref", "")):
        return True
    return "hi" in (merchant.get("identity", {}).get("languages") or [])


def _is_placeholder(trigger: dict) -> bool:
    return bool((trigger.get("payload") or {}).get("placeholder"))


def _payload(trigger: dict) -> dict:
    return trigger.get("payload") or {}


def _pct(x, signed=False) -> str:
    if not isinstance(x, (int, float)):
        return "some"
    val = f"{abs(x) * 100:.0f}%"
    if signed:
        return f"+{val}" if x >= 0 else f"-{val}"
    return val


def _human_date(iso: Optional[str]) -> Optional[str]:
    if not iso:
        return None
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return dt.strftime("%d %b").lstrip("0")
    except Exception:
        return None


def _active_offer(merchant: dict) -> Optional[str]:
    for o in merchant.get("offers", []):
        if o.get("status") == "active":
            return o.get("title")
    return None


def _catalog_offer(category: dict, audience: Optional[str] = None) -> Optional[str]:
    catalog = category.get("offer_catalog", []) or []
    if audience:
        for o in catalog:
            if o.get("audience") == audience:
                return o.get("title")
    return catalog[0]["title"] if catalog else None


def _best_offer(merchant: dict, category: dict, audience: Optional[str] = None) -> str:
    return _active_offer(merchant) or _catalog_offer(category, audience) or "your current offer"


def _top_signal(merchant: dict) -> Optional[str]:
    sig = merchant.get("signals") or []
    return sig[0] if sig else None


def _top_review_theme(merchant: dict) -> Optional[dict]:
    themes = merchant.get("review_themes") or []
    return themes[0] if themes else None


def _dentist_style(category: dict, name: str) -> str:
    if category.get("slug") == "dentists" and not name.startswith("Dr."):
        return f"Dr. {name}"
    return name


# --------------------------------------------------------------------------
# merchant-scope, external triggers
# --------------------------------------------------------------------------

def _research_digest(category, merchant, trigger, customer, digest_item) -> dict:
    name = _dentist_style(category, _first_name(merchant))
    if digest_item:
        title = digest_item.get("title", "")
        source = digest_item.get("source", "")
        actionable = digest_item.get("actionable", "")
        body = f"{name}, {source} just landed — {title}. {actionable}. Want me to draft a quick note for your patients on this?"
        return {"body": body, "cta": "open_ended", "send_as": "vera",
                "rationale": "Category research digest, anchored on the actual cited item."}
    body = f"{name}, there's a new item in this week's {category.get('display_name', 'category')} research digest — want me to pull the one most relevant to your patient mix?"
    return {"body": body, "cta": "open_ended", "send_as": "vera",
            "rationale": "Research digest trigger without a resolvable item; asked rather than fabricated."}


def _cde_opportunity(category, merchant, trigger, customer, digest_item) -> dict:
    name = _dentist_style(category, _first_name(merchant))
    payload = _payload(trigger)
    credits = payload.get("credits")
    fee = payload.get("fee")
    if digest_item and digest_item.get("kind") == "cde":
        title = digest_item.get("title", "")
        bits = []
        if credits:
            bits.append(f"{credits} CE credits")
        if fee == "free_for_members":
            bits.append("free for members")
        extra = f" ({', '.join(bits)})" if bits else ""
        body = f"{name}, {title}{extra}. Want me to send you the registration link?"
        return {"body": body, "cta": "binary", "send_as": "vera",
                "rationale": "CDE opportunity anchored on the resolved digest item."}
    body = f"{name}, there's a continuing-education session open for {category.get('display_name', 'your category')} this week — want the details?"
    return {"body": body, "cta": "binary", "send_as": "vera",
            "rationale": "CDE trigger without a resolvable digest item; kept general rather than inventing a session name."}


def _regulation_change(category, merchant, trigger, customer, digest_item) -> dict:
    name = _dentist_style(category, _first_name(merchant))
    payload = _payload(trigger)
    deadline = _human_date(payload.get("deadline_iso"))
    if digest_item:
        title = digest_item.get("title", "")
        actionable = digest_item.get("actionable", "")
        # only append the deadline separately if the actionable text doesn't already state one
        by_txt = f" (deadline {deadline})" if deadline and deadline.lower() not in actionable.lower() and "before" not in actionable.lower() and "by " not in actionable.lower() else ""
        body = f"{name}, heads up — {title}. {actionable}{by_txt}."
        return {"body": body, "cta": "open_ended", "send_as": "vera",
                "rationale": "Regulation change anchored on the resolved compliance digest item."}
    by_txt = f" by {deadline}" if deadline else ""
    body = f"{name}, there's a regulatory update affecting {category.get('display_name', 'your category')}{by_txt} — want the summary?"
    return {"body": body, "cta": "binary", "send_as": "vera",
            "rationale": "Regulation trigger without a resolvable item; asked rather than fabricated."}


def _competitor_opened(category, merchant, trigger, customer, digest_item) -> dict:
    name = _dentist_style(category, _first_name(merchant))
    payload = _payload(trigger)
    if not _is_placeholder(trigger):
        comp = payload.get("competitor_name", "a new competitor")
        dist = payload.get("distance_km")
        offer = payload.get("their_offer")
        dist_txt = f"{dist} km from you" if isinstance(dist, (int, float)) else "nearby"
        offer_txt = f" running {offer}" if offer else ""
        body = f"{name}, {comp} just opened {dist_txt}{offer_txt}. Want me to check how your own offers compare?"
        return {"body": body, "cta": "binary", "send_as": "vera",
                "rationale": "Competitor-opened trigger with real competitor/distance/offer data."}
    sig = _top_signal(merchant)
    sig_txt = f" ({sig})" if sig else ""
    body = f"{name}, spotted new competition in your locality{sig_txt} — want me to check how your listing compares on offers and reviews?"
    return {"body": body, "cta": "binary", "send_as": "vera",
            "rationale": "Competitor trigger without real competitor detail; used merchant signal instead of fabricating a name."}


def _festival_upcoming(category, merchant, trigger, customer, digest_item) -> dict:
    name = _dentist_style(category, _first_name(merchant))
    payload = _payload(trigger)
    if not _is_placeholder(trigger):
        festival = payload.get("festival", "the festival season")
        days = payload.get("days_until")
        days_txt = f" ({days} days out)" if isinstance(days, int) else ""
        body = f"{name}, {festival} is coming up{days_txt} — want me to draft a seasonal post or offer for it?"
        return {"body": body, "cta": "binary", "send_as": "vera",
                "rationale": "Festival trigger with real festival name/date."}
    beats = category.get("seasonal_beats", [])
    note = beats[0]["note"] if beats else "the upcoming festive season"
    body = f"{name}, {note} — want a plan drafted for it?"
    return {"body": body, "cta": "binary", "send_as": "vera",
            "rationale": "Festival trigger without a specific date; drew on the category's own seasonal pattern instead of inventing one."}


def _category_seasonal(category, merchant, trigger, customer, digest_item) -> dict:
    name = _first_name(merchant)
    payload = _payload(trigger)
    trends = payload.get("trends") or []
    top = trends[0].replace("_", " ") if trends else "a seasonal demand shift"
    body = f"{name}, seasonal shift incoming — {top} — worth adjusting your shelf/stock focus this week?"
    return {"body": body, "cta": "binary", "send_as": "vera",
            "rationale": "Category-wide seasonal demand trend, real payload data."}


def _ipl_match_today(category, merchant, trigger, customer, digest_item) -> dict:
    name = _first_name(merchant)
    payload = _payload(trigger)
    match = payload.get("match", "tonight's IPL match")
    weeknight = payload.get("is_weeknight")
    tag = "weeknight" if weeknight else "weekend"
    body = f"{name}, {match} is on tonight ({tag}) — want me to push a match-night combo post before kickoff?"
    return {"body": body, "cta": "binary", "send_as": "vera",
            "rationale": "IPL match-night trigger, real fixture data."}


def _gbp_unverified(category, merchant, trigger, customer, digest_item) -> dict:
    name = _first_name(merchant)
    payload = _payload(trigger)
    uplift = payload.get("estimated_uplift_pct")
    uplift_txt = f" (~{_pct(uplift)} more visibility)" if uplift else ""
    body = f"{name}, your listing isn't verified yet{uplift_txt} — want me to send the quick verification steps?"
    return {"body": body, "cta": "binary", "send_as": "vera",
            "rationale": "GBP verification trigger, real payload."}


def _supply_alert(category, merchant, trigger, customer, digest_item) -> dict:
    name = _first_name(merchant)
    payload = _payload(trigger)
    molecule = payload.get("molecule", "a stocked molecule")
    body = f"{name}, recall notice on {molecule} — want the affected batch numbers so you can pull stock today?"
    return {"body": body, "cta": "binary", "send_as": "vera",
            "rationale": "Supply/recall alert, real payload."}


def _winback_eligible(category, merchant, trigger, customer, digest_item) -> dict:
    name = _first_name(merchant)
    payload = _payload(trigger)
    days = payload.get("days_since_expiry")
    added = payload.get("lapsed_customers_added_since_expiry")
    bits = []
    if isinstance(days, int):
        bits.append(f"{days} days since your plan lapsed")
    if isinstance(added, int):
        bits.append(f"{added} customers have gone quiet since")
    detail = "; ".join(bits) if bits else "your plan has lapsed"
    body = f"{name}, {detail} — want me to send a quick winback offer to reactivate them?"
    return {"body": body, "cta": "binary", "send_as": "vera",
            "rationale": "Winback-eligible trigger anchored on real lapse metrics."}


def _review_theme_emerged(category, merchant, trigger, customer, digest_item) -> dict:
    name = _first_name(merchant)
    payload = _payload(trigger)
    if not _is_placeholder(trigger):
        theme = payload.get("theme", "a recurring theme").replace("_", " ")
        occ = payload.get("occurrences_30d")
        occ_txt = f" ({occ}x this month)" if isinstance(occ, int) else ""
        body = f"{name}, '{theme}' is coming up in reviews{occ_txt} — want a suggested response or fix?"
        return {"body": body, "cta": "binary", "send_as": "vera",
                "rationale": "Review-theme trigger with real theme/frequency data."}
    top = _top_review_theme(merchant)
    if top:
        theme = top.get("theme", "a recurring theme").replace("_", " ")
        occ = top.get("occurrences_30d")
        occ_txt = f" ({occ}x this month)" if occ else ""
        body = f"{name}, '{theme}' keeps coming up in reviews{occ_txt} — want a suggested response?"
        return {"body": body, "cta": "binary", "send_as": "vera",
                "rationale": "Placeholder trigger payload; used merchant's own real review_themes instead."}
    body = f"{name}, a new pattern showed up in your recent reviews — want me to pull the details?"
    return {"body": body, "cta": "binary", "send_as": "vera",
            "rationale": "No review theme data available on trigger or merchant; kept the ask open rather than fabricating."}


def _milestone_reached(category, merchant, trigger, customer, digest_item) -> dict:
    name = _first_name(merchant)
    payload = _payload(trigger)
    if not _is_placeholder(trigger):
        metric = payload.get("metric", "milestone").replace("_", " ")
        now = payload.get("value_now")
        target = payload.get("milestone_value")
        if isinstance(now, (int, float)) and isinstance(target, (int, float)):
            body = f"{name}, you're at {now} {metric}, {target - now} away from {target} — want a post ready for when you hit it?"
        else:
            body = f"{name}, you're close to a {metric} milestone — want a post ready for it?"
        return {"body": body, "cta": "binary", "send_as": "vera",
                "rationale": "Milestone trigger with real progress numbers."}
    total = merchant.get("customer_aggregate", {}).get("total_unique_ytd")
    if isinstance(total, int):
        body = f"{name}, you've served {total} unique customers this year — want a shoutout post for it?"
        return {"body": body, "cta": "binary", "send_as": "vera",
                "rationale": "Placeholder milestone trigger; used merchant's real YTD customer count instead."}
    body = f"{name}, you're closing in on a milestone — want me to check exactly which one and draft a post?"
    return {"body": body, "cta": "binary", "send_as": "vera",
            "rationale": "No milestone data available; kept the ask open rather than fabricating a number."}


def _active_planning_intent(category, merchant, trigger, customer, digest_item) -> dict:
    name = _first_name(merchant)
    payload = _payload(trigger)
    topic = payload.get("intent_topic", "that idea").replace("_", " ")
    body = f"{name}, on the {topic} — want me to draft the first version so you can react to specifics instead of a blank page?"
    return {"body": body, "cta": "open_ended", "send_as": "vera",
            "rationale": "Continues the merchant's own in-progress planning thread; avoids re-qualifying (anti-pattern D)."}


def _seasonal_perf_dip(category, merchant, trigger, customer, digest_item) -> dict:
    name = _first_name(merchant)
    payload = _payload(trigger)
    metric = payload.get("metric", "traffic")
    delta = payload.get("delta_pct")
    note = (payload.get("season_note") or "").replace("_", " ")
    delta_txt = _pct(delta, signed=True) if isinstance(delta, (int, float)) else "lower than usual"
    body = f"{name}, {metric} is {delta_txt} this week — this lines up with {note or 'the usual seasonal pattern'}, so no action needed unless you want a plan for the rebound."
    return {"body": body, "cta": "open_ended", "send_as": "vera",
            "rationale": "Expected seasonal dip — reassurance framing, not a false alarm."}


def _renewal_due(category, merchant, trigger, customer, digest_item) -> dict:
    name = _first_name(merchant)
    payload = _payload(trigger)
    if not _is_placeholder(trigger):
        days = payload.get("days_remaining")
        amount = payload.get("renewal_amount")
        plan = payload.get("plan", "your plan")
        amt_txt = f" (₹{amount})" if amount else ""
        body = f"{name}, {plan} renews in {days} days{amt_txt} — want me to lock it in now so there's no gap?"
        return {"body": body, "cta": "binary", "send_as": "vera",
                "rationale": "Renewal trigger with real days-remaining/amount."}
    sub = merchant.get("subscription", {})
    days = sub.get("days_remaining")
    plan = sub.get("plan", "your plan")
    if isinstance(days, int):
        body = f"{name}, {plan} renews in {days} days — want me to lock it in now so there's no gap?"
        return {"body": body, "cta": "binary", "send_as": "vera",
                "rationale": "Placeholder renewal trigger; used merchant's real subscription data instead."}
    body = f"{name}, your subscription is coming up for renewal — want the details?"
    return {"body": body, "cta": "binary", "send_as": "vera",
            "rationale": "No renewal specifics available; kept general rather than fabricating a date."}


def _curious_ask_due(category, merchant, trigger, customer, digest_item) -> dict:
    name = _first_name(merchant)
    body = f"{name}, quick one — what's the one service getting the most walk-in asks this week? Helps me target your next post."
    return {"body": body, "cta": "open_ended", "send_as": "vera",
            "rationale": "Low-pressure relationship-building question; doesn't depend on trigger payload, so it's placeholder-safe by design."}


def _dormant_with_vera(category, merchant, trigger, customer, digest_item) -> dict:
    name = _first_name(merchant)
    body = f"{name}, haven't heard from you in a bit — want a quick profile check-in, or should I hold off?"
    return {"body": body, "cta": "binary", "send_as": "vera",
            "rationale": "Dormant re-engagement, low-pressure binary ask."}


def _perf_dip_spike(category, merchant, trigger, customer, digest_item) -> dict:
    name = _first_name(merchant)
    kind = trigger.get("kind", "")
    perf = merchant.get("performance", {})
    delta = perf.get("delta_7d", {})
    views_pct = delta.get("views_pct")
    direction = "up" if kind == "perf_spike" else "down"
    pct_txt = _pct(views_pct) if isinstance(views_pct, (int, float)) else "a fair bit"
    body = (
        f"{name}, heads up — your views are {direction} {pct_txt} vs last week "
        f"({perf.get('views', '?')} views / {perf.get('calls', '?')} calls, 30d). "
        f"Want me to show what's driving it?"
    )
    return {"body": body, "cta": "open_ended", "send_as": "vera",
            "rationale": f"Internal {kind} trigger anchored on real merchant performance delta (payload-agnostic, so placeholder-safe)."}


# --------------------------------------------------------------------------
# customer-scope triggers
# --------------------------------------------------------------------------

def _recall_due(category, merchant, trigger, customer, digest_item) -> dict:
    if not customer:
        return _dormant_with_vera(category, merchant, trigger, customer, digest_item)
    cname = _cust_name(customer)
    mname = merchant.get("identity", {}).get("name", "we")
    payload = _payload(trigger)
    slots = payload.get("available_slots") or []
    if not _is_placeholder(trigger) and slots:
        slot_txt = slots[0].get("label", "your usual slot")
        service = (payload.get("service_due") or "your next visit").replace("_", " ")
        body = f"Hi {cname}, {mname} here. Time for {service} — {slot_txt} is open, or reply with a time that works?"
        return {"body": body, "cta": "open_ended", "send_as": "merchant_on_behalf",
                "rationale": "Customer recall_due trigger with a real proposed slot."}
    active_offers = [o for o in merchant.get("offers", []) if o.get("status") == "active"]
    offer_txt = active_offers[0]["title"] if active_offers else "your next visit"
    body = (
        f"Hi {cname}, {mname} here. It's been a while since your last visit — worth booking {offer_txt.lower()}? "
        f"Reply with a time that works."
    )
    return {"body": body, "cta": "open_ended", "send_as": "merchant_on_behalf",
            "rationale": "Customer recall_due trigger with a real active offer (single CTA)."}


def _wedding_package_followup(category, merchant, trigger, customer, digest_item) -> dict:
    if not customer:
        return _dormant_with_vera(category, merchant, trigger, customer, digest_item)
    cname = _cust_name(customer)
    mname = merchant.get("identity", {}).get("name", "we")
    payload = _payload(trigger)
    days = payload.get("days_to_wedding")
    days_txt = f" ({days} days to go)" if isinstance(days, int) else ""
    body = f"Hi {cname}, {mname} here — following up on your bridal trial{days_txt}. Ready to lock in the pre-wedding skin prep sessions?"
    return {"body": body, "cta": "binary", "send_as": "merchant_on_behalf",
            "rationale": "Wedding-package followup, real days-to-wedding data."}


def _customer_lapsed_hard(category, merchant, trigger, customer, digest_item) -> dict:
    if not customer:
        return _dormant_with_vera(category, merchant, trigger, customer, digest_item)
    cname = _cust_name(customer)
    mname = merchant.get("identity", {}).get("name", "we")
    payload = _payload(trigger)
    focus = (payload.get("previous_focus") or "your goals").replace("_", " ")
    body = f"Hi {cname}, {mname} here — it's been a while since we worked on {focus} together. Want to pick back up, or should I check in later?"
    return {"body": body, "cta": "binary", "send_as": "merchant_on_behalf",
            "rationale": "Hard-lapsed customer, real previous-focus data, low-pressure binary."}


def _customer_lapsed_soft(category, merchant, trigger, customer, digest_item) -> dict:
    if not customer:
        return _dormant_with_vera(category, merchant, trigger, customer, digest_item)
    cname = _cust_name(customer)
    mname = merchant.get("identity", {}).get("name", "we")
    rel = customer.get("relationship", {})
    last_visit = _human_date(rel.get("last_visit")) if rel.get("last_visit") else None
    since_txt = f" since {last_visit}" if last_visit else " in a while"
    offer = _best_offer(merchant, category)
    body = f"Hi {cname}, {mname} here — haven't seen you{since_txt}. {offer} is on if you want to swing back?"
    return {"body": body, "cta": "binary", "send_as": "merchant_on_behalf",
            "rationale": "Placeholder trigger; used the customer's own real last_visit date and merchant's real active offer."}


def _trial_followup(category, merchant, trigger, customer, digest_item) -> dict:
    if not customer:
        return _dormant_with_vera(category, merchant, trigger, customer, digest_item)
    cname = _cust_name(customer)
    mname = merchant.get("identity", {}).get("name", "we")
    payload = _payload(trigger)
    options = payload.get("next_session_options") or []
    if not _is_placeholder(trigger) and options:
        slot = options[0].get("label", "your next slot")
        body = f"Hi {cname}, {mname} here — how was the trial? {slot} is open if you'd like to continue, or tell us what works."
        return {"body": body, "cta": "open_ended", "send_as": "merchant_on_behalf",
                "rationale": "Trial followup with a real proposed next-session slot."}
    body = f"Hi {cname}, {mname} here — how was the trial session? Want me to set up your next one?"
    return {"body": body, "cta": "binary", "send_as": "merchant_on_behalf",
            "rationale": "Placeholder trial followup; kept the ask general rather than inventing a slot."}


def _chronic_refill_due(category, merchant, trigger, customer, digest_item) -> dict:
    if not customer:
        return _dormant_with_vera(category, merchant, trigger, customer, digest_item)
    cname = _cust_name(customer)
    mname = merchant.get("identity", {}).get("name", "we")
    payload = _payload(trigger)
    molecules = payload.get("molecule_list") or []
    if not _is_placeholder(trigger) and molecules:
        med_txt = molecules[0] if len(molecules) == 1 else f"{molecules[0]} + {len(molecules)-1} more"
        runs_out = _human_date(payload.get("stock_runs_out_iso"))
        runs_out_txt = f" runs out around {runs_out}" if runs_out else " is running low"
        body = f"Hi {cname}, {mname} here — your {med_txt}{runs_out_txt}. Shall I get the refill ready for pickup or delivery?"
        return {"body": body, "cta": "binary", "send_as": "merchant_on_behalf",
                "rationale": "Chronic refill trigger with real molecule/stock-out data."}
    body = f"Hi {cname}, {mname} here — looks like your regular refill is due soon. Shall I get it ready?"
    return {"body": body, "cta": "binary", "send_as": "merchant_on_behalf",
            "rationale": "Placeholder refill trigger; kept the ask general rather than naming a molecule we don't actually have."}


def _appointment_tomorrow(category, merchant, trigger, customer, digest_item) -> dict:
    if not customer:
        return _dormant_with_vera(category, merchant, trigger, customer, digest_item)
    cname = _cust_name(customer)
    mname = merchant.get("identity", {}).get("name", "we")
    body = f"Hi {cname}, quick reminder — you're booked in with {mname} tomorrow. Reply if you need to reschedule."
    return {"body": body, "cta": "binary", "send_as": "merchant_on_behalf",
            "rationale": "Appointment reminder; always a placeholder-payload kind, so built entirely from merchant/customer identity (no fabricated time)."}


# --------------------------------------------------------------------------
# dispatch
# --------------------------------------------------------------------------

KIND_HANDLERS = {
    "research_digest": _research_digest,
    "cde_opportunity": _cde_opportunity,
    "regulation_change": _regulation_change,
    "competitor_opened": _competitor_opened,
    "festival_upcoming": _festival_upcoming,
    "category_seasonal": _category_seasonal,
    "ipl_match_today": _ipl_match_today,
    "gbp_unverified": _gbp_unverified,
    "supply_alert": _supply_alert,
    "winback_eligible": _winback_eligible,
    "review_theme_emerged": _review_theme_emerged,
    "milestone_reached": _milestone_reached,
    "active_planning_intent": _active_planning_intent,
    "seasonal_perf_dip": _seasonal_perf_dip,
    "renewal_due": _renewal_due,
    "curious_ask_due": _curious_ask_due,
    "dormant_with_vera": _dormant_with_vera,
    "perf_dip": _perf_dip_spike,
    "perf_spike": _perf_dip_spike,
    "recall_due": _recall_due,
    "wedding_package_followup": _wedding_package_followup,
    "customer_lapsed_hard": _customer_lapsed_hard,
    "customer_lapsed_soft": _customer_lapsed_soft,
    "trial_followup": _trial_followup,
    "chronic_refill_due": _chronic_refill_due,
    "appointment_tomorrow": _appointment_tomorrow,
}


def compose_fallback(
    category: dict,
    merchant: dict,
    trigger: dict,
    customer: Optional[dict] = None,
    resolved_digest_item: Optional[dict] = None,
) -> dict:
    kind = trigger.get("kind", "")
    handler = KIND_HANDLERS.get(kind)
    if handler:
        return handler(category, merchant, trigger, customer, resolved_digest_item)

    # last-resort generic fallback for any kind not in the table above
    name = _first_name(merchant)
    sig = ", ".join(merchant.get("signals", [])[:2]) or "your account"
    body = f"{name}, quick note on {sig} — want me to take a look and suggest next steps?"
    return {"body": body, "cta": "open_ended", "send_as": "vera",
            "rationale": f"No specific template for trigger kind '{kind}'; used merchant's real signals rather than filler."}
