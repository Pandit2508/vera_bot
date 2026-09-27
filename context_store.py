"""
context_store.py
=================
Holds category / merchant / customer / trigger context in memory, keyed the
same way the testing brief's /v1/context endpoint expects: (scope, context_id) -> {version, payload}.

Also resolves cross-references, e.g. a trigger's payload.top_item_id needs to
be looked up inside the category's digest list to get the actual research item.
This "resolved" view is what we hand to the composer, so the composer never
has to know about raw wire-format quirks.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional


class ContextStore:
    """In-memory, versioned context store. Thread-unsafe by design (single-process demo)."""

    def __init__(self):
        # (scope, context_id) -> {"version": int, "payload": dict}
        self._store: dict[tuple[str, str], dict] = {}

    # ---- write path (mirrors POST /v1/context semantics) -----------------
    def push(self, scope: str, context_id: str, version: int, payload: dict) -> dict:
        key = (scope, context_id)
        cur = self._store.get(key)
        if cur and cur["version"] >= version:
            return {"accepted": False, "reason": "stale_version", "current_version": cur["version"]}
        self._store[key] = {"version": version, "payload": payload}
        return {"accepted": True, "current_version": version}

    # ---- read path ---------------------------------------------------------
    def get(self, scope: str, context_id: str) -> Optional[dict]:
        entry = self._store.get((scope, context_id))
        return entry["payload"] if entry else None

    def counts(self) -> dict:
        out: dict[str, int] = {}
        for (scope, _cid) in self._store.keys():
            out[scope] = out.get(scope, 0) + 1
        return out

    def all_of(self, scope: str) -> list[dict]:
        return [v["payload"] for (s, _cid), v in self._store.items() if s == scope]

    def trigger_ids(self) -> list[str]:
        return [cid for (scope, cid) in self._store.keys() if scope == "trigger"]

    # ---- resolution helpers -------------------------------------------------
    def resolve_digest_item(self, category_slug: str, item_id: str) -> Optional[dict]:
        cat = self.get("category", category_slug)
        if not cat:
            return None
        for item in cat.get("digest", []):
            if item.get("id") == item_id:
                return item
        return None

    def resolve_bundle(
        self, merchant_id: str, trigger_id: Optional[str] = None, customer_id: Optional[str] = None
    ) -> dict:
        """Assemble the full (category, merchant, trigger, customer) bundle the composer needs."""
        merchant = self.get("merchant", merchant_id)
        category = self.get("category", merchant["category_slug"]) if merchant else None
        trigger = self.get("trigger", trigger_id) if trigger_id else None
        customer = self.get("customer", customer_id) if customer_id else None

        resolved_digest_item = None
        if trigger and category:
            item_id = trigger.get("payload", {}).get("top_item_id") or trigger.get("payload", {}).get("digest_item_id")
            if item_id:
                resolved_digest_item = self.resolve_digest_item(category["slug"], item_id)

        return {
            "category": category,
            "merchant": merchant,
            "trigger": trigger,
            "customer": customer,
            "resolved_digest_item": resolved_digest_item,
        }


# --------------------------------------------------------------------------
# Bulk loader for the flat seed files shipped in dataset/ (used for local
# testing / offline submission.jsonl generation, not part of the live server
# wire protocol).
# --------------------------------------------------------------------------

def load_seed_dataset(dataset_dir: str | Path) -> ContextStore:
    dataset_dir = Path(dataset_dir)
    store = ContextStore()

    for f in sorted((dataset_dir / "categories").glob("*.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        store.push("category", data["slug"], version=1, payload=data)

    merchants = json.loads((dataset_dir / "merchants_seed.json").read_text(encoding="utf-8"))["merchants"]
    for m in merchants:
        store.push("merchant", m["merchant_id"], version=1, payload=m)

    customers = json.loads((dataset_dir / "customers_seed.json").read_text(encoding="utf-8"))["customers"]
    for c in customers:
        store.push("customer", c["customer_id"], version=1, payload=c)

    triggers = json.loads((dataset_dir / "triggers_seed.json").read_text(encoding="utf-8"))["triggers"]
    for t in triggers:
        store.push("trigger", t["id"], version=1, payload=t)

    return store
