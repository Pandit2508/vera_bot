#!/usr/bin/env python3
"""
generate_submission.py
=======================
Offline deliverable per challenge-brief.md §7.2: reads the 30 canonical
(merchant, trigger[, customer]) test pairs from the expanded dataset and
writes submission.jsonl, one composed message per line, using the exact same
compose() used by the live HTTP server (bot.py).

Usage:
    python generate_dataset.py --seed-dir dataset --out dataset/expanded   # once, to produce the expanded set
    python generate_submission.py --expanded-dir dataset/expanded --out submission.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from context_store import ContextStore
from composer import compose


def load_expanded(expanded_dir: Path) -> ContextStore:
    store = ContextStore()
    for f in sorted((expanded_dir / "categories").glob("*.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        store.push("category", data["slug"], 1, data)
    for f in sorted((expanded_dir / "merchants").glob("*.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        store.push("merchant", data["merchant_id"], 1, data)
    for f in sorted((expanded_dir / "customers").glob("*.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        store.push("customer", data["customer_id"], 1, data)
    for f in sorted((expanded_dir / "triggers").glob("*.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        store.push("trigger", data["id"], 1, data)
    return store


def line_used_fallback(line: dict) -> bool:
    """Whether this line was produced by the deterministic fallback composer rather
    than the real LLM. Relies on the explicit 'used_fallback' field composer.py now
    writes into every line. Older submission.jsonl files (generated before this fix)
    won't have this field — regenerate from scratch rather than trying to patch them,
    since there's no reliable way to tell fallback from real LLM output after the fact
    just from rationale text."""
    return bool(line.get("used_fallback", False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expanded-dir", default="dataset/expanded")
    ap.add_argument("--out", default="submission.jsonl")
    ap.add_argument(
        "--pace-seconds", type=float,
        default=float(os.environ.get("VERA_PACE_SECONDS", "0")),
        help="Sleep this long between LLM calls before each one. On Groq's free "
             "tier (8000 TPM, ~4000 tokens/call for this prompt), the account can "
             "only sustain roughly 1 call per 30s long-run -- but with --patient "
             "retries now built in (up to 6 retries / 65s each), a shorter pace "
             "like 4-5s plus retries usually gets there anyway, just slower on "
             "the calls that hit the limit. Set 0 (default) to rely entirely on "
             "retries, or ~30 for closer to zero retry-waiting."
    )
    ap.add_argument(
        "--retry-fallback-only", action="store_true",
        help="Read the existing --out file, find lines flagged used_fallback=true "
             "(LLM call failed/unavailable when they were generated), and "
             "regenerate ONLY those test pairs via the LLM -- merging the results "
             "back into the file in original order. Much cheaper than a full rerun "
             "when only a handful of calls got rate-limited."
    )
    args = ap.parse_args()

    expanded_dir = Path(args.expanded_dir)
    store = load_expanded(expanded_dir)
    pairs = json.loads((expanded_dir / "test_pairs.json").read_text(encoding="utf-8"))["pairs"]
    pairs_by_id = {p["test_id"]: p for p in pairs}

    out_path = Path(args.out)

    if args.retry_fallback_only:
        if not out_path.exists():
            print(f"[ERROR] --retry-fallback-only needs an existing {out_path} to patch. Run a full generation first.")
            return
        existing = [json.loads(l) for l in out_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        existing_by_id = {l["test_id"]: l for l in existing}
        if existing and not any("used_fallback" in l for l in existing):
            print(f"[WARN] {out_path} predates the 'used_fallback' field — can't reliably tell which lines "
                  f"were real LLM output vs. fallback. Recommend regenerating from scratch instead (drop "
                  f"--retry-fallback-only); the new retry logic should need far fewer fallbacks this time.")
            return
        to_retry = [tid for tid, line in existing_by_id.items() if line_used_fallback(line)]
        print(f"Found {len(to_retry)}/{len(existing)} lines using the fallback composer: {to_retry}")
        if not to_retry:
            print("Nothing to retry — every line already used the real LLM composer.")
            return
        for i, test_id in enumerate(to_retry):
            if i > 0 and args.pace_seconds > 0:
                time.sleep(args.pace_seconds)
            pair = pairs_by_id[test_id]
            bundle = store.resolve_bundle(
                pair["merchant_id"], trigger_id=pair.get("trigger_id"), customer_id=pair.get("customer_id")
            )
            result = compose(
                bundle["category"], bundle["merchant"], bundle["trigger"],
                bundle["customer"], bundle["resolved_digest_item"], patient=True,
            )
            existing_by_id[test_id] = {
                "test_id": test_id,
                "body": result["body"],
                "cta": result.get("cta", "open_ended"),
                "send_as": result.get("send_as", "vera"),
                "suppression_key": result.get("suppression_key", ""),
                "rationale": result.get("rationale", ""),
                "used_fallback": result.get("used_fallback", False),
            }
            still_fallback = existing_by_id[test_id]["used_fallback"]
            print(f"  [{'STILL FALLBACK' if still_fallback else 'OK — now LLM'}] {test_id}")

        ordered = [existing_by_id[p["test_id"]] for p in pairs if p["test_id"] in existing_by_id]
        with out_path.open("w", encoding="utf-8") as f:
            for line in ordered:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
        remaining = sum(1 for l in ordered if line_used_fallback(l))
        print(f"\nUpdated {out_path}. {remaining}/{len(ordered)} lines still on fallback.")
        return

    lines = []
    for i, pair in enumerate(pairs):
        if i > 0 and args.pace_seconds > 0:
            time.sleep(args.pace_seconds)
        test_id = pair["test_id"]
        bundle = store.resolve_bundle(
            pair["merchant_id"], trigger_id=pair.get("trigger_id"), customer_id=pair.get("customer_id")
        )
        if not (bundle["merchant"] and bundle["category"] and bundle["trigger"]):
            print(f"  [WARN] {test_id}: missing context, skipping "
                  f"(merchant={bool(bundle['merchant'])} category={bool(bundle['category'])} trigger={bool(bundle['trigger'])})")
            continue

        result = compose(
            bundle["category"], bundle["merchant"], bundle["trigger"],
            bundle["customer"], bundle["resolved_digest_item"], patient=True,
        )
        lines.append({
            "test_id": test_id,
            "body": result["body"],
            "cta": result.get("cta", "open_ended"),
            "send_as": result.get("send_as", "vera"),
            "suppression_key": result.get("suppression_key", ""),
            "rationale": result.get("rationale", ""),
            "used_fallback": result.get("used_fallback", False),
        })
        tag = "FALLBACK" if result.get("used_fallback", False) else "OK"
        print(f"  [{tag}] {test_id} ({pair['merchant_id']} / {pair.get('trigger_id')})")

    with out_path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    fallback_count = sum(1 for l in lines if l.get("used_fallback", False))
    print(f"\nWrote {len(lines)} lines to {out_path} ({fallback_count} used the fallback composer).")
    if fallback_count:
        print(f"Tip: run with --retry-fallback-only to regenerate just those {fallback_count} via the LLM.")


if __name__ == "__main__":
    main()
