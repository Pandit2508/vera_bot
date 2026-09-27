# Vera Rebuild — submission README

## Approach

The bot is a thin pipeline, not a giant prompt hack:

1. **`context_store.py`** — versioned in-memory store for the 4 context scopes, plus a
   `resolve_bundle()` that joins a trigger to its merchant, category, customer, and (for
   `research_digest`/`regulation_change` triggers) the actual digest item it points at via
   `payload.top_item_id`. The composer never sees raw wire-format cross-references — only fully
   resolved objects.
2. **`prompts.py`** — one system prompt that encodes the challenge's own rubric directly
   (specificity, category-voice matching, merchant personalization, trigger relevance, compulsion
   levers, and every anti-pattern in §11 of the brief) so a frontier model's *default* output
   already clears the bar, rather than hoping the model infers good judgment unprompted.
3. **`llm.py` / `composer.py`** — calls Claude (temperature 0, for the determinism the spec
   requires) with that prompt, parses the JSON, and runs one repair re-prompt if validation flags
   issues.
4. **`validators.py`** — a cheap, non-LLM pass that checks for the concrete anti-patterns called
   out in the brief: stacked CTAs, promotional hype tone in clinical categories, taboo vocabulary,
   verbatim repeats of prior messages, and unnecessary re-introductions.
5. **`conversation_handlers.py`** — the multi-turn layer, built specifically against the three
   Phase-4 replay scenarios in the testing brief:
   - **Auto-reply detection** combines content heuristics (canned-reply phrasing in English/Hindi)
     with repetition counting, so it can flag a likely auto-reply on the *first* occurrence and
     confirm-and-exit on the second — faster than the "2-3 turns burned" pattern called out as
     production Vera's weakness (brief §3, item 1).
   - **Intent transitions** ("let's do it", "haan karo") switch the state machine straight to an
     action-mode message, never back into qualifying questions (the Pattern-D failure in §9).
   - **Graceful exit** on explicit "not interested"/"stop" signals, and after
     `MAX_UNANSWERED_NUDGES` (3) ticks with no reply.
   - **Hostile/off-topic** messages get one polite redirect back to Vera's mission rather than
     engaging or refusing outright.
6. **`bot.py`** — wraps all of the above in the 5 HTTP endpoints from
   `challenge-testing-brief.md` §2 (`/v1/context`, `/v1/tick`, `/v1/reply`, `/v1/healthz`,
   `/v1/metadata`), with suppression-key dedup so the same trigger is never fired twice at the
   same merchant, and a `/v1/teardown` hook per §11's privacy requirement.
7. **`generate_submission.py`** — runs the identical `compose()` used by the live server against
   the 30 canonical test pairs to produce `submission.jsonl`, as a hedge in case the simpler
   offline format in `challenge-brief.md` §7 is what's actually scored instead of (or alongside)
   the live harness.

## Tradeoffs

- **Deterministic fallback composer** (`fallback_composer.py`): if `ANTHROPIC_API_KEY` isn't set,
  the whole pipeline still runs end-to-end using hand-written templates for **every trigger kind
  in the dataset** (all 24 seed kinds plus the 2 generator-only kinds), each of which degrades
  gracefully when the generator hands it a placeholder-only payload — it pulls real data instead
  from the merchant/customer/category objects (performance deltas, active offers, review themes,
  subscription status, relationship history) rather than ever printing placeholder junk or falling
  back to one generic filler line. Verified against all 100 expanded triggers: 0 validator flags,
  0 URLs, 0 crashes. **This still exists purely so the HTTP server, validators, and multi-turn
  logic are testable and demo-able without spending API credits — it is not meant to win the
  rubric on its own.** The `submission.jsonl` in this bundle was generated in fallback mode; a real
  submission needs `ANTHROPIC_API_KEY` set so `compose_via_llm()` actually runs, at which point
  every trigger kind gets a properly composed, rubric-optimized message instead of the template.
- **One repair re-prompt, not a validation loop.** We re-prompt once on a validation failure and
  keep whichever attempt has fewer flags, rather than looping — this stays well inside the 30s
  budget per `/v1/tick` action and the 30s reply SLA on `/v1/reply`.
- **Multi-turn intent/auto-reply/stop detection is rule-based, not another LLM call.** Once intent
  is clearly signaled, we don't want a second LLM round-trip to possibly re-ask a qualifying
  question — the state machine routes deterministically. Everything else in a live conversation
  still goes through the LLM composer with the running thread passed in, so it stays contextual.
- **Auto-reply handling is a 3-strike pattern** (nudge → wait 4h → end), matching the harness's
  own worked example in `examples/api-call-examples.md` (Ex 4.1) rather than exiting on the second
  occurrence — this costs one extra turn of patience but matches the documented "good bot"
  behavior exactly.
- **Off-topic/hostile redirects reference the live trigger thread**, not a generic "let's get back
  on track" — e.g. declining a GST-filing question and pointing back at "the JIDA piece" — per
  Ex 2.7. This needs the resolved trigger/digest item to be available at reply time.
- **Trigger recovery on `/v1/reply` is best-effort.** Our own `conversation_id` format
  (`conv_{merchant}_{trigger}_{rand}`) embeds the trigger id so we can look it up; if the judge
  ever calls `/v1/reply` for a conversation we didn't originate, we fall back to a minimal
  synthetic trigger rather than failing the call.
- **No URLs, ever.** `challenge-brief.md` §5.4 says URLs are "allowed when they add clear value,"
  but `examples/api-call-examples.md` (Ex F.4) says a URL in the body is a hard -3 penalty because
  Meta would reject it. We resolved the conflict conservatively: templates ask "want me to send you
  the link?" rather than including one. Verified: 0 URLs across all 100 composed triggers.
- **Free-tier rate limits get a real retry budget, tuned per caller.** `llm.py` supports both
  Anthropic and Groq (OpenAI-compatible endpoint); on Groq's free tier (8000 TPM, ~4000
  tokens/call for this prompt), a burst of calls hits 429s fast. `compose_via_llm(..., patient=)`
  splits the retry budget by caller: `patient=False` (default; used by `bot.py`'s live `/v1/tick`
  and `/v1/reply` handlers) fails fast — max 2 retries / 10s each — so a single call can never
  blow the 30s SLA in `challenge-testing-brief.md`. `patient=True` (used only by the offline
  `generate_submission.py`, which has no real-time constraint) allows up to 6 retries / 65s each,
  long enough to ride out a full rolling-60s TPM window. The wait-time parser also handles both of
  Groq's error formats (`"try again in 14.1s"` and `"...in 300ms"` — the first version only matched
  the former).
- **`used_fallback` is an explicit field on every composed message**, not inferred from rationale
  text (rationale is free-form LLM/template prose and too fragile to string-match reliably).
  `generate_submission.py --retry-fallback-only` reads this field to patch just the handful of
  lines that hit a rate limit, instead of re-running and re-paying for all 30. It refuses to guess
  on a `submission.jsonl` written before this field existed — regenerate from scratch instead.

## What additional context would have helped most

- **A stable `digest item -> category` reverse index** in the wire payloads (right now we resolve
  `top_item_id` by scanning `category.digest`, which is fine at this scale but wouldn't be at
  production volume).
- **Explicit merchant reply-rate / preferred-contact-hour signals**, to make cadence decisions in
  `/v1/tick` (currently we send once per available trigger per tick and rely on suppression keys
  for restraint, but a "don't message before 10am" or "prefers evenings" signal would sharpen
  timing).
- **A ground-truth label on which `conversation_history` entries are confirmed auto-replies**,
  to validate the detection heuristic against real production examples rather than the two
  Hindi/English phrase lists we hand-wrote from the brief's own excerpts.

## Running it

```bash
pip install fastapi uvicorn
export ANTHROPIC_API_KEY=sk-ant-...   # preferred — best output quality
# or, as a free alternative:
export GROQ_API_KEY=gsk_...           # picked up automatically if ANTHROPIC_API_KEY is unset
# omit both to run in deterministic fallback mode (for testing the plumbing only)
uvicorn bot:app --host 0.0.0.0 --port 8080
```

Offline submission file:
```bash
python dataset/generate_dataset.py --seed-dir dataset --out dataset/expanded
python generate_submission.py --expanded-dir dataset/expanded --out submission.jsonl
# on a tight free-tier rate limit, space out the calls:
python generate_submission.py --expanded-dir dataset/expanded --out submission.jsonl --pace-seconds 5
# if only a few lines hit the rate limit, patch just those instead of a full rerun:
python generate_submission.py --expanded-dir dataset/expanded --out submission.jsonl --retry-fallback-only
```

Self-test against the provided harness simulator:
```bash
export BOT_URL=http://localhost:8080
python judge_simulator.py
```
