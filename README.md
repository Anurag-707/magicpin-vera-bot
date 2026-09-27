# Vera bot -- magicpin AI Challenge submission

A stateful FastAPI server implementing the 5-endpoint contract from
`challenge-testing-brief.md` §2 (`/v1/context`, `/v1/tick`, `/v1/reply`,
`/v1/healthz`, `/v1/metadata`, plus optional `/v1/teardown` per §11).

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# edit .env and paste your real GROQ_API_KEY
uvicorn bot:app --host 0.0.0.0 --port 8080
```

Self-test against the provided `judge_simulator.py` in a separate terminal
once the server above is running:

```bash
export BOT_URL=http://localhost:8080
python judge_simulator.py
```

## Files

- `bot.py` -- the FastAPI app. All 5 endpoints, including the multi-turn
  `/v1/reply` handler (auto-reply detection, hostile handling, intent
  handoff, graceful exit -- brief section 12's open challenges).
- `composer.py` -- LLM composition (Groq via LangChain). One prompt builder
  for first-touch proactive sends (`compose_proactive`, used by `/v1/tick`),
  and a shared builder for in-conversation replies (`compose_continue`,
  `compose_action_handoff`, used by `/v1/reply`).
- `detectors.py` -- fast, regex-based, non-LLM checks for hostility,
  known canned-reply phrasing, and explicit yes/no intent. Kept out of the
  LLM path entirely so these are cheap and deterministic.
- `state.py` -- in-memory context store (versioned, idempotent per the
  testing brief's `/v1/context` contract) + per-conversation state.
- `.env.example` -- copy to `.env` and put your real key there.

There is no separate `conversation_handlers.py` / `respond()` module. The
main brief's §7.4 describes that as an *optional* deliverable for a
batch-style submission; the testing brief's actual technical contract makes
`/v1/reply` itself the live multi-turn handler the judge calls, so a second
module reimplementing the same logic would just be a second copy to keep in
sync -- not additional capability.

## Approach

**Proactive composition (`/v1/tick`).** `compose_proactive` builds one
system prompt that serializes all four context layers (category, merchant,
trigger, optional customer) as JSON, plus a fixed rule list taken directly
from the brief (single verifiable fact, category voice, single CTA in the
last sentence, language match, no re-introduction, no repeating past
messages, correct `send_as` framing). A `TRIGGER_FRAMING` lookup table adds
trigger-specific guidance (e.g. `recall_due` -> offer concrete slots and a
binary CTA; `research_digest` -> lead with the most relevant digest item and
a soft ask) so one prompt skeleton adapts its shape per trigger kind. The
model returns JSON (`body`, `cta`, `rationale`); `send_as` and
`suppression_key` are set programmatically rather than trusted to the
model, since they're deterministic given the inputs. Temperature is pinned
to 0. A regex checks for more than one `"reply ..."` instruction in the
output body; if found, a single repair call rewrites it down to one CTA.

**Conversation continuation (`/v1/reply`).** Before any LLM call, three
cheap regex detectors run in order: hostility (`is_hostile`, exits
immediately), auto-reply, and explicit no/yes. Auto-reply detection is
layered: a fast path fires when text matches a known canned phrase
(`AUTO_REPLY_PATTERNS`) *and* has already been seen once before from this
sender (so it exits on the 2nd occurrence, one turn earlier than the
brief's "3+ times" hint -- directly targeting the brief's stated pain point
that production Vera "burns 2-3 turns" on this); a content-agnostic
fallback also fires if the exact same text repeats 3+ times regardless of
phrasing, since a real judge's canned text won't always match our keyword
list. Detection is keyed on `merchant_id` (not `conversation_id`), since a
canned WhatsApp Business reply can arrive under a fresh conversation ID
each time. Once a conversation is marked `exited`, any further `/v1/reply`
call for it short-circuits rather than resuming the pitch.

Once past the detectors, `compose_continue` / `compose_action_handoff`
(the intent-transition handoff) both receive the full `merchant`,
`category`, *and*, when the conversation is customer-facing, `customer`
context plus the correct `send_as` -- so a reply to a customer keeps their
name/language/preferences and the "don't mention Vera by name" rule, not
just the first message in the thread. Both return structured JSON
(`body` + `rationale`) rather than a bare string, so the rationale returned
to the judge is specific to that turn instead of a fixed placeholder.

**Reliability.** Every LLM call in both `/v1/tick` and `/v1/reply` is
wrapped in `asyncio.wait_for` (default 22s) so a slow Groq call degrades to
a fallback message instead of silently blowing the judge's 30s budget and
losing the whole turn/tick.

## Trade-offs / what would have helped

- No retrieval over `digest` / `patient_content_library` -- the whole
  category context is stuffed into the prompt. Fine at the sizes shown in
  the brief's examples; if a category's digest or content library is large
  in the real dataset, a retrieval step (embed digest items, pull the top
  1-2 relevant to the trigger) would keep prompts smaller and more focused.
- CTA-shape validation is a single regex heuristic (counts `"reply"`
  occurrences), so a multi-choice message that doesn't use the word "reply"
  (e.g. "Would you like whitening or aligners?") could still slip through
  as an undetected multi-CTA. A stricter pass -- asking the model to
  self-report its CTA count, or a small rule set per CTA type -- would catch
  more of these.
- No per-turn language *switch* detection (merchant flips English -> Hindi
  mid-conversation, brief section 12 challenge #4) -- `compose_continue`
  is told to match whatever language the prior bot messages used, not to
  re-detect language from the merchant's latest message.
- The WhatsApp 24h-session-window constraint (brief §5.1: first outbound
  must use a pre-approved template) is represented structurally
  (`template_name`/`template_params` are always populated in `/v1/tick`
  actions) but the bot doesn't currently distinguish "first-ever contact"
  from "later free-form message" in how it composes the body itself.
- Real `dataset/` + `test_pairs.json` would let this be tested against the
  actual category/merchant/trigger field shapes rather than the worked
  examples in the two briefs; field names may need small adjustments in
  `composer.py`'s prompt builders if the real JSON differs.

## Note on the provided `judge_simulator.py`

It ships with a hardcoded-looking Groq API key in the configuration
section. Worth rotating/removing regardless of whether it's a real,
currently-valid key, since it's now sitting in a shared file.
