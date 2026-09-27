"""
Vera bot HTTP server -- implements the 5 endpoints from
challenge-testing-brief.md section 2, backed by composer.py for LLM
composition and detectors.py for fast pattern-based conversation handling
(auto-reply, hostile, intent transition).

Multi-turn handling (brief section 7.4 / section 12) lives directly in
/v1/reply below rather than in a separate conversation_handlers.py module:
testing-brief.md's HTTP contract makes /v1/reply the actual live multi-turn
path the judge calls, so a second, standalone "respond()" module would just
be a second (and easy to let drift out of sync) copy of the same logic.

Run:
    uvicorn bot:app --host 0.0.0.0 --port 8080

Self-test against the provided judge_simulator.py (in a separate terminal
once the server above is running):
    export BOT_URL=http://localhost:8080
    python judge_simulator.py
"""

import os
import time
import asyncio
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from state import store
import composer
import detectors

app = FastAPI(title="Vera Bot - magicpin AI Challenge")

VALID_SCOPES = {"category", "merchant", "customer", "trigger"}

# Per-call budget for LLM-backed composition. NOTE: judge_simulator.py's
# actual local HTTP client uses a 15s timeout on both /v1/tick and
# /v1/reply (JudgeClient.tick / .reply) -- tighter than the brief's nominal
# 30s. This constant used to be 22s, which combined with composer.py's own
# 25s ChatGroq timeout meant a slow call could structurally never finish
# before the judge gave up and logged "timed out". Sized here with real
# headroom under the judge's 15s so FastAPI/serialization overhead can't
# push us over even in the worst case.
LLM_CALL_TIMEOUT = 10

# Wall-clock ceiling for the WHOLE /v1/tick handler, regardless of how many
# triggers are being composed. Belt-and-suspenders on top of per-call
# timeouts: if anything unexpected stalls (thread-pool contention from a
# still-running zombie call, etc.), we still answer the judge on time with
# whatever's ready rather than risk the whole batch reading as "timed out".
TICK_DEADLINE = 13

# Self-limit: even if the judge hands us up to 20 available_triggers in one
# /v1/tick call, we deliberately only *compose* for the top few by urgency.
# The testing brief explicitly sanctions this ("the bot can use any subset
# (or none) of available_triggers"; "Restraint is rewarded; spam is
# penalized"). This also keeps concurrent Groq usage per tick comfortably
# under the shared 8,000 tokens/minute budget instead of firing everything
# at once and 429-ing most of it.
MAX_TRIGGERS_PER_TICK = 3


# ---------------------------------------------------------------------------
# 2.4 / 2.5 -- health + metadata
# ---------------------------------------------------------------------------

@app.get("/v1/healthz")
async def healthz():
    return {
        "status": "ok",
        "uptime_seconds": int(time.time() - store.start_time),
        "contexts_loaded": store.contexts_loaded_counts(),
    }


@app.get("/v1/metadata")
async def metadata():
    return {
        "team_name": os.environ.get("TEAM_NAME", "Solo Team"),
        "team_members": [n for n in os.environ.get("TEAM_MEMBERS", "").split(",") if n],
        "model": composer.GROQ_MODEL,
        "approach": (
            "Single-prompt composer per trigger (full category/merchant/trigger/customer JSON "
            "serialized into the prompt); regex-based fast-path detectors for auto-reply, "
            "hostility, and explicit intent ahead of an LLM fallback for continuing conversations. "
            "Customer context and send_as (vera vs merchant_on_behalf) are threaded through every "
            "reply, not just the first proactive send."
        ),
        "contact_email": os.environ.get("CONTACT_EMAIL", "team@example.com"),
        "version": "0.2.0",
        "submitted_at": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# 2.1 -- context push
# ---------------------------------------------------------------------------

class ContextBody(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: dict[str, Any]
    delivered_at: str


@app.post("/v1/context")
async def push_context(body: ContextBody):
    if body.scope not in VALID_SCOPES:
        return JSONResponse(status_code=400, content={
            "accepted": False, "reason": "invalid_scope",
            "details": f"scope must be one of {sorted(VALID_SCOPES)}, got '{body.scope}'",
        })

    result, version = store.push_context(body.scope, body.context_id, body.version, body.payload)
    if result == "stale":
        return JSONResponse(status_code=409, content={
            "accepted": False, "reason": "stale_version", "current_version": version,
        })

    return {
        "accepted": True,
        "ack_id": f"ack_{body.context_id}_v{body.version}",
        "stored_at": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# 2.2 -- tick (proactive sends)
# ---------------------------------------------------------------------------

class TickBody(BaseModel):
    now: str
    available_triggers: list[str] = []


def _is_expired(trigger: dict, now_iso: str) -> bool:
    expires_at = trigger.get("expires_at")
    if not expires_at:
        return False
    try:
        now = datetime.fromisoformat(now_iso.replace("Z", "+00:00"))
        expiry = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
        return now > expiry
    except Exception:
        return False


# Groq's free tier caps openai/gpt-oss-120b at 8,000 tokens/MINUTE (not just
# 30 requests/minute) -- see https://console.groq.com/docs/rate-limits.
# Previously this was Semaphore(2), which serialized a 5-trigger batch into
# ~3 sequential rounds -- and each round could itself take up to the old
# 22s per-call timeout, guaranteeing the WHOLE tick blew past the judge's
# real 15s client timeout long before all triggers were even attempted.
# Now that MAX_TRIGGERS_PER_TICK caps what we actually attempt per tick to a
# small, urgency-ranked subset, we can afford to run that whole subset
# concurrently in a single round instead of serializing it -- one round of
# MAX_TRIGGERS_PER_TICK prompts (~2-2.5K tokens each) still comfortably fits
# the 8K TPM budget, and finishes in one LLM_CALL_TIMEOUT window rather than
# several stacked ones.
_LLM_CONCURRENCY = asyncio.Semaphore(MAX_TRIGGERS_PER_TICK)


async def _build_action_for_trigger(trigger_id: str, now_iso: str) -> Optional[dict]:
    trigger = store.get_context("trigger", trigger_id)
    if not trigger or _is_expired(trigger, now_iso):
        return None

    suppression_key = trigger.get("suppression_key")
    if store.already_sent(suppression_key):
        return None

    merchant_id = trigger.get("merchant_id")
    merchant = store.get_context("merchant", merchant_id)
    if not merchant:
        return None

    category = store.get_context("category", merchant.get("category_slug"))
    if not category:
        return None

    customer = None
    customer_id = trigger.get("customer_id")
    if trigger.get("scope") == "customer" and customer_id:
        customer = store.get_context("customer", customer_id)

    try:
        async with _LLM_CONCURRENCY:
            composed = await asyncio.wait_for(
                asyncio.to_thread(composer.compose_proactive, category, merchant, trigger, customer),
                timeout=LLM_CALL_TIMEOUT,
            )
    except Exception:
        # Never let one slow/bad trigger break the whole tick -- this
        # trigger just doesn't produce an action this cycle.
        return None

    conversation_id = f"conv_{merchant_id}_{trigger_id}"
    conv = store.get_or_create_conversation(conversation_id, merchant_id, customer_id)
    conv.bot_messages.append(composed["body"])
    store.mark_sent(suppression_key)

    merchant_name = (merchant.get("identity", {}) or {}).get("name", merchant_id)

    return {
        "conversation_id": conversation_id,
        "merchant_id": merchant_id,
        "customer_id": customer_id,
        "send_as": composed["send_as"],
        "trigger_id": trigger_id,
        "template_name": f"vera_{trigger.get('kind', 'generic')}_v1",
        "template_params": [merchant_name, trigger.get("kind", "")],
        "body": composed["body"],
        "cta": composed["cta"],
        "suppression_key": composed["suppression_key"],
        "rationale": composed["rationale"],
    }


def _is_actionable(trigger_id: str, now_iso: str) -> bool:
    """Cheap, local, no-LLM-call check: does this trigger even have a chance
    of producing an action? Used to filter BEFORE ranking/selecting the top
    MAX_TRIGGERS_PER_TICK -- otherwise an expired-but-high-urgency trigger
    can occupy one of our limited slots and crowd out a genuinely actionable
    lower-urgency one (this is what capped batch 1 at 1 action instead of 2:
    trg_004/trg_005 -- both urgency 4 but both expired -- filled the top-3
    ahead of trg_003, which was still valid)."""
    t = store.get_context("trigger", trigger_id)
    if not t or _is_expired(t, now_iso):
        return False
    return not store.already_sent(t.get("suppression_key"))


def _rank_by_urgency(trigger_ids: list[str]) -> list[str]:
    """Sort by the trigger's own declared urgency (descending)."""
    def urgency(tid: str) -> int:
        t = store.get_context("trigger", tid)
        return t.get("urgency", 0) if t else -1
    return sorted(trigger_ids, key=urgency, reverse=True)


@app.post("/v1/tick")
async def tick(body: TickBody):
    trigger_ids = list(dict.fromkeys(body.available_triggers))[:20]  # dedupe, cap at 20/tick

    # Filter to genuinely actionable triggers FIRST (cheap, local, no LLM
    # calls), THEN rank by urgency and take the self-limited top slice --
    # so our limited MAX_TRIGGERS_PER_TICK slots never go to a trigger
    # that was going to be rejected anyway (expired / already sent).
    actionable = [tid for tid in trigger_ids if _is_actionable(tid, body.now)]
    selected = _rank_by_urgency(actionable)[:MAX_TRIGGERS_PER_TICK]

    tasks = [asyncio.ensure_future(_build_action_for_trigger(tid, body.now)) for tid in selected]
    if tasks:
        done, pending = await asyncio.wait(tasks, timeout=TICK_DEADLINE)
        for p in pending:
            # Can't truly kill the underlying thread if one is stuck inside
            # it, but this stops us from waiting on it and lets the tick
            # respond to the judge on time regardless.
            p.cancel()
        results = [t.result() for t in done if not t.cancelled() and t.exception() is None]
    else:
        results = []

    actions = [a for a in results if a is not None][:20]
    return {"actions": actions}


# ---------------------------------------------------------------------------
# 2.3 -- reply (continue a conversation)
# ---------------------------------------------------------------------------

class ReplyBody(BaseModel):
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    from_role: str
    message: str
    received_at: str
    turn_number: int


@app.post("/v1/reply")
async def reply(body: ReplyBody):
    conv = store.get_or_create_conversation(body.conversation_id, body.merchant_id, body.customer_id)
    merchant_id = body.merchant_id or conv.merchant_id
    customer_id = body.customer_id or conv.customer_id
    message = body.message

    # 0. Conversation already ended -- don't let a further call resume
    #    pitching after we've already said goodbye.
    if conv.mode == "exited":
        return {"action": "end", "rationale": "Conversation was already ended; not resuming it."}

    # 1. Hostile -> end immediately, no apology loop, no further pitching
    if detectors.is_hostile(message):
        conv.mode = "exited"
        store.record_inbound(merchant_id, message)
        conv.merchant_messages.append(message)
        return {
            "action": "end",
            "rationale": "Message is hostile/abusive; ending rather than continuing to pitch.",
        }

    # 2. Auto-reply detection -- keyed on merchant_id, not conversation_id,
    #    since a canned WhatsApp Business reply can arrive under a fresh
    #    conversation_id each time. Two tripwires:
    #      a) fast path -- text matches a known canned phrase AND has
    #         already been seen once before (i.e. this is occurrence #2).
    #         Ends one turn earlier than waiting for a 3rd repeat, which is
    #         the brief's own stated opportunity ("better detection + faster
    #         routing wins").
    #      b) content-agnostic fallback -- doesn't match any known phrase,
    #         but is the exact same text verbatim for the 3rd+ time. This is
    #         the brief's literal hint, and catches canned replies whose
    #         wording isn't in our keyword list at all.
    prior_occurrences = store.inbound_occurrences(merchant_id, message)
    looks_canned = detectors.is_auto_reply_pattern(message)
    store.record_inbound(merchant_id, message)
    conv.merchant_messages.append(message)

    if (looks_canned and prior_occurrences >= 1) or detectors.is_pure_repeat_auto_reply(prior_occurrences):
        conv.mode = "exited"
        reason = (
            "Same canned reply seen again from this sender; confirmed auto-reply, exiting instead "
            "of burning more turns." if looks_canned else
            "Exact same message seen verbatim 3+ times; treating as an auto-reply even though it "
            "doesn't match a known canned phrase."
        )
        return {"action": "end", "rationale": reason}

    if looks_canned:
        nudge = ("Samajh gayi, yeh automated reply lag raha hai. Agar aap khud dekhna chahein ki "
                 "exact kya missing hai, 2 minute ka kaam hai -- chalega?")
        conv.bot_messages.append(nudge)
        return {
            "action": "send", "body": nudge, "cta": "open_ended",
            "rationale": "First occurrence of a canned-looking reply; trying once more before assuming auto-reply.",
        }

    # 3. Explicit not-interested -> exit gracefully
    if detectors.is_explicit_no(message):
        conv.mode = "exited"
        farewell = "Koi baat nahi, samajh gayi. Zaroorat padne par main yahin hoon. All the best!"
        conv.bot_messages.append(farewell)
        return {
            "action": "send", "body": farewell, "cta": "none",
            "rationale": "Sender signaled not interested; exiting politely without further pitching.",
        }

    merchant = store.get_context("merchant", merchant_id) if merchant_id else None
    category = store.get_context("category", merchant.get("category_slug")) if merchant else None
    customer = store.get_context("customer", customer_id) if customer_id else None
    send_as = "merchant_on_behalf" if customer_id else "vera"

    # 4. Explicit yes/intent while still pitching -> jump straight to action
    if detectors.is_explicit_yes(message) and conv.mode == "pitch":
        conv.mode = "action"
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(composer.compose_action_handoff, merchant, category,
                                   customer, send_as, conv.bot_messages, message),
                timeout=LLM_CALL_TIMEOUT,
            )
        except Exception:
            result = {"body": "Done -- starting that now, I'll confirm here as soon as it's set up.",
                      "rationale": "Fallback used after handoff composition failed or timed out."}
        conv.bot_messages.append(result["body"])
        return {
            "action": "send", "body": result["body"], "cta": "open_ended",
            "rationale": result.get("rationale") or "Detected explicit intent; routed straight to action instead of re-qualifying.",
        }

    # 5. Normal engaged reply -- continue naturally via LLM
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(composer.compose_continue, merchant, category,
                               customer, send_as, conv.bot_messages, message),
            timeout=LLM_CALL_TIMEOUT,
        )
    except Exception:
        result = {"body": "Got it -- give me a moment and I'll come back with the details.",
                  "rationale": "Fallback used after continuation composition failed or timed out."}
    conv.bot_messages.append(result["body"])
    return {"action": "send", "body": result["body"], "cta": "open_ended",
            "rationale": result.get("rationale") or "Continuing an engaged conversation."}


# ---------------------------------------------------------------------------
# Optional teardown (testing brief section 11 -- privacy)
# ---------------------------------------------------------------------------

@app.post("/v1/teardown")
async def teardown():
    store.wipe()
    return {"status": "wiped"}
