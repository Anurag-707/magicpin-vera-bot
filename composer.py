"""
LLM composition logic for the Vera bot -- shared by /v1/tick (new proactive
messages) and /v1/reply (continuing an existing conversation, once past the
fast regex detectors in detectors.py).

Uses Groq via LangChain.
"""

import os
import re
import json
import time
from datetime import datetime, timezone
from typing import Optional

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
# NOTE ON THIS NUMBER: the brief says the judge waits 30s/call, but
# judge_simulator.py's actual local HTTP client uses a 15s timeout for both
# /v1/tick and /v1/reply (see JudgeClient.tick / .reply). This constant --
# and bot.py's LLM_CALL_TIMEOUT -- are sized against that tighter, real
# constraint, not the brief's nominal number. Previously this was 25s
# (bigger than the judge's own client timeout!), which meant a slow call
# could NEVER finish before the judge gave up on it.
LLM_TIMEOUT_SECONDS = int(os.environ.get("LLM_TIMEOUT_SECONDS", "9"))

_llm = None


def get_llm():
    global _llm
    if _llm is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found. Put your real key in .env.")
        try:
            # Bound the underlying HTTP call so a slow/hanging Groq request
            # can't blow past the judge's real (15s local / 30s brief)
            # per-call budget on its own; bot.py additionally wraps calls in
            # asyncio.wait_for as a second, app-level backstop.
            _llm = ChatGroq(groq_api_key=api_key, model_name=GROQ_MODEL,
                             temperature=0, timeout=LLM_TIMEOUT_SECONDS)
        except TypeError:
            # Older langchain-groq versions may not accept `timeout` here --
            # fall back rather than crash the whole bot on startup.
            _llm = ChatGroq(groq_api_key=api_key, model_name=GROQ_MODEL, temperature=0)
    return _llm


def _invoke_with_retry(llm, messages, max_retries: int = 1, base_delay: float = 0.0):
    """llm.invoke() wrapper.

    IMPORTANT: this used to retry a 429 with time.sleep(4)/sleep(8)/sleep(16)
    -- up to 28s of *real, blocking* sleep inside the worker thread that
    asyncio.to_thread() spawned for this call. That was actively harmful
    under our actual constraints:

      1. The judge's client-side timeout (15s locally) is shorter than that
         retry budget on its own, so a retried call could never finish in
         time to matter -- bot.py's asyncio.wait_for(...) would already have
         given up and returned to the judge.
      2. Cancelling the *asyncio* task that awaits a to_thread() call does
         NOT stop the underlying OS thread -- Python threads aren't
         preemptible. So even after bot.py "gave up" on a trigger, that
         thread kept sleeping and then retrying against Groq in the
         background, continuing to burn the shared 8,000 TPM budget for a
         result nobody would ever use. That's what turned one rate-limited
         tick batch into several subsequent batches also failing (0 actions
         in single-digit ms -- an instant 429, not a real timeout).

    Given a 429 typically needs several real seconds for the token bucket to
    refill anyway, retrying inside our tight per-call budget can't succeed
    -- it can only cost us. So: single attempt, no sleep. bot.py's fallback
    message (per call site) is the actual recovery path when this fails.
    max_retries/base_delay are kept as parameters (rather than deleted) so a
    deployment with a paid Groq tier / looser timeout budget can re-enable
    real backoff via env-driven config without an API change.
    """
    last_err = None
    for attempt in range(max_retries):
        try:
            return llm.invoke(messages)
        except Exception as e:
            last_err = e
            if attempt == max_retries - 1:
                raise
            if base_delay:
                time.sleep(base_delay * (2 ** attempt))
    raise last_err


# Per-trigger-kind framing hints, covering every kind that appears in
# triggers_seed.json (+ the generator's additional_kinds list), so /v1/tick
# has sensible guidance no matter which trigger fires.
TRIGGER_FRAMING = {
    "research_digest": "Lead with the single most relevant digest item; cite the source inline; invite a soft yes/no, not a hard ask.",
    "regulation_change": "State the specific regulation change plainly, cite source, no alarmism, offer to help comply/communicate it.",
    "recall_due": "Action/booking trigger. Offer concrete open slots and/or the catalog price. Single binary or short-choice CTA (reply 1/2).",
    "perf_dip": "State the real numeric drop plainly, no alarmism. One concrete next step. Loss-aversion framing.",
    "renewal_due": "State the real days-remaining and renewal amount plainly. Single clear CTA to renew.",
    "festival_upcoming": "Anchor on the specific festival + date. Offer to draft a themed post/offer.",
    "wedding_package_followup": "Anchor on the specific wedding date and days-to-wedding. Offer the concrete next-step program.",
    "curious_ask_due": "Curiosity/knowledge angle -- ask the merchant something specific about their business. No hard CTA.",
    "winback_eligible": "Loss-aversion framing on the concrete performance drop since expiry. Single clear renew CTA.",
    "ipl_match_today": "Anchor on the specific match + venue + time. Tie to a timely offer/post.",
    "review_theme_emerged": "Surface the specific recurring review theme (with count/quote) and offer a concrete fix or draft response.",
    "milestone_reached": "Congratulate on the specific number reached, note how close to the next milestone.",
    "active_planning_intent": "Merchant is already engaged and planning something concrete -- give the next concrete step, not another qualifying question.",
    "seasonal_perf_dip": "Note the dip is a known seasonal pattern; reassure briefly, then offer one concrete lever to counter it.",
    "customer_lapsed_hard": "Win-back framing referencing their prior program/focus. Single clear re-engagement CTA.",
    "trial_followup": "Customer-facing. Offer the concrete next session slot. Binary confirm CTA.",
    "supply_alert": "High urgency, plain and clear. State the specific batches/molecule. Offer to filter/notify affected customers.",
    "chronic_refill_due": "Customer-facing. State the specific medicines and when supply runs out. Concrete delivery offer, binary CTA.",
    "category_seasonal": "Anchor on the specific seasonal trend stats. Offer one concrete shelf/stock action.",
    "gbp_unverified": "State the concrete uplift estimate from verifying. One clear CTA to start verification.",
    "cde_opportunity": "Low-urgency FYI framing. Note the credits/cost. Soft CTA only.",
    "competitor_opened": "State the specific competitor fact neutrally. Pivot to what the merchant can control.",
    "perf_spike": "Celebrate briefly with the real number, then pivot to compounding it. Momentum framing.",
    "dormant_with_vera": "Light, low-pressure check-in referencing the last topic. One specific, low-effort question. No hard CTA.",
    "customer_lapsed_soft": "Customer-facing recall message. Concrete slots/price. Binary or short-choice CTA.",
    "appointment_tomorrow": "Customer-facing reminder. Confirm date/time plainly. Binary confirm CTA.",
}


def _default_suppression_key(trigger: dict, merchant: dict) -> str:
    date_key = datetime.now(timezone.utc).strftime("%Y-%W")
    return f"{trigger.get('kind', 'unknown')}:{merchant.get('merchant_id', 'unknown')}:{date_key}"


def _language_hint(merchant: Optional[dict], customer: Optional[dict]) -> str:
    lang = None
    if customer:
        prefs = customer.get("preferences", {}) or {}
        ident = customer.get("identity", {}) or {}
        lang = prefs.get("language_pref") or ident.get("language_pref")
    if not lang:
        languages = ((merchant or {}).get("identity", {}) or {}).get("languages")
        if languages:
            lang = ", ".join(languages) if isinstance(languages, list) else str(languages)
    if not lang:
        lang = "hi-en mix (Hindi-English code-mix), unless prior conversation_history shows English-only replies"

    # Every worked example in challenge-brief.md ("Apke liye 2 slots ready
    # hain", "Aapka Google profile abhi 62.5% complete hai") is Romanized
    # Hinglish, never Devanagari -- but a bare "hi-en mix" string alone
    # doesn't say that, and the model sometimes guesses Devanagari, which
    # reads as an off-voice miss (this is what dropped Merchant Fit to 5/10
    # on the Priya recall message).
    if lang.strip().lower() not in ("english", "en"):
        lang += (" -- Romanized/Latin script code-mix (Hinglish: 'aapke liye', 'chalega?'), "
                  "never Devanagari script")
    return lang


def _clean_llm_text(text: str) -> str:
    """Strip code fences and a single layer of wrapping quotes some models
    add even when asked for raw text / raw JSON."""
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ('"', "'"):
        text = text[1:-1].strip()
    return text


def _extract_json(text: str) -> dict:
    text = _clean_llm_text(text)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"Could not find JSON in model output: {text[:200]}")
    return json.loads(match.group(0))


def _count_ctas(body: str) -> int:
    return len(re.findall(r"\breply\b", body, re.IGNORECASE))


_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")


def _has_devanagari(text: str) -> bool:
    return bool(_DEVANAGARI_RE.search(text))


def _romanize_if_needed(llm, body: str) -> str:
    """Every worked example in the brief renders Hinglish in Romanized/Latin
    script ("Apke liye 2 slots ready hain"), never Devanagari -- but the
    model sometimes ignores the prompt's language-hint instruction and
    writes Devanagari anyway (this is what dropped Merchant Fit to 5/10 on
    the Priya recall message: "...Aapke pa..." became "...\u0906\u092a\u0915\u0947 \u092a\u093e...").
    Prompt instructions alone aren't reliable enough, so this is a
    deterministic post-check with a single, cheap repair call -- same
    pattern as the CTA-count fix below -- rather than trusting the model to
    have followed rule 4 on the first try."""
    if not _has_devanagari(body):
        return body
    fix_prompt = (
        "This WhatsApp message contains Devanagari script. Rewrite it using ONLY Latin/Roman "
        "alphabet characters (Romanized Hinglish, e.g. 'aapke liye', 'chalega?') -- never "
        "Devanagari -- keeping the meaning, tone and CTA identical. Return ONLY the rewritten "
        "message body, no quotes, no commentary:\n\n" + body
    )
    try:
        fixed = _clean_llm_text(_invoke_with_retry(llm, [HumanMessage(content=fix_prompt)]).content)
        return fixed if fixed and not _has_devanagari(fixed) else body
    except Exception:
        return body


def _compact_json(obj, max_chars: int = 2500) -> str:
    """Compact (no indent) JSON, truncated. indent=2 alone costs real tokens
    on nothing but whitespace, and the proactive prompt previously dumped
    category+merchant+trigger(+customer) with NO cap at all -- unlike
    _reply_prompt, which already truncates. On Groq's free tier,
    openai/gpt-oss-120b is capped at 8,000 tokens/MINUTE (not just
    requests/minute); a tick batch fires several of these concurrently, so
    oversized prompts are what actually exhausts the TPM budget mid-batch."""
    text = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    if len(text) > max_chars:
        text = text[:max_chars] + "...(truncated)"
    return text


def _build_proactive_system_prompt(category: dict, merchant: dict, trigger: dict,
                                    customer: Optional[dict]) -> str:
    framing = TRIGGER_FRAMING.get(
        trigger.get("kind", ""), "Use the single most relevant fact from the contexts as the hook."
    )
    lang_hint = _language_hint(merchant, customer)
    send_as = "merchant_on_behalf" if customer else "vera"

    parts = [
        "You are composing ONE outbound WhatsApp message for magicpin's merchant-AI assistant, Vera.",
        "Rules (violating any of these is a failure):",
        "1. Anchor on a concrete, verifiable fact from the context below (a number, date, headline, or stat). Never invent data not present in the context.",
        "2. Match the category voice exactly. No hype, no ALL CAPS, no exclamation-heavy promo copy unless the category voice explicitly allows it.",
        "3. Personalize to this specific merchant's real numbers, offers, and conversation history. Nothing generic.",
        f"4. Language: write in {lang_hint}.",
        "5. Exactly ONE call-to-action, landing in the last sentence. Binary (reply YES/STOP, or reply 1/2) for action triggers; no hard CTA for pure-information triggers -- a soft open-ended question is fine instead.",
        "6. No preambles. Don't re-introduce yourself if conversation_history shows prior turns.",
        "7. Never repeat a message verbatim that already appears in conversation_history.",
        f"8. send_as = {send_as}."
        + (" This message goes FROM the merchant TO their own customer -- do not mention Vera or magicpin by name in the body."
           if send_as == "merchant_on_behalf"
           else " This message goes FROM Vera TO the merchant."),
        f"9. Trigger-specific guidance: {framing}",
        (
            "10. Engagement: prefer a lever that doesn't require guessing -- curiosity ('want the "
            "details?'), effort externalization ('I've drafted X, just say go'), or a direct, specific "
            "question to the merchant about their own business (not a yes/no qualifier). If category "
            "peer_stats are present in the context below, a real peer-benchmark comparison (their number "
            "vs the actual peer_stats figure) is a strong, safe social-proof lever -- but never invent a "
            "competitor name, a locality claim, or a headcount ('3 dentists nearby...') that isn't "
            "literally present in the context."
            if send_as == "vera" else
            "10. Engagement: this message goes to the CUSTOMER, not the merchant -- ground it in the "
            "customer's own relationship with this business (their name, visit history, preferred "
            "slots/service from CUSTOMER CONTEXT below), not the merchant's performance numbers. Prefer a "
            "concrete, low-effort next step (a specific slot, a specific price) over a generic check-in. "
            "Never invent a competitor name or a claim not literally present in the context."
        ),
        "",
        "=== CATEGORY CONTEXT ===", _compact_json(category),
        "", "=== MERCHANT CONTEXT ===", _compact_json(merchant),
        "", "=== TRIGGER CONTEXT ===", _compact_json(trigger, max_chars=1200),
    ]
    if customer:
        parts += ["", "=== CUSTOMER CONTEXT ===", _compact_json(customer, max_chars=1500)]
    parts += [
        "",
        "Respond with ONLY a JSON object (no markdown fences, no commentary), exactly these keys:",
        '{"body": "...", "cta": "binary" | "open_ended" | "none", "rationale": "one sentence on why this message / what it should achieve"}',
    ]
    return "\n".join(parts)


def compose_proactive(category: dict, merchant: dict, trigger: dict,
                       customer: Optional[dict] = None) -> dict:
    """Used by /v1/tick. Deterministic (temperature=0)."""
    llm = get_llm()
    system_prompt = _build_proactive_system_prompt(category, merchant, trigger, customer)
    response = _invoke_with_retry(llm, [SystemMessage(content=system_prompt), HumanMessage(content="Compose the message now.")])
    parsed = _extract_json(response.content)

    body = parsed.get("body", "").strip()
    cta = parsed.get("cta", "none")
    rationale = parsed.get("rationale", "").strip()

    if _count_ctas(body) > 1:
        fix_prompt = (
            "This WhatsApp message has more than one call-to-action. Rewrite it with exactly ONE "
            "call-to-action landing in the last sentence, keeping everything else the same. "
            "Return ONLY the rewritten message body, no quotes, no commentary:\n\n" + body
        )
        body = _clean_llm_text(_invoke_with_retry(llm, [HumanMessage(content=fix_prompt)]).content)

    body = _romanize_if_needed(llm, body)

    send_as = "merchant_on_behalf" if customer else "vera"
    suppression_key = trigger.get("suppression_key") or _default_suppression_key(trigger, merchant)

    return {
        "body": body, "cta": cta, "send_as": send_as,
        "suppression_key": suppression_key, "rationale": rationale,
    }


def _reply_prompt(mode: str, merchant: Optional[dict], category: Optional[dict],
                   customer: Optional[dict], send_as: str, bot_messages: list,
                   incoming_message: str) -> str:
    """Shared prompt builder for both continuing a pitch and handing off to
    action once the merchant has agreed. `mode` is 'handoff' or 'continue'."""
    if send_as == "merchant_on_behalf":
        voice_rule = ("This conversation is FROM the merchant TO their own customer -- do not "
                      "mention Vera or magicpin by name in the body.")
        who = "the customer"
    else:
        voice_rule = "This conversation is FROM Vera TO the merchant."
        who = "the merchant"

    lang_hint = _language_hint(merchant, customer if send_as == "merchant_on_behalf" else None)

    if mode == "handoff":
        instruction = (
            f"{who.capitalize()} just said yes / expressed clear intent to proceed. Do NOT ask "
            "another qualifying question ('would you like...', 'do you want...'). Take the "
            "concrete next action step immediately and say, in a decisive tone ('done', "
            "'sending', 'started', 'confirmed'), what you just did or what you need from them next."
        )
    else:
        instruction = (
            "Continue this WhatsApp conversation naturally. Keep the same voice/language as prior "
            "messages, exactly one clear next step, no re-introduction, and don't repeat a prior "
            "message verbatim."
        )

    parts = [
        f"You are composing Vera's next WhatsApp message, addressed to {who}.",
        voice_rule,
        f"Language: write in {lang_hint}.",
        instruction,
        "Never invent data not present in the context below.",
        "",
    ]
    if merchant:
        parts += ["=== MERCHANT CONTEXT ===", json.dumps(merchant, ensure_ascii=False, indent=2)[:4000], ""]
    if category:
        parts += ["=== CATEGORY VOICE ===", json.dumps(category.get("voice", {}), ensure_ascii=False), ""]
    if customer and send_as == "merchant_on_behalf":
        parts += ["=== CUSTOMER CONTEXT ===", json.dumps(customer, ensure_ascii=False, indent=2)[:2000], ""]

    parts += [
        "=== PRIOR MESSAGES IN THIS CONVERSATION (most recent last) ===",
        "\n".join(bot_messages[-3:]) if bot_messages else "(none yet)",
        "",
        f"{who.capitalize()} just replied: {incoming_message}",
        "",
        "Respond with ONLY a JSON object (no markdown fences, no commentary), exactly these keys:",
        '{"body": "...", "rationale": "one sentence on why this message / what it should achieve"}',
    ]
    return "\n".join(parts)


def _compose_reply(mode: str, merchant: Optional[dict], category: Optional[dict],
                    customer: Optional[dict], send_as: str, bot_messages: list,
                    incoming_message: str) -> dict:
    llm = get_llm()
    prompt = _reply_prompt(mode, merchant, category, customer, send_as, bot_messages, incoming_message)
    response = _invoke_with_retry(llm, [HumanMessage(content=prompt)])
    try:
        parsed = _extract_json(response.content)
        body = parsed.get("body", "").strip()
        rationale = parsed.get("rationale", "").strip()
        if not body:
            raise ValueError("model returned an empty body")
        body = _romanize_if_needed(llm, body)
        return {"body": body, "rationale": rationale}
    except Exception:
        # Model didn't return clean JSON -- degrade to using the raw text
        # rather than failing the whole turn over a formatting slip.
        return {"body": _clean_llm_text(response.content), "rationale": ""}


def compose_action_handoff(merchant: Optional[dict], category: Optional[dict],
                            customer: Optional[dict], send_as: str,
                            bot_messages: list, incoming_message: str) -> dict:
    """Merchant/customer just said yes -- jump straight to a concrete next
    action, no more qualifying. Returns {"body": ..., "rationale": ...}."""
    return _compose_reply("handoff", merchant, category, customer, send_as, bot_messages, incoming_message)


def compose_continue(merchant: Optional[dict], category: Optional[dict],
                      customer: Optional[dict], send_as: str,
                      bot_messages: list, incoming_message: str) -> dict:
    """Normal engaged reply -- continue the conversation naturally.
    Returns {"body": ..., "rationale": ...}."""
    return _compose_reply("continue", merchant, category, customer, send_as, bot_messages, incoming_message)