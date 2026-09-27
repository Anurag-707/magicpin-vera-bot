"""
Fast, deterministic pattern detectors used ahead of any LLM call in
/v1/reply. Keeping these regex-based (not LLM calls) keeps replies
comfortably inside the judge's 30s timeout and makes the three replay-test
behaviors (auto-reply, intent transition, hostile) reliable and cheap.
"""

import re

AUTO_REPLY_PATTERNS = [
    r"thank you for contacting",
    r"will respond shortly",
    r"we have received your message",
    r"automated (response|reply|message)",
    r"team will (get back|respond|reach out)",
    r"currently (unavailable|away)",
    r"aap(k)?i jaankari ke liye.*shukriya",
    r"main.*automated assistant",
    r"business hours",
]

HOSTILE_PATTERNS = [
    r"\bstop messaging\b", r"\bspam\b", r"\buseless\b", r"\bshut up\b",
    r"\bidiot\b", r"\bstupid\b", r"\bharass", r"\bfuck", r"\bbullshit\b",
    r"\bwaste of time\b", r"\bannoying\b", r"\bpathetic\b",
]

# Patterns unambiguous enough to count as YES on their own -- they only
# make sense as agreement, so no negation check is needed.
STRONG_YES_PATTERNS = [
    r"\byes\b", r"ok(ay)?.*(let'?s|lets)", r"let'?s do it", r"\bchalo\b",
    r"\bjudrna hai\b", r"jo?ir?n[ae]? hai\b", r"join.*(hai|karna)",
    r"go ahead", r"\bconfirmed?\b", r"\bproceed\b",
]

# Patterns that are only weak evidence of YES on their own -- "sure",
# "start", "done" etc. are just as at home in a negative sentence ("not
# sure", "please don't start", "not done yet"), so these only count if
# there's no negation word anywhere in the same message.
WEAK_YES_PATTERNS = [
    r"\bha+n?\b", r"\bdone\b", r"\bstart\b", r"\bsure\b", r"what'?s next",
]

NEGATION_MARKERS = [
    r"\bnot\b", r"\bnah+i\b", r"\bmat\b", r"don'?t\b", r"do not\b", r"\bno\b",
    r"\bnever ?mind\b", r"\bkabhi nahi\b",
]

INTENT_NO_PATTERNS = [
    r"\bnahi\b", r"\bno\b", r"not interested", r"\bstop\b(?! messaging)",
    r"band karo", r"don'?t.*message", r"\bunsubscribe\b",
]


def _matches_any(patterns, text: str) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in patterns)


def is_auto_reply_pattern(text: str) -> bool:
    """Text looks like a known canned/auto-reply phrasing (fast path)."""
    return _matches_any(AUTO_REPLY_PATTERNS, text)


def is_pure_repeat_auto_reply(prior_occurrences: int, threshold: int = 2) -> bool:
    """
    Content-agnostic fallback for the brief's literal hint: "same message
    verbatim 3+ times = auto-reply". `prior_occurrences` is how many times
    this exact (normalized) text was already seen from this sender BEFORE
    the current message, so threshold=2 means the current message is the
    3rd verbatim occurrence. This fires even when the text doesn't match
    any known canned phrase in AUTO_REPLY_PATTERNS, which is the whole
    point -- a real judge's canned text won't always match our keyword list.
    """
    return prior_occurrences >= threshold


def is_hostile(text: str) -> bool:
    return _matches_any(HOSTILE_PATTERNS, text)


def is_explicit_yes(text: str) -> bool:
    t = text.lower()
    if any(re.search(p, t) for p in STRONG_YES_PATTERNS):
        return True
    if any(re.search(p, t) for p in WEAK_YES_PATTERNS):
        # A weak cue only counts as YES if nothing in the message reads as
        # a negation -- guards against "not sure", "don't start", etc.
        if not any(re.search(n, t) for n in NEGATION_MARKERS):
            return True
    return False


def is_explicit_no(text: str) -> bool:
    return _matches_any(INTENT_NO_PATTERNS, text)


def count_ctas(body: str) -> int:
    return len(re.findall(r"\breply\b", body, re.IGNORECASE))
