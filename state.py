"""
In-memory state stores for the Vera bot HTTP server.

Everything here resets when the process restarts -- fine per the testing
brief ("storing in memory is fine; just don't restart between calls").
An in-memory dict is also enough to satisfy section 11's privacy rule
(wipe on teardown, nothing persisted to disk).
"""

import threading
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ContextEntry:
    version: int
    payload: dict


@dataclass
class Conversation:
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    mode: str = "pitch"  # "pitch" | "action" | "exited"
    bot_messages: list = field(default_factory=list)
    merchant_messages: list = field(default_factory=list)


class Store:
    def __init__(self):
        self._lock = threading.Lock()
        self.start_time = time.time()
        # (scope, context_id) -> ContextEntry
        self.contexts: dict[tuple[str, str], ContextEntry] = {}
        # conversation_id -> Conversation
        self.conversations: dict[str, Conversation] = {}
        # merchant_id -> list of raw inbound message texts, across ALL
        # conversations for that merchant. The judge's own auto-reply test
        # starts a brand-new conversation_id on every turn, so detection
        # has to key off merchant_id, not conversation_id.
        self.merchant_inbound: dict[str, list] = {}
        # suppression_keys already sent this run, so the same trigger
        # doesn't get composed and sent twice.
        self.sent_suppression_keys: set[str] = set()

    # ---- contexts ----
    def get_context(self, scope: str, context_id: Optional[str]) -> Optional[dict]:
        if not context_id:
            return None
        entry = self.contexts.get((scope, context_id))
        return entry.payload if entry else None

    def push_context(self, scope: str, context_id: str, version: int, payload: dict):
        """Returns ('accepted' | 'stale', version_to_report)."""
        with self._lock:
            key = (scope, context_id)
            cur = self.contexts.get(key)
            if cur is None or version > cur.version:
                self.contexts[key] = ContextEntry(version=version, payload=payload)
                return "accepted", version
            if version == cur.version:
                return "accepted", version  # idempotent no-op
            return "stale", cur.version

    def contexts_loaded_counts(self) -> dict:
        counts = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
        for scope, _cid in self.contexts:
            counts[scope] = counts.get(scope, 0) + 1
        return counts

    # ---- conversations ----
    def get_or_create_conversation(self, conversation_id: str, merchant_id: Optional[str],
                                    customer_id: Optional[str]) -> Conversation:
        with self._lock:
            conv = self.conversations.get(conversation_id)
            if conv is None:
                conv = Conversation(conversation_id=conversation_id, merchant_id=merchant_id,
                                     customer_id=customer_id)
                self.conversations[conversation_id] = conv
            else:
                # keep merchant/customer id fresh if the caller supplied one
                conv.merchant_id = conv.merchant_id or merchant_id
                conv.customer_id = conv.customer_id or customer_id
            return conv

    def record_inbound(self, merchant_id: Optional[str], message: str):
        if not merchant_id:
            return
        with self._lock:
            self.merchant_inbound.setdefault(merchant_id, []).append(message)

    def inbound_occurrences(self, merchant_id: Optional[str], message: str) -> int:
        if not merchant_id:
            return 0
        normalized = message.strip().lower()
        history = self.merchant_inbound.get(merchant_id, [])
        return sum(1 for m in history if m.strip().lower() == normalized)

    def mark_sent(self, suppression_key: Optional[str]):
        if suppression_key:
            with self._lock:
                self.sent_suppression_keys.add(suppression_key)

    def already_sent(self, suppression_key: Optional[str]) -> bool:
        if not suppression_key:
            return False
        return suppression_key in self.sent_suppression_keys

    def wipe(self):
        with self._lock:
            self.contexts.clear()
            self.conversations.clear()
            self.merchant_inbound.clear()
            self.sent_suppression_keys.clear()


store = Store()
