"""Task 8 - session memory with LangChain's InMemoryChatMessageHistory +
RunnableWithMessageHistory.

History lives in-process only (one InMemoryChatMessageHistory per session_id); it
does not survive a restart, which the brief allows. Under MOCK_LLM the
"history-aware query rewriting" step is deterministic: a follow-up that refers
to an order without naming it ("Can I still return it?") inherits the most
recent order ID mentioned in the session.

Note: RunnableWithMessageHistory emits a LangChainDeprecationWarning that points
to LangGraph persistence. As the brief says, that is expected; the class works.
"""
from __future__ import annotations

import re
from typing import Any, Callable

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory

import config

_SESSIONS: dict[str, InMemoryChatMessageHistory] = {}
_REFERENCE = re.compile(r"\b(it|its|this order|that order|my order|the order|this one|that one)\b", re.IGNORECASE)


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    return _SESSIONS.setdefault(session_id, InMemoryChatMessageHistory())


def clear_session(session_id: str) -> None:
    _SESSIONS.pop(session_id, None)


def last_order_id(history: list[BaseMessage]) -> str | None:
    for msg in reversed(history):
        ids = re.findall(config.ORDER_ID_PATTERN, str(msg.content), flags=re.IGNORECASE)
        if ids:
            return ids[-1].upper()
    return None


def resolve_query(query: str, history: list[BaseMessage]) -> str:
    """Make a follow-up self-contained using the session history."""
    if re.search(config.ORDER_ID_PATTERN, query, flags=re.IGNORECASE) or not _REFERENCE.search(query):
        return query
    order_id = last_order_id(history)
    return f"{query} {order_id}" if order_id else query


def build_memory_chain(respond: Callable[[str], dict[str, Any]]) -> RunnableWithMessageHistory:
    """Wrap `respond(resolved_query) -> {"answer": str, ...}` with per-session history.

    The human message stored is the (already PII-masked) input; the AI message is
    the final answer after all guardrails.
    """

    def turn(inputs: dict) -> dict:
        resolved = resolve_query(inputs["input"], inputs["history"])
        out = respond(resolved)
        return {**out, "resolved_query": resolved, "history_before": len(inputs["history"])}

    return RunnableWithMessageHistory(
        RunnableLambda(turn),
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
        output_messages_key="answer",
    )
