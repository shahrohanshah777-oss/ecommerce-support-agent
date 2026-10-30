"""Task 8 - session memory across turns, plus a separate fresh-session transcript.

Writes two transcripts:
  transcripts/task08_memory_multiturn.txt  state carried across 3 turns of one session
  transcripts/task08_memory_fresh.txt      a brand-new session: state correctly absent

Run:  python -m demos.task08_memory
"""
import config  # noqa: F401

from demos._transcript import transcript
from memory import get_session_history
from pipeline import handle

FOLLOW_UP = "How many days do I have to return it?"
MULTI_TURN = [
    "Hi, where is my order NYK-1009?",
    FOLLOW_UP,
    "Is it going to be escalated?",
]


def show_turn(i: int, query: str, session_id: str) -> None:
    before = len(get_session_history(session_id).messages)
    r = handle(query, session_id)
    print(f"--- Turn {i} | history before: {before} messages")
    print(f"User           : {query}")
    print(f"Resolved query : {r.resolved_query}" + ("   <- order ID filled in from memory"
                                                    if r.resolved_query != query else "   (unchanged)"))
    print(f"Tools invoked  : {r.tool_calls}")
    print(f"Order in reply : {r.response.order.record_id if r.response and r.response.order else None}")
    print(f"Assistant      : {r.answer}\n")
    return r


def dump_history(session_id: str) -> None:
    print(f"Stored history for session {session_id} (InMemoryChatMessageHistory):")
    for m in get_session_history(session_id).messages:
        print(f"  {m.type:5s}: {m.content[:110]}")


def multi_turn() -> None:
    sid = "session-multiturn"
    print(f"Session {sid} - one conversation, three turns\n")
    results = [show_turn(i, q, sid) for i, q in enumerate(MULTI_TURN, 1)]
    dump_history(sid)
    assert results[1].resolved_query == f"{FOLLOW_UP} NYK-1009"
    assert results[1].response.order.record_id == "NYK-1009"
    assert "returned within 15 days" in results[1].answer
    assert results[2].response.order.escalate is True
    print("\nPASS: turns 2 and 3 never mention the order ID, yet both answer about NYK-1009 from memory.")


def fresh_session() -> None:
    sid = "session-fresh"
    print(f"Session {sid} - a NEW conversation (no prior turns)\n")
    r = show_turn(1, FOLLOW_UP, sid)
    dump_history(sid)
    assert r.resolved_query == FOLLOW_UP
    assert r.tool_calls["check_order_status"] == [] and r.response.order is None
    assert "share its order ID" in r.answer
    print("\nPASS: the same follow-up in a fresh session has no order to refer to - nothing leaked from")
    print("the other session; the agent answers the general policy and asks for the order ID.")


if __name__ == "__main__":
    with transcript("task08_memory_multiturn"):
        multi_turn()
    with transcript("task08_memory_fresh"):
        fresh_session()
