"""Task 7 - the CrewAI support crew: Retrieval Agent -> Lookup Agent -> Response Composer.

Each agent gets exactly the tools it needs (least autonomy, see Task 15):
  Retrieval Agent   : rag_lookup
  Lookup Agent      : check_order_status
  Response Composer : no tools; turns both outputs into a SupportResponse (Task 9)

Under MOCK_LLM every agent runs on its own MockLLM; the per-agent `responder`
functions below are the deterministic stand-ins for what each agent's LLM writes.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import config  # telemetry off before crewai import
from crewai import Agent, Crew, Process, Task

from crew.schemas import OrderInfo, SupportResponse
from governance import enforce_least_autonomy
from llm.mock_llm import MockLLM, MockTurn
from rag.generate import IDK_ANSWER
from rag.kb import split_sentences
from tools.crew_tools import OrderStatusTool, RagLookupTool
from tools.order_status import ESCALATION_THRESHOLD

NO_ORDER_ID = {"tool": None, "found": False, "record_id": None,
               "error": "No order ID (format NYK-####) in the customer query."}


# --- Mock responders (what each agent's "LLM" writes as its Final Answer) -----
def retrieval_responder(turn: MockTurn) -> str:
    return turn.observation or json.dumps({"grounded": False, "answer": IDK_ANSWER, "sources": [],
                                           "top_similarity": 0.0, "context": []})


def lookup_responder(turn: MockTurn) -> str:
    return turn.observation or json.dumps(NO_ORDER_ID)


def json_objects(text: str) -> list[dict]:
    """Every top-level JSON object embedded in `text`."""
    decoder, found, i = json.JSONDecoder(), [], 0
    while (i := text.find("{", i)) != -1:
        try:
            obj, end = decoder.raw_decode(text, i)
        except json.JSONDecodeError:
            i += 1
            continue
        if isinstance(obj, dict):
            found.append(obj)
        i = end
    return found


ASK_FOR_ORDER_ID = ("If your question is about a specific order, please share its order ID "
                    "(for example NYK-1234) so I can check it.")
# Only questions about the state of ONE specific purchase ("return it", "where is
# my order", "cancel this order") - not generic mentions such as "deliver my
# order to London" or "sent the item back", which are policy questions.
_REFERS_TO_AN_ORDER = re.compile(
    r"\b(return|cancel|track|exchange|refund) it\b"
    r"|\bwhere is (my|the|this) (order|package|parcel)\b"
    r"|\b(status of|track|cancel|return|exchange) (my|this|that) (order|package|parcel|purchase)\b",
    re.IGNORECASE,
)


def order_sentences(order: OrderInfo) -> list[str]:
    """Customer-facing facts about one order, rendered only from the lookup record."""
    days = order.days_since_created
    when = "today" if days == 0 else "1 day ago" if days == 1 else f"{days} days ago"
    out = [f"Order {order.record_id} ({order.category}, Rs {order.order_value_inr:,}) "
           f"is currently {order.status}, placed {when}."]
    if order.delayed_shipment:
        out.append("It is flagged as a delayed shipment.")
    if order.escalate:
        out.append(f"Its escalation score is {order.escalation_score:.2f} (threshold "
                   f"{ESCALATION_THRESHOLD:.2f}), so it has been escalated to Level 2 support.")
    else:
        out.append(f"Its escalation score is {order.escalation_score:.2f}, below the "
                   f"{ESCALATION_THRESHOLD:.2f} threshold, so no escalation is needed.")
    return out


def order_not_found_sentences(record_id: str) -> list[str]:
    return [f"I couldn't find an order with ID {record_id}.",
            "Please check the order ID on your confirmation email."]


def template_sentences(order: OrderInfo | None, missing_id: str | None) -> list[str]:
    """Every sentence the Composer may write that does not come from the KB.
    Used by the output groundedness check (guardrails.check_groundedness)."""
    allowed = split_sentences(IDK_ANSWER) + split_sentences(ASK_FOR_ORDER_ID)
    if order is not None:
        allowed += order_sentences(order)
    if missing_id:
        allowed += order_not_found_sentences(missing_id)
    return allowed


def compose_response(retrieval: dict, lookup: dict, query: str = "") -> SupportResponse:
    """Deterministic composition of the two upstream outputs into one draft."""
    parts, sources, order = [], [], None
    policy_ok = bool(retrieval.get("grounded"))

    if lookup.get("record_id"):  # the customer asked about a specific order
        intent = "order_status"
        if lookup.get("found"):
            order = OrderInfo(**{k: lookup[k] for k in OrderInfo.model_fields})
            parts += order_sentences(order)
            sources.append("orders_dataset")
        else:
            parts += order_not_found_sentences(lookup["record_id"])
        # Pure status questions match the KB only weakly; attach policy only for a
        # real order when the customer clearly asked a policy question too.
        strong = retrieval.get("top_similarity", 0.0) >= config.ORDER_POLICY_MIN_SIM
        if order is not None and policy_ok and strong:
            parts.append(f"Relevant policy: {retrieval['answer']}")
            sources += retrieval.get("sources", [])
    elif policy_ok:
        intent = "policy"
        parts.append(retrieval["answer"])
        sources += retrieval.get("sources", [])
        if _REFERS_TO_AN_ORDER.search(query):  # "Can I return it?" with no order on record
            parts.append(ASK_FOR_ORDER_ID)
    else:
        intent = "out_of_scope"
        parts.append(IDK_ANSWER)

    return SupportResponse(
        answer=" ".join(parts),
        intent=intent,
        grounded=bool(sources),
        sources=list(dict.fromkeys(sources)),
        order=order,
        escalate=bool(order and order.escalate),
        retrieval_top_similarity=float(retrieval.get("top_similarity", 0.0)),
    )


def composer_responder(turn: MockTurn) -> str:
    blobs = json_objects(turn.context or "")
    retrieval = next((b for b in blobs if "grounded" in b), {"grounded": False, "top_similarity": 0.0})
    lookup = next((b for b in blobs if "found" in b), NO_ORDER_ID)
    return compose_response(retrieval, lookup, turn.query).model_dump_json()


# --- Crew --------------------------------------------------------------------
@dataclass
class CrewRun:
    query: str
    response: SupportResponse
    retrieval: dict
    lookup: dict
    tool_calls: dict[str, list[str]]
    decisions: list[str] = field(default_factory=list)
    llm_calls: int = 0
    tokens_used: int = 0


RETRIEVAL, LOOKUP, COMPOSER = "Retrieval Agent", "Lookup Agent", "Response Composer"


def build_agents(extra_tools: dict[str, list] | None = None) -> dict[str, Agent]:
    """Each tool is issued to (bound to) exactly one role; governance enforces the
    allow-list before the crew may run. `extra_tools` exists only so the Task 15
    demo can attempt a forbidden wiring and show it being refused."""
    common = dict(max_iter=3, allow_delegation=False, verbose=False)
    extra = extra_tools or {}
    agents = {
        "retrieval": Agent(
            role=RETRIEVAL,
            goal="Find the Nykaa policy text that answers the customer's question.",
            backstory="Knows the Nykaa support knowledge base and only reports what it retrieves.",
            llm=MockLLM(model="mock-retrieval", responder=retrieval_responder),
            tools=[RagLookupTool(bound_role=RETRIEVAL), *extra.get("retrieval", [])], **common,
        ),
        "lookup": Agent(
            role=LOOKUP,
            goal="Fetch the status of the customer's order when they give an order ID.",
            backstory="The only agent allowed to query the orders system.",
            llm=MockLLM(model="mock-lookup", responder=lookup_responder),
            tools=[OrderStatusTool(bound_role=LOOKUP), *extra.get("lookup", [])], **common,
        ),
        "composer": Agent(
            role=COMPOSER,
            goal="Combine retrieved policy and order data into one accurate, grounded reply.",
            backstory="Writes concise customer replies and never adds facts that were not provided.",
            llm=MockLLM(model="mock-composer", responder=composer_responder),
            tools=[*extra.get("composer", [])], **common,
        ),
    }
    enforce_least_autonomy(agents.values())
    return agents


def run_crew(query: str) -> CrewRun:
    query = " ".join(query.split())  # one line: the mock reads the `Customer query:` line
    agents = build_agents()
    t_retrieve = Task(
        description=f"Retrieve the Nykaa policy relevant to the customer's question.\nCustomer query: {query}",
        expected_output="JSON with the grounded answer, sources and retrieved context.",
        agent=agents["retrieval"], context=[],
    )
    t_lookup = Task(
        description=f"If the customer mentions an order ID, look up that order.\nCustomer query: {query}",
        expected_output="JSON with the order record and escalation score, or found=false.",
        agent=agents["lookup"], context=[],
    )
    t_compose = Task(
        description=(f"Write the final reply using ONLY the retrieval and lookup results.\n"
                     f"Customer query: {query}"),
        expected_output="A SupportResponse JSON object.",
        agent=agents["composer"], context=[t_retrieve, t_lookup],
        output_pydantic=SupportResponse,
    )
    crew = Crew(
        agents=list(agents.values()), tasks=[t_retrieve, t_lookup, t_compose],
        process=Process.sequential, tracing=False, verbose=False,
    )
    result = crew.kickoff()

    # Task 9: validate the final output against the schema in code.
    response = SupportResponse.model_validate(result.pydantic.model_dump() if result.pydantic
                                              else json.loads(result.raw))
    outputs = [t.raw for t in result.tasks_output]
    llms = [a.llm for a in agents.values()]
    return CrewRun(
        query=query,
        response=response,
        retrieval=(json_objects(outputs[0]) or [{}])[0],
        lookup=(json_objects(outputs[1]) or [{}])[0],
        tool_calls={tool.name: list(tool.calls) for a in agents.values() for tool in a.tools},
        decisions=[d for llm in llms for d in llm.decisions],
        llm_calls=sum(llm.call_count for llm in llms),
        tokens_used=sum(llm.tokens_used for llm in llms),
    )
