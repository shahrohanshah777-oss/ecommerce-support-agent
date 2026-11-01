"""The CrewAI support crew: Retrieval Agent -> Lookup Agent -> Response Composer.

Each agent gets exactly the tools it needs (least autonomy, see governance.py):
  Retrieval Agent   : rag_lookup
  Lookup Agent      : check_order_status
  Response Composer : no tools; turns both outputs into a SupportResponse

Under MOCK_LLM every agent runs on its own MockLLM; the per-agent `responder`
functions below are the deterministic stand-ins for what each agent's LLM writes.
With MOCK_LLM=false each agent runs on the configured real LLM (llm/providers.py):
the agents choose their tool calls themselves and the Composer writes the
SupportResponse, following REAL_MODE_INSTRUCTIONS.
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
from rag.generate import IDK_ANSWER, llm_generator
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

# Real-LLM mode only: appended to each task, after the `Customer query:` line the
# mock reads, so MOCK_LLM prompts (and their metered token counts) are unchanged.
REAL_MODE_INSTRUCTIONS = {
    "retrieval": "Call the rag_lookup tool exactly once, passing the customer's question as the query.",
    "lookup": ("If the customer query contains an order ID (format NYK-####), call check_order_status "
               "exactly once with that ID. If it contains no order ID, call no tool and reply with exactly "
               f"this JSON: {json.dumps(NO_ORDER_ID)}"),
    "composer": f"""Rules for the SupportResponse:
- Use ONLY facts stated in the rag_lookup JSON (its "answer" and "context") and the check_order_status JSON. Never add policies, numbers, coupons, discounts or promises that are not stated there.
- intent: "order_status" if the lookup JSON has a record_id; otherwise "policy" if the rag_lookup JSON has "grounded": true; otherwise "out_of_scope".
- out_of_scope: answer exactly: {IDK_ANSWER}
- Order found: give its status, category, value in Rs and how many days ago it was placed; say if it is flagged as a delayed shipment; give its escalation score against the {ESCALATION_THRESHOLD:.2f} threshold and, if escalate is true, say it has been escalated to Level 2 support. Copy the `order` object field-for-field from the lookup JSON. Add the policy only if the customer also asked a policy question about the order.
- Order not found: say no order exists with that ID and ask the customer to check the order ID on their confirmation email; order must be null.
- No order ID, but the customer asks about one specific order ("return it", "where is my order"): end with exactly: {ASK_FOR_ORDER_ID}
- sources: the rag_lookup "sources" you used, plus "orders_dataset" if an order was found. grounded: true if sources is non-empty. escalate: the order's escalate value, else false. retrieval_top_similarity: copy "top_similarity" from the rag_lookup JSON.
- Keep the answer under 80 words.""",
}


def _llm(mock_name: str, responder):
    if config.MOCK_LLM:
        return MockLLM(model=mock_name, responder=responder)
    from llm.providers import crew_llm

    return crew_llm()


def build_agents(extra_tools: dict[str, list] | None = None) -> dict[str, Agent]:
    """Each tool is issued to (bound to) exactly one role; governance enforces the
    allow-list before the crew may run. `extra_tools` exists only so the governance
    demo can attempt a forbidden wiring and show it being refused."""
    common = dict(max_iter=3, allow_delegation=False, verbose=False)
    extra = extra_tools or {}
    # Real mode: a tool's JSON becomes its task's output verbatim, so the Composer
    # sees exactly what the tools returned (and no extra LLM call restates it).
    direct = dict(result_as_answer=not config.MOCK_LLM)
    agents = {
        "retrieval": Agent(
            role=RETRIEVAL,
            goal="Find the Nykaa policy text that answers the customer's question.",
            backstory="Knows the Nykaa support knowledge base and only reports what it retrieves.",
            llm=_llm("mock-retrieval", retrieval_responder),
            tools=[RagLookupTool(bound_role=RETRIEVAL, **direct), *extra.get("retrieval", [])], **common,
        ),
        "lookup": Agent(
            role=LOOKUP,
            goal="Fetch the status of the customer's order when they give an order ID.",
            backstory="The only agent allowed to query the orders system.",
            llm=_llm("mock-lookup", lookup_responder),
            tools=[OrderStatusTool(bound_role=LOOKUP, **direct), *extra.get("lookup", [])], **common,
        ),
        "composer": Agent(
            role=COMPOSER,
            goal="Combine retrieved policy and order data into one accurate, grounded reply.",
            backstory="Writes concise customer replies and never adds facts that were not provided.",
            llm=_llm("mock-composer", composer_responder),
            tools=[*extra.get("composer", [])], **common,
        ),
    }
    enforce_least_autonomy(agents.values())
    return agents


def _describe(text: str, key: str) -> str:
    return text if config.MOCK_LLM else f"{text}\n\n{REAL_MODE_INSTRUCTIONS[key]}"


def _tool_result(agent: Agent, task_output: str, default: dict) -> dict:
    """What the agent's tool actually returned (last call); else whatever JSON the agent wrote."""
    tool = next((t for t in agent.tools if getattr(t, "outputs", None)), None)
    if tool is not None:
        return json.loads(tool.outputs[-1])
    return (json_objects(task_output) or [default])[0]


def run_crew(query: str) -> CrewRun:
    query = " ".join(query.split())  # one line: the mock reads the `Customer query:` line
    agents = build_agents()
    generated_before = llm_generator.tokens_used
    t_retrieve = Task(
        description=_describe(f"Retrieve the Nykaa policy relevant to the customer's question.\n"
                              f"Customer query: {query}", "retrieval"),
        expected_output="JSON with the grounded answer, sources and retrieved context.",
        agent=agents["retrieval"], context=[],
    )
    t_lookup = Task(
        description=_describe(f"If the customer mentions an order ID, look up that order.\n"
                              f"Customer query: {query}", "lookup"),
        expected_output="JSON with the order record and escalation score, or found=false.",
        agent=agents["lookup"], context=[],
    )
    t_compose = Task(
        description=_describe(f"Write the final reply using ONLY the retrieval and lookup results.\n"
                              f"Customer query: {query}", "composer"),
        expected_output="A SupportResponse JSON object.",
        agent=agents["composer"], context=[t_retrieve, t_lookup],
        output_pydantic=SupportResponse,
    )
    crew = Crew(
        agents=list(agents.values()), tasks=[t_retrieve, t_lookup, t_compose],
        process=Process.sequential, tracing=False, verbose=False,
    )
    result = crew.kickoff()

    # Validate the final output against the schema in code.
    response = SupportResponse.model_validate(result.pydantic.model_dump() if result.pydantic
                                              else json.loads(result.raw))
    outputs = [t.raw for t in result.tasks_output]
    llms = [a.llm for a in agents.values()]
    if config.MOCK_LLM:
        decisions = [d for llm in llms for d in llm.decisions]
        llm_calls = sum(llm.call_count for llm in llms)
        tokens = sum(llm.tokens_used for llm in llms)
    else:  # provider-reported usage; RAG generation runs inside the rag_lookup tool
        usage = [llm.get_token_usage_summary() for llm in llms]
        decisions = [f"[{a.role}] Action {t.name} {{'arg': {c!r}}}"
                     for a in agents.values() for t in a.tools for c in t.calls]
        llm_calls = sum(u.successful_requests for u in usage)
        tokens = sum(u.total_tokens for u in usage) + llm_generator.tokens_used - generated_before
    return CrewRun(
        query=query,
        response=response,
        retrieval=_tool_result(agents["retrieval"], outputs[0], {}),
        lookup=_tool_result(agents["lookup"], outputs[1], NO_ORDER_ID),
        tool_calls={tool.name: list(tool.calls) for a in agents.values() for tool in a.tools},
        decisions=decisions,
        llm_calls=llm_calls,
        tokens_used=tokens,
    )
