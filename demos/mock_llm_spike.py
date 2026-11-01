"""Spike: prove the MockLLM drives real CrewAI `.kickoff()` runs offline.

Uses DUMMY tools (the real RAG and order tools came later). Checks:
  * tools genuinely execute (call counters) and their output reaches the answer
  * pitfall 1: the system prompt's "Observation: the result of the action"
    text never leaks into an answer
  * pitfall 2: args come from each tool's declared schema - `rag_lookup` gets
    `query`, `check_order_status` gets `record_id`
  * multi-agent context passing (Retrieval -> Composer)

Run:  python -m demos.mock_llm_spike
"""
import config  # noqa: F401  (must be first: disables telemetry/tracing)

from crewai import Agent, Crew, Process, Task
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from demos._transcript import transcript
from llm.mock_llm import MockLLM, MockTurn

TEMPLATE_PLACEHOLDER = "the result of the action"
RUNS = {"rag_lookup": [], "check_order_status": []}


class QueryArgs(BaseModel):
    query: str = Field(..., description="The customer's policy question.")


class RecordArgs(BaseModel):
    record_id: str = Field(..., description="Order ID, e.g. NYK-1042.")


class DummyRagLookup(BaseTool):
    name: str = "rag_lookup"  # contains "lookup" on purpose (pitfall 2)
    description: str = "Look up Nykaa policy text for a question."
    args_schema: type[BaseModel] = QueryArgs

    def _run(self, query: str) -> str:
        RUNS["rag_lookup"].append(query)
        return "[DUMMY KB] Footwear can be returned within 15 days of delivery if unworn."


class DummyOrderStatus(BaseTool):
    name: str = "check_order_status"
    description: str = "Get the status of an order by its ID."
    args_schema: type[BaseModel] = RecordArgs

    def _run(self, record_id: str) -> str:
        RUNS["check_order_status"].append(record_id)
        return f'{{"record_id": "{record_id}", "status": "Shipped", "order_value_inr": 1499}}'


def compose(turn: MockTurn) -> str:
    return f"Here is what I found for your question: {turn.context}"


def run_single(query: str, llm: MockLLM) -> str:
    agent = Agent(
        role="Support Agent",
        goal="Answer Nykaa customer questions using tools.",
        backstory="Uses the right tool for each question.",
        llm=llm,
        tools=[DummyRagLookup(), DummyOrderStatus()],
        max_iter=3,
    )
    task = Task(
        description=f"Answer the customer.\nCustomer query: {query}",
        expected_output="A short answer.",
        agent=agent,
    )
    return Crew(agents=[agent], tasks=[task], tracing=False).kickoff().raw


def show(llm: MockLLM, answer: str) -> None:
    for d in llm.decisions:
        print(f"    llm decision : {d}")
    print(f"    final answer : {answer}")
    print(f"    llm calls    : {llm.call_count}")


def main() -> None:
    # A: policy question. The agent has both tools; only rag_lookup's schema can be filled.
    print("CASE A - policy question -> rag_lookup(query)")
    llm_a = MockLLM(model="mock-react")
    ans_a = run_single("What is the return window for footwear?", llm_a)
    show(llm_a, ans_a)
    assert RUNS["rag_lookup"] == ["What is the return window for footwear?"]
    assert RUNS["check_order_status"] == []
    assert "[DUMMY KB]" in ans_a and TEMPLATE_PLACEHOLDER not in ans_a
    assert llm_a.decisions[0].startswith("[Support Agent] Action rag_lookup")
    print("    PASS\n")

    # B: order question. The record_id schema can be filled, and the more specific arg wins.
    print("CASE B - order question -> check_order_status(record_id)")
    llm_b = MockLLM(model="mock-react")
    ans_b = run_single("Where is my order NYK-1042?", llm_b)
    show(llm_b, ans_b)
    assert RUNS["check_order_status"] == ["NYK-1042"]
    assert len(RUNS["rag_lookup"]) == 1  # unchanged from case A
    assert '"status": "Shipped"' in ans_b and TEMPLATE_PLACEHOLDER not in ans_b
    print("    PASS\n")

    # C: two agents, sequential. The Composer receives the Retrieval output as context.
    print("CASE C - Retrieval Agent -> Composer (context passing)")
    retr_llm, comp_llm = MockLLM(model="mock-react"), MockLLM(model="mock-react", responder=compose)
    retriever = Agent(role="Retrieval Agent", goal="Fetch policy text.", backstory="KB expert.",
                      llm=retr_llm, tools=[DummyRagLookup()], max_iter=3)
    composer = Agent(role="Response Composer", goal="Write the final reply.", backstory="Writer.",
                     llm=comp_llm, tools=[], max_iter=3)
    q = "Customer query: Can I return sneakers after 10 days?"
    t1 = Task(description=f"Retrieve relevant policy.\n{q}", expected_output="Policy text.", agent=retriever)
    t2 = Task(description=f"Compose the reply.\n{q}", expected_output="Reply.", agent=composer, context=[t1])
    ans_c = Crew(agents=[retriever, composer], tasks=[t1, t2], process=Process.sequential,
                 tracing=False).kickoff().raw
    for d in retr_llm.decisions + comp_llm.decisions:
        print(f"    llm decision : {d}")
    print(f"    final answer : {ans_c}")
    assert ans_c.startswith("Here is what I found") and "[DUMMY KB]" in ans_c
    assert "Provide your complete response" not in ans_c  # CrewAI template trailer stripped
    assert TEMPLATE_PLACEHOLDER not in ans_c
    print("    PASS\n")

    print(f"Tool executions: {RUNS}")
    print("ALL SPIKE CHECKS PASSED")


if __name__ == "__main__":
    with transcript("mock_llm_spike"):
        main()
