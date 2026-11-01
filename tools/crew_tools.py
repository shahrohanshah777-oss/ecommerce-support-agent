"""CrewAI tool wrappers for the RAG core and the order lookup.

Each tool records its invocations in `calls` (and what it returned in `outputs`)
so demos can prove it actually ran, and is bound to the agent role it was issued to: `_run` calls
governance.authorize() first, so a tool wired to the wrong agent fails closed
(least autonomy, see governance.py).
"""
from __future__ import annotations

import json

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from governance import authorize
from rag.generate import answer
from tools.order_status import check_order_status


class RagLookupArgs(BaseModel):
    query: str = Field(..., description="The customer's question about Nykaa policies.")


class OrderStatusArgs(BaseModel):
    record_id: str = Field(..., description="Nykaa order ID, e.g. NYK-1042.")


class RagLookupTool(BaseTool):
    name: str = "rag_lookup"
    description: str = (
        "Search the Nykaa policy knowledge base and return a grounded answer with its "
        "retrieved context, or an 'I don't know' fallback when nothing relevant is found."
    )
    args_schema: type[BaseModel] = RagLookupArgs
    calls: list[str] = Field(default_factory=list, exclude=True)
    outputs: list[str] = Field(default_factory=list, exclude=True)
    bound_role: str | None = Field(default=None, exclude=True)

    def _run(self, query: str) -> str:
        authorize(self.name, self.bound_role)
        self.calls.append(query)
        r = answer(query)
        out = json.dumps(
            {
                "tool": self.name,
                "answer": r.answer,
                "grounded": r.grounded,
                "top_similarity": r.top_similarity,
                "sources": r.sources,
                "context": r.evidence,  # everything the answer may quote (small-to-big)
            },
            ensure_ascii=False,
        )
        self.outputs.append(out)
        return out


class OrderStatusTool(BaseTool):
    name: str = "check_order_status"
    description: str = (
        "Look up a Nykaa order by ID. Returns status, order_value_inr and a designed "
        "escalation_score in [0, 1]."
    )
    args_schema: type[BaseModel] = OrderStatusArgs
    calls: list[str] = Field(default_factory=list, exclude=True)
    outputs: list[str] = Field(default_factory=list, exclude=True)
    bound_role: str | None = Field(default=None, exclude=True)

    def _run(self, record_id: str) -> str:
        authorize(self.name, self.bound_role)  # only the Lookup Agent may read order data
        self.calls.append(record_id)
        out = json.dumps({"tool": self.name, **check_order_status(record_id)})
        self.outputs.append(out)
        return out
