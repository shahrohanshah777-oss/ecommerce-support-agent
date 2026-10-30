"""CrewAI tool wrappers for the RAG core (Tasks 3-5) and the order lookup (Task 6).

Each tool records its invocations in `calls` so demos can prove it actually ran.
"""
from __future__ import annotations

import json

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

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

    def _run(self, query: str) -> str:
        self.calls.append(query)
        r = answer(query)
        return json.dumps(
            {
                "tool": self.name,
                "answer": r.answer,
                "grounded": r.grounded,
                "top_similarity": r.top_similarity,
                "sources": r.sources,
                "context": [c.text for c in r.context if c.similarity >= r.threshold],
            },
            ensure_ascii=False,
        )


class OrderStatusTool(BaseTool):
    name: str = "check_order_status"
    description: str = (
        "Look up a Nykaa order by ID. Returns status, order_value_inr and a designed "
        "escalation_score in [0, 1]."
    )
    args_schema: type[BaseModel] = OrderStatusArgs
    calls: list[str] = Field(default_factory=list, exclude=True)

    def _run(self, record_id: str) -> str:
        self.calls.append(record_id)
        return json.dumps({"tool": self.name, **check_order_status(record_id)})
