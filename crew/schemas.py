"""Task 9 - the structured-output schema every crew response must conform to."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class OrderInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    record_id: str = Field(pattern=r"^NYK-\d{4,6}$")
    category: str
    status: str
    order_value_inr: int = Field(ge=0)
    days_since_created: int = Field(ge=0, le=30)
    delayed_shipment: bool
    escalation_score: float = Field(ge=0.0, le=1.0)
    escalate: bool


class SupportResponse(BaseModel):
    """Final draft answer produced by the Response Composer agent."""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1, description="Customer-facing reply.")
    intent: Literal["policy", "order_status", "out_of_scope"]
    grounded: bool = Field(description="True only if every claim comes from the KB or the orders dataset.")
    sources: list[str] = Field(default_factory=list, description="KB doc_ids and/or 'orders_dataset'.")
    order: OrderInfo | None = None
    escalate: bool = False
    retrieval_top_similarity: float = Field(ge=-1.0, le=1.0)
