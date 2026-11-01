"""Every crew response conforms to the SupportResponse Pydantic schema.

Enforced at three points:
  1. The Composer task declares output_pydantic=SupportResponse, so CrewAI passes it
     to the LLM as response_model (the structured-output / response_format contract).
  2. MockLLM validates its own final answer against that response_model before
     returning it, as a real structured-output API would.
  3. run_crew() re-validates the crew's final output in code with
     SupportResponse.model_validate(...).

Run:  python -m demos.structured_output
"""
import json

import config  # noqa: F401
from pydantic import ValidationError

from crew.crew import run_crew
from crew.schemas import SupportResponse
from demos._transcript import transcript

QUERIES = [
    "What is the return window for footwear?",
    "Where is my order NYK-1019?",
    "Can I cancel my order NYK-1004?",
    "What is the status of order NYK-9999?",
    "Who won the football world cup in 2022?",
]

INVALID_PAYLOADS = {
    "unknown intent": {"intent": "chit_chat"},
    "escalation_score > 1": {"order": {"record_id": "NYK-1001", "category": "Beauty", "status": "Shipped",
                                       "order_value_inr": 999, "days_since_created": 3,
                                       "delayed_shipment": False, "escalation_score": 1.5, "escalate": True}},
    "empty answer": {"answer": ""},
    "extra field (hallucinated key)": {"discount_code": "FREE50"},
}


def main() -> None:
    schema = SupportResponse.model_json_schema()
    print("SupportResponse fields (required marked *):")
    for name, prop in schema["properties"].items():
        star = "*" if name in schema.get("required", []) else " "
        kind = prop.get("type") or prop.get("enum") or [p.get("type", p.get("$ref")) for p in prop.get("anyOf", [])]
        print(f"  {star} {name:26s} {kind}")

    print("\n1) Every crew response validates:")
    for q in QUERIES:
        run = run_crew(q)
        validated = SupportResponse.model_validate(run.response.model_dump())
        print(f"  VALID  intent={validated.intent:13s} grounded={str(validated.grounded):5s} "
              f"escalate={str(validated.escalate):5s} sources={validated.sources} | {q}")

    print("\n2) Malformed outputs are rejected by the same schema:")
    base = run_crew(QUERIES[0]).response.model_dump()
    for label, patch in INVALID_PAYLOADS.items():
        try:
            SupportResponse.model_validate({**base, **patch})
            raise AssertionError(f"{label} was accepted")
        except ValidationError as e:
            err = e.errors()[0]
            print(f"  REJECTED {label:32s} -> {'.'.join(map(str, err['loc']))}: {err['msg']}")

    print("\nExample validated response (JSON):")
    print(json.dumps(run_crew(QUERIES[1]).response.model_dump(), indent=2))


if __name__ == "__main__":
    with transcript("structured_output"):
        main()
