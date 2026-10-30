"""Task 10 - input-side guardrails (PII masking, prompt-injection detection) and the
output-side groundedness check.

PII scope: phone numbers and payment-card last-4 digits are fixed-format and are
masked. Customer names and delivery addresses are free text with no reliable
pattern and are out of scope for this keyless, regex-only masker.
"""
from __future__ import annotations

import re

from pydantic import BaseModel

from rag.kb import split_sentences

# --- PII masking ---------------------------------------------------------------
# Full card numbers (13-19 digits, optional space/dash groups) are masked whole.
_FULL_CARD = re.compile(r"\b(?:\d[ -]?){12,18}\d\b")
# Indian mobile numbers: optional +91 / 91 / 0 prefix, then 10 digits starting 6-9,
# optionally split 5+5 by a space or dash.
_PHONE = re.compile(r"(?<![\w-])(?:\+91[\s-]?|91[\s-]?|0)?[6-9]\d{4}[\s-]?\d{5}(?!\d)")
# Card last-4: four digits only when a card cue precedes them, so order IDs
# (NYK-1009) and amounts (Rs 1499) are never touched.
_CARD_LAST4 = re.compile(
    r"(?P<cue>\b(?:card|visa|mastercard|master card|rupay|amex|debit|credit)\b[^.\d\n]{0,40}?"
    r"(?:ending(?:\s+(?:in|with))?|last\s*(?:4|four)(?:\s*digits)?(?:\s*(?:is|are|:))?|no\.?|number)?\s*"
    r"(?:[x*•]{2,}[\s-]*)*)(?P<digits>\d{4})(?!\d)",
    re.IGNORECASE,
)
_MASKED_STARS = re.compile(r"(?:[x*•]{4}[\s-]?){2,3}(?P<digits>\d{4})(?!\d)", re.IGNORECASE)

PHONE_TOKEN = "[PHONE]"
CARD_LAST4_TOKEN = "[CARD_LAST4]"
CARD_TOKEN = "[CARD_NUMBER]"


class MaskResult(BaseModel):
    text: str
    findings: dict[str, int]

    @property
    def masked(self) -> bool:
        return bool(self.findings)


def mask_pii(text: str) -> MaskResult:
    findings: dict[str, int] = {}

    def sub(pattern: re.Pattern, repl, label: str, s: str) -> str:
        out, n = pattern.subn(repl, s)
        if n:
            findings[label] = findings.get(label, 0) + n
        return out

    out = sub(_FULL_CARD, CARD_TOKEN, "card_number", text)
    out = sub(_PHONE, PHONE_TOKEN, "phone", out)
    out = sub(_MASKED_STARS, CARD_LAST4_TOKEN, "card_last4", out)
    out = sub(_CARD_LAST4, lambda m: m.group("cue") + CARD_LAST4_TOKEN, "card_last4", out)
    return MaskResult(text=out, findings=findings)


# --- Prompt-injection detection ---------------------------------------------------
INJECTION_PATTERNS = {
    "ignore_instructions": r"\b(ignore|disregard|forget|override)\b.{0,30}\b(previous|prior|above|earlier|all|your|the)\b.{0,20}\b(instructions?|rules|prompts?|guidelines|polic(y|ies))\b",
    "role_override": r"\b(you are now|from now on you are|act as (an?|the|my|if|though)|pretend (to be|you are)|roleplay as)\b",
    "prompt_exfiltration": r"\b(reveal|show|print|repeat|leak)\b.{0,30}\b(system prompt|hidden prompt|your (prompt|instructions)|initial instructions)\b",
    "jailbreak_keyword": r"\b(jailbreak|developer mode|DAN mode|do anything now)\b",
    "fake_system_turn": r"(^|\n)\s*(system|assistant)\s*:",
    "policy_bypass": r"\b(bypass|disable|turn off)\b.{0,20}\b(guardrails?|safety|filters?|restrictions)\b",
}
_INJECTION = {name: re.compile(p, re.IGNORECASE | re.DOTALL) for name, p in INJECTION_PATTERNS.items()}


class InjectionResult(BaseModel):
    is_injection: bool
    matched: list[str]


def detect_injection(text: str) -> InjectionResult:
    matched = [name for name, rx in _INJECTION.items() if rx.search(text)]
    return InjectionResult(is_injection=bool(matched), matched=matched)


INJECTION_REFUSAL = (
    "I can't help with that request. I can answer questions about Nykaa orders and "
    "policies such as returns, refunds, delivery and cancellations."
)


# --- Output-side groundedness check -----------------------------------------------
UNGROUNDED_REFUSAL = (
    "I'm sorry, I can't confirm an answer to that from Nykaa's policy documents or your "
    "order records, so I won't guess. Please contact Nykaa customer support for help."
)


class GroundednessResult(BaseModel):
    passed: bool
    reason: str
    unsupported: list[str] = []


def check_groundedness(
    answer: str,
    intent: str,
    top_similarity: float,
    threshold: float,
    context: list[str],
    allowed_sentences: list[str],
) -> GroundednessResult:
    """Refuse unless every sentence of the answer is backed by evidence.

    A sentence is supported if it appears verbatim in the retrieved context, or is
    one of `allowed_sentences` (order facts rendered from the lookup tool's record,
    and fixed system templates such as the fallback).
    """
    if intent == "policy" and top_similarity < threshold:
        return GroundednessResult(
            passed=False,
            reason=f"retrieved context does not support the question "
                   f"(top-1 similarity {top_similarity:.3f} < {threshold})",
        )
    evidence = " ".join(context)
    allowed = set(allowed_sentences)
    unsupported = []
    for sentence in split_sentences(answer.replace("Relevant policy: ", "")):
        if sentence in allowed or sentence.replace("...", "").strip() in evidence:
            continue
        unsupported.append(sentence)
    if unsupported:
        return GroundednessResult(
            passed=False,
            reason=f"{len(unsupported)} sentence(s) not supported by retrieved context or order data",
            unsupported=unsupported,
        )
    return GroundednessResult(passed=True, reason="every sentence is backed by retrieved context or order data")
