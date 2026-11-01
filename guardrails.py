"""Input-side guardrails (PII masking, prompt-injection detection) and the
output-side groundedness check.

PII scope: phone numbers and payment-card last-4 digits are fixed-format and are
masked. Customer names and delivery addresses are free text with no reliable
pattern and are out of scope for this keyless, regex-only masker.
"""
from __future__ import annotations

import re
from functools import lru_cache

import numpy as np
from pydantic import BaseModel

import config
from rag.kb import answer_sentences, split_sentences

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
    # real-LLM mode only: NLI entailment probability of each non-verbatim sentence
    entailment: dict[str, float] = {}


_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")


def numbers_in(text: str) -> set[float]:
    """Every number in `text` as a value, so "Rs 6,269" == "6269" and "0.70" == "0.7"."""
    found = set()
    for n in _NUMBER.findall(text):
        try:
            found.add(float(n.replace(",", "")))
        except ValueError:  # e.g. a dotted version string
            continue
    return found


@lru_cache(maxsize=1)
def get_nli():
    """Local NLI cross-encoder (real-LLM mode only); downloaded once on first use."""
    import torch
    from sentence_transformers import CrossEncoder

    kwargs = dict(model_kwargs={"dtype": torch.float32})  # the checkpoint is fp16, which is slow on CPU
    try:
        return CrossEncoder(config.NLI_MODEL, local_files_only=True, **kwargs)
    except OSError:
        print(f"[guardrails] {config.NLI_MODEL} not cached yet - downloading once (~360 MB)...")
        return CrossEncoder(config.NLI_MODEL, **kwargs)


def premise_units(*groups: list[str]) -> list[str]:
    """NLI premises: every evidence sentence, every adjacent pair (facts often span
    two sentences, e.g. "...counted from the delivery date. Footwear ... 15 days"),
    and all the evidence together."""
    units: list[str] = []
    for group in groups:
        sents = [s for text in group for s in split_sentences(text)]
        units += sents + [f"{a} {b}" for a, b in zip(sents, sents[1:])]
    whole = " ".join(t for group in groups for t in group)
    return list(dict.fromkeys(units + ([whole] if whole else [])))


# Contextual header for every premise, as chunks get "<title>: " before embedding:
# KB sentences rarely name Nykaa, and without it the NLI model reads "your Nykaa
# reward points expire..." as NOT entailed by "Reward points expire..." (0.10 vs 0.98).
NLI_PREMISE_PREFIX = "Nykaa policy: "
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}\b)")


def _nli_text(text: str) -> str:
    return _THOUSANDS.sub("", text)  # "Rs 6,269" and "Rs 6269" read the same


def entailment_support(sentences: list[str], premises: list[str], known_numbers: set[float]
                       ) -> list[tuple[str, float, set[float]]]:
    """(sentence, best P(entailment) over the premises, numbers absent from the evidence)."""
    if not sentences or not premises:
        return [(s, 0.0, numbers_in(s) - known_numbers) for s in sentences]
    model = get_nli()
    label = {v.lower(): k for k, v in model.config.id2label.items()}["entailment"]
    probs = model.predict([(NLI_PREMISE_PREFIX + _nli_text(p), _nli_text(s)) for s in sentences for p in premises],
                          apply_softmax=True)
    probs = np.asarray(probs)[:, label].reshape(len(sentences), len(premises))
    return [(s, round(float(row.max()), 3), numbers_in(s) - known_numbers) for s, row in zip(sentences, probs)]


def check_groundedness(
    answer: str,
    intent: str,
    top_similarity: float,
    threshold: float,
    context: list[str],
    allowed_sentences: list[str],
    semantic: bool = False,
    query: str = "",
    require_entailment: bool = True,
) -> GroundednessResult:
    """Refuse unless every sentence of the answer is backed by evidence.

    A sentence is supported if it appears verbatim in the retrieved context, or is
    one of `allowed_sentences` (order facts rendered from the lookup tool's record,
    and fixed system templates such as the fallback).

    `semantic=True` (real-LLM mode, where answers paraphrase) instead requires
    every non-quoted sentence to have all its numbers in the evidence or in the
    customer's own query, and scores it with a local NLI model. With
    `require_entailment=True` it must also reach config.GROUNDING_MIN_ENTAILMENT.
    The pipeline sets it False when the Autogen reviewer (a real LLM) has already
    judged support: the small NLI model wrongly rejects correct sentences that need
    world knowledge ("a dress is apparel"; demos/real_llm_grounding_calibrate.py),
    so its scores are then recorded, not enforced.
    """
    if intent == "policy" and top_similarity < threshold:
        return GroundednessResult(
            passed=False,
            reason=f"retrieved context does not support the question "
                   f"(top-1 similarity {top_similarity:.3f} < {threshold})",
        )
    if not answer_sentences(answer):
        return GroundednessResult(passed=False, reason="empty answer")
    evidence = " ".join(context)
    allowed = set(allowed_sentences)
    unsupported = []
    for sentence in answer_sentences(answer):
        if sentence in allowed or sentence in evidence:
            continue
        unsupported.append(sentence)
    entailment: dict[str, float] = {}
    if semantic and unsupported:
        known = numbers_in(" ".join(context + list(allowed_sentences)) + " " + query)
        premises = premise_units(context, list(allowed_sentences))
        checked, unsupported = entailment_support(unsupported, premises, known), []
        for sentence, prob, missing in checked:
            entailment[sentence] = prob
            if missing or (require_entailment and prob < config.GROUNDING_MIN_ENTAILMENT):
                unsupported.append(sentence)
    if unsupported:
        return GroundednessResult(
            passed=False,
            reason=f"{len(unsupported)} sentence(s) not supported by retrieved context or order data",
            unsupported=unsupported, entailment=entailment,
        )
    if entailment and require_entailment:
        return GroundednessResult(passed=True, entailment=entailment,
                                  reason="every sentence is quoted from, or entailed by, retrieved context "
                                         "or order data")
    if entailment:
        return GroundednessResult(passed=True, entailment=entailment,
                                  reason="every sentence is quoted, or reviewer-approved with all its numbers "
                                         "found in the evidence (NLI scores advisory)")
    return GroundednessResult(passed=True, reason="every sentence is backed by retrieved context or order data")
