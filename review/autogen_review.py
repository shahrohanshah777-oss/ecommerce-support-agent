"""Autogen review stage after the crew's draft answer.

A 2-agent RoundRobinGroupChat (Policy-Compliance-Reviewer -> Final-Editor),
bounded with max_turns=2. The Final-Editor answers with a structured
ReviewVerdict via output_content_type, so the team is constructed with
custom_message_types=[StructuredMessage[ReviewVerdict]].

Input: the Composer's draft + the original retrieved context + verified facts
(order sentences rendered from the lookup tool's record and standard phrases).

Under MOCK_LLM each agent runs on a RuleBasedChatClient. As with the CrewAI mock,
the editor reads "UNSUPPORTED:" lines ONLY from the reviewer agent's own message
(source == "policy_reviewer"), never from system prompts, whose instructions
contain the literal example text "UNSUPPORTED: <sentence>".
With MOCK_LLM=false both agents run on the configured real LLM (llm/providers.py)
with the same system prompts and the same structured ReviewVerdict output.
"""
from __future__ import annotations

import asyncio
import re
import warnings
from typing import Sequence

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import StructuredMessage
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core.models import LLMMessage, UserMessage
from pydantic import BaseModel

import config
from rag.kb import answer_sentences
from review.mock_client import RuleBasedChatClient

# autogen-ext 0.7.5 logs a harmless pydantic serializer warning for every
# structured-output response from a real model.
warnings.filterwarnings("ignore", message="Pydantic serializer warnings", category=UserWarning)

REVIEWER, EDITOR = "policy_reviewer", "final_editor"
REVIEW_FALLBACK = ("I'm sorry, I can't confirm an answer to that from Nykaa's policy documents or your "
                   "order records. Please contact Nykaa customer support for help.")


class ReviewVerdict(BaseModel):
    approved: bool
    final_answer: str
    reason: str


TASK_TEMPLATE = """Review the DRAFT ANSWER before it is sent to a Nykaa customer.

DRAFT ANSWER:
{draft}

RETRIEVED CONTEXT:
{context}

VERIFIED FACTS:
{facts}
"""

REVIEWER_SYSTEM = (
    "You are the Policy-Compliance Reviewer for Nykaa customer support. Check every sentence of the "
    "DRAFT ANSWER against the RETRIEVED CONTEXT and VERIFIED FACTS. List each sentence that is not "
    "supported on its own line as 'UNSUPPORTED: <sentence>'. If all are supported, say so."
)
EDITOR_SYSTEM = (
    "You are the Final Editor. Using the reviewer's findings, either approve the draft unchanged or "
    "remove every sentence the reviewer marked 'UNSUPPORTED: <sentence>'. Never add new claims. "
    "Reply as JSON with fields approved, final_answer, reason."
)
# Real-LLM mode only (the mock's prompts, and so its metered tokens, stay unchanged).
EDITOR_REAL_SUFFIX = " Set approved to false whenever final_answer differs from the draft in any way."


# --- task parsing ----------------------------------------------------------------
def _section(text: str, name: str) -> str:
    m = re.search(rf"{name}:\n(.*?)(?:\n\n[A-Z ]+:\n|\Z)", text, re.DOTALL)
    return m.group(1).strip() if m else ""


def _bullets(block: str) -> list[str]:
    return [line[2:].strip() for line in block.splitlines() if line.startswith("- ")]


def _task(messages: Sequence[LLMMessage]) -> str:
    for m in messages:
        if isinstance(m, UserMessage) and m.source == "user" and "DRAFT ANSWER:" in str(m.content):
            return str(m.content)
    return ""


def draft_sentences(draft: str) -> list[str]:
    return answer_sentences(draft)


# --- mock "models" ---------------------------------------------------------------
def reviewer_respond(messages: Sequence[LLMMessage], json_output) -> str:
    task = _task(messages)
    draft = _section(task, "DRAFT ANSWER")
    evidence = " ".join(_bullets(_section(task, "RETRIEVED CONTEXT")))
    facts = set(_bullets(_section(task, "VERIFIED FACTS")))
    sentences = draft_sentences(draft)
    unsupported = [s for s in sentences if s not in facts and s not in evidence]
    if not unsupported:
        return f"REVIEW: all {len(sentences)} sentences are supported by the retrieved context or verified facts."
    lines = [f"REVIEW: {len(sentences) - len(unsupported)}/{len(sentences)} sentences supported."]
    lines += [f"UNSUPPORTED: {s}" for s in unsupported]
    return "\n".join(lines)


def editor_respond(messages: Sequence[LLMMessage], json_output) -> str:
    draft = _section(_task(messages), "DRAFT ANSWER")
    review = next((str(m.content) for m in reversed(messages)
                   if isinstance(m, UserMessage) and m.source == REVIEWER), "")
    flagged = [line.split("UNSUPPORTED:", 1)[1].strip() for line in review.splitlines()
               if line.startswith("UNSUPPORTED:")]
    if not flagged:
        verdict = ReviewVerdict(approved=True, final_answer=draft,
                                reason="Reviewer found every sentence supported; approved unchanged.")
    else:
        revised = draft
        for s in flagged:
            revised = revised.replace(s, "")
        revised = " ".join(revised.replace("Relevant policy:  ", "").split())
        if revised.endswith("Relevant policy:"):
            revised = revised[: -len("Relevant policy:")].strip()
        kept = draft_sentences(revised)
        verdict = ReviewVerdict(
            approved=False,
            final_answer=revised if kept else REVIEW_FALLBACK,
            reason=f"Removed {len(flagged)} unsupported sentence(s): " + " | ".join(flagged),
        )
    return verdict.model_dump_json()


# --- the team ------------------------------------------------------------------------
class ReviewOutcome(BaseModel):
    verdict: ReviewVerdict
    conversation: list[tuple[str, str, str]]  # (source, message type, content)
    tokens_used: int = 0


def build_team() -> tuple[RoundRobinGroupChat, list]:
    if config.MOCK_LLM:
        clients = [RuleBasedChatClient(reviewer_respond, REVIEWER), RuleBasedChatClient(editor_respond, EDITOR)]
        editor_system = EDITOR_SYSTEM
    else:
        from llm.providers import autogen_client

        clients = [autogen_client(), autogen_client()]
        editor_system = EDITOR_SYSTEM + EDITOR_REAL_SUFFIX
    reviewer = AssistantAgent(REVIEWER, model_client=clients[0], system_message=REVIEWER_SYSTEM,
                              description="Checks every claim in the draft against the evidence.")
    editor = AssistantAgent(EDITOR, model_client=clients[1], system_message=editor_system,
                            output_content_type=ReviewVerdict,
                            description="Approves or revises the draft; returns a ReviewVerdict.")
    team = RoundRobinGroupChat([reviewer, editor], max_turns=2,
                               custom_message_types=[StructuredMessage[ReviewVerdict]])
    return team, clients


async def _review(draft: str, context: list[str], verified_facts: list[str]) -> ReviewOutcome:
    task = TASK_TEMPLATE.format(
        draft=draft,
        context="\n".join(f"- {c}" for c in context) or "- (none)",
        facts="\n".join(f"- {f}" for f in verified_facts) or "- (none)",
    )
    team, clients = build_team()
    try:
        result = await team.run(task=task)
    finally:
        for client in clients:
            await client.close()
    last = result.messages[-1]
    if not isinstance(last, StructuredMessage) or not isinstance(last.content, ReviewVerdict):
        raise RuntimeError(f"review team ended without a ReviewVerdict ({type(last).__name__})")
    conversation = [(m.source, type(m).__name__,
                     m.content.model_dump_json() if isinstance(m, StructuredMessage) else str(m.content))
                    for m in result.messages]
    tokens = sum(c.total_usage().prompt_tokens + c.total_usage().completion_tokens for c in clients)
    return ReviewOutcome(verdict=last.content, conversation=conversation, tokens_used=tokens)


def review_draft(draft: str, context: list[str], verified_facts: list[str]) -> ReviewOutcome:
    return asyncio.run(_review(draft, context, verified_facts))
