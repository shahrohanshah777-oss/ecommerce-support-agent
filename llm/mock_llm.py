"""Deterministic, offline "LLM" for CrewAI's ReAct loop (MOCK_LLM mode).

Extends `crewai.llms.base_llm.BaseLLM` (CrewAI's documented extension point for
non-litellm models). Each call it decides between:

  * emitting one tool call   ->  Thought / Action / Action Input
  * emitting the final answer ->  Thought / Final Answer

The two pitfalls called out in the brief are handled explicitly:

1. **Template "Observation:" text.** CrewAI's system prompt literally contains
   "Observation: the result of the action" on every call. Tool results are only
   read from *assistant* messages that contain this mock's own "Action:" line
   (CrewAI appends "\nObservation: <result>" to that message after the tool
   runs). System messages are never searched for observations.

2. **Dispatch by schema, not by name.** Tool arguments are filled from each
   tool's declared `args_schema` (its required property names), never by
   matching substrings of the tool's name. A tool named `rag_lookup` with a
   `query` argument therefore receives a query, not an order ID.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable

from pydantic import Field, PrivateAttr

import config
from crewai.llms.base_llm import BaseLLM

# --- Argument extractors (keyed by declared argument name) -------------------
# "Specific" args are pattern-matched from the task; a tool whose specific args
# resolve is preferred over a tool that only needs free text.
SPECIFIC_ARG_EXTRACTORS: dict[str, Callable[[str], str | None]] = {
    "record_id": lambda text: _first_match(config.ORDER_ID_PATTERN, text),
}
FREE_TEXT_ARGS = {"query", "question", "text", "input"}

_QUERY_LINE = re.compile(r"Customer query:\s*(.+)", re.IGNORECASE)
_CONTEXT_MARKER = "This is the context you're working with:"
# Trailers CrewAI appends after the task input (translations/en.json "task_no_tools" / "task").
_TASK_TRAILERS = ("\n\nProvide your complete response:", "\n\nBegin! This is VERY important")
_TOOL_CALL_LINE = re.compile(r"^\s*Action:\s*\S+", re.MULTILINE)
_OBSERVATION = "\nObservation:"


def _first_match(pattern: str, text: str) -> str | None:
    m = re.search(pattern, text)
    return m.group(0) if m else None


@dataclass
class MockTurn:
    """Everything a responder needs to write a final answer."""

    agent_role: str
    task_text: str
    query: str
    context: str | None
    observation: str | None
    tool_name: str | None


# --- Pure parsing helpers (unit-tested in tests/test_mock_llm.py) ------------
def as_messages(messages: str | list[dict[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]
    return [m if isinstance(m, dict) else {"role": "user", "content": str(m)} for m in messages]


def _content(msg: dict[str, Any]) -> str:
    c = msg.get("content")
    return c if isinstance(c, str) else str(c or "")


def extract_task_text(msgs: list[dict[str, Any]]) -> str:
    """The task prompt: first user message (CrewAI's "Current Task: ..." turn)."""
    for m in msgs:
        if m.get("role") == "user":
            return _content(m)
    return ""


def extract_query(task_text: str) -> str:
    """The customer's question. Tasks state it on a `Customer query:` line;
    otherwise fall back to the first non-empty line of the task."""
    m = _QUERY_LINE.search(task_text)
    if m:
        return m.group(1).strip().strip('"')
    for line in task_text.splitlines():
        line = line.replace("Current Task:", "").strip()
        if line:
            return line
    return ""


def extract_context(task_text: str) -> str | None:
    """Output of upstream tasks, which CrewAI appends after a fixed marker."""
    if _CONTEXT_MARKER not in task_text:
        return None
    context = task_text.split(_CONTEXT_MARKER, 1)[1]
    for trailer in _TASK_TRAILERS:
        context = context.split(trailer, 1)[0]
    return context.strip() or None


def latest_observation(msgs: list[dict[str, Any]]) -> tuple[str | None, str | None]:
    """(tool_name, observation) from the most recent tool step, or (None, None).

    Only assistant messages carrying this model's own `Action:` line are
    inspected; the system prompt's example "Observation:" text is never read.
    """
    for m in reversed(msgs):
        if m.get("role") != "assistant":
            continue
        text = _content(m)
        if not _TOOL_CALL_LINE.search(text) or _OBSERVATION not in text:
            continue
        tool = re.search(r"^\s*Action:\s*(\S+)", text, re.MULTILINE).group(1)
        return tool, text.rsplit(_OBSERVATION, 1)[1].strip()
    return None, None


def build_args(schema: dict[str, Any], task_text: str, query: str) -> tuple[dict[str, str], int] | None:
    """Fill a tool's required args from its JSON schema.

    Returns (args, specificity) or None if any required arg can't be filled.
    specificity = number of pattern-matched (non free-text) args.
    """
    props = schema.get("properties", {})
    required = schema.get("required", list(props))
    args: dict[str, str] = {}
    specificity = 0
    for name in required:
        if name in SPECIFIC_ARG_EXTRACTORS:
            value = SPECIFIC_ARG_EXTRACTORS[name](task_text)
            if value is None:
                return None
            args[name] = value
            specificity += 1
        elif name in FREE_TEXT_ARGS and props.get(name, {}).get("type", "string") == "string":
            if not query:
                return None
            args[name] = query
        else:
            return None  # unknown argument: this mock can't fill it, so don't call
    return args, specificity


def choose_tool(tools: list[Any], task_text: str, query: str) -> tuple[str, dict[str, str]] | None:
    """Pick the callable tool with the most specific resolvable args (ties: list order)."""
    best: tuple[int, int, str, dict[str, str]] | None = None
    for idx, tool in enumerate(tools):
        schema = tool.args_schema.model_json_schema()
        built = build_args(schema, task_text, query)
        if built is None:
            continue
        args, spec = built
        key = (spec, -idx)
        if best is None or key > best[:2]:
            best = (spec, -idx, tool.name, args)
    return (best[2], best[3]) if best else None


def default_responder(turn: MockTurn) -> str:
    if turn.observation is not None:
        return turn.observation
    if turn.context:
        return turn.context
    return "I don't know."


# --- The LLM -----------------------------------------------------------------
class MockLLM(BaseLLM):
    """Rule-based ReAct model. Zero network, zero API keys, fully deterministic."""

    llm_type: str = "mock"
    provider: str = "mock"
    # Optional per-agent hook that writes the Final Answer text.
    responder: Callable[[MockTurn], str] | None = Field(default=None, exclude=True)

    _calls: int = PrivateAttr(default=0)
    _decisions: list[str] = PrivateAttr(default_factory=list)

    @property
    def call_count(self) -> int:
        return self._calls

    @property
    def decisions(self) -> list[str]:
        return list(self._decisions)

    def supports_function_calling(self) -> bool:
        return False  # keep CrewAI on the text ReAct loop

    def get_input_tokens_count(self, messages, tools=None, response_model=None) -> int:
        text = messages if isinstance(messages, str) else "".join(_content(m) for m in as_messages(messages))
        return (len(text) + 3) // 4  # same 4-chars/token estimate, without the warning

    def call(
        self,
        messages,
        tools=None,
        callbacks=None,
        available_functions=None,
        from_task=None,
        from_agent=None,
        response_model=None,
        **kwargs: Any,
    ) -> str:
        self._calls += 1
        msgs = as_messages(messages)
        role = getattr(from_agent, "role", "unknown")
        task_text = extract_task_text(msgs)
        query = extract_query(task_text)
        tool_name, observation = latest_observation(msgs)
        agent_tools = list(getattr(from_agent, "tools", None) or [])

        if observation is None and agent_tools:
            choice = choose_tool(agent_tools, task_text, query)
            if choice is not None:
                name, args = choice
                self._decisions.append(f"[{role}] Action {name} {args}")
                return (
                    f"Thought: I need the `{name}` tool to answer this.\n"
                    f"Action: {name}\n"
                    f"Action Input: {json.dumps(args, ensure_ascii=False)}"
                )

        turn = MockTurn(
            agent_role=role,
            task_text=task_text,
            query=query,
            context=extract_context(task_text),
            observation=observation,
            tool_name=tool_name,
        )
        answer = (self.responder or default_responder)(turn)
        if response_model is not None:
            # Structured output: like a real response_format API, never emit
            # text that does not conform to the requested schema.
            answer = response_model.model_validate_json(answer).model_dump_json()
        self._decisions.append(f"[{role}] Final Answer ({len(answer)} chars)")
        return f"Thought: I now know the final answer\nFinal Answer: {answer}"
