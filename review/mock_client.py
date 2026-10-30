"""Deterministic Autogen ChatCompletionClient for MOCK_LLM mode.

Implements autogen_core.models.ChatCompletionClient so AssistantAgents run
unmodified; each instance is given a `respond(messages, json_output) -> str`
function that plays the role of the model.
"""
from __future__ import annotations

from typing import Any, AsyncGenerator, Callable, Literal, Mapping, Optional, Sequence

from autogen_core import CancellationToken
from autogen_core.models import (
    ChatCompletionClient, CreateResult, LLMMessage, ModelCapabilities, ModelInfo, RequestUsage,
)
from autogen_core.tools import Tool, ToolSchema
from pydantic import BaseModel

Responder = Callable[[Sequence[LLMMessage], Optional[bool | type[BaseModel]]], str]


def _text(messages: Sequence[LLMMessage]) -> str:
    return "\n".join(str(getattr(m, "content", "")) for m in messages)


class RuleBasedChatClient(ChatCompletionClient):
    def __init__(self, respond: Responder, name: str = "mock") -> None:
        self._respond = respond
        self._name = name
        self._usage = RequestUsage(prompt_tokens=0, completion_tokens=0)
        self.calls = 0

    async def create(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Tool | ToolSchema] = [],
        tool_choice: Tool | Literal["auto", "required", "none"] = "auto",
        json_output: Optional[bool | type[BaseModel]] = None,
        extra_create_args: Mapping[str, Any] = {},
        cancellation_token: Optional[CancellationToken] = None,
    ) -> CreateResult:
        self.calls += 1
        content = self._respond(messages, json_output)
        if isinstance(json_output, type) and issubclass(json_output, BaseModel):
            content = json_output.model_validate_json(content).model_dump_json()  # structured output
        usage = RequestUsage(prompt_tokens=self.count_tokens(messages), completion_tokens=len(content) // 4)
        self._usage = RequestUsage(prompt_tokens=self._usage.prompt_tokens + usage.prompt_tokens,
                                   completion_tokens=self._usage.completion_tokens + usage.completion_tokens)
        return CreateResult(finish_reason="stop", content=content, usage=usage, cached=False)

    async def create_stream(self, messages: Sequence[LLMMessage], **kwargs: Any) -> AsyncGenerator[str | CreateResult, None]:
        yield await self.create(messages, **kwargs)

    async def close(self) -> None:
        return None

    def actual_usage(self) -> RequestUsage:
        return self._usage

    def total_usage(self) -> RequestUsage:
        return self._usage

    def count_tokens(self, messages: Sequence[LLMMessage], *, tools: Sequence[Tool | ToolSchema] = []) -> int:
        return (len(_text(messages)) + 3) // 4

    def remaining_tokens(self, messages: Sequence[LLMMessage], *, tools: Sequence[Tool | ToolSchema] = []) -> int:
        return 128_000 - self.count_tokens(messages)

    @property
    def capabilities(self) -> ModelCapabilities:  # type: ignore[override]
        return {"vision": False, "function_calling": False, "json_output": True}  # type: ignore[return-value]

    @property
    def model_info(self) -> ModelInfo:
        return ModelInfo(vision=False, function_calling=False, json_output=True,
                         structured_output=True, family="unknown", multiple_system_messages=True)
