"""Real-LLM mode (MOCK_LLM=false): one OpenAI-compatible endpoint for every LLM call site.

    grounded generation (Task 4)  -> complete()
    CrewAI crew (Task 7)          -> crew_llm()        native function calling
    Autogen review (Task 14)      -> autogen_client()  structured ReviewVerdict output
    LLM-as-judge (Task 13)        -> complete(json_mode=True)

Model, endpoint and the *name* of the key variable come from config (LLM_MODEL,
LLM_BASE_URL, LLM_API_KEY_ENV); the default is Groq. Nothing here is called under
MOCK_LLM, which stays keyless and offline.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

import config


class MissingAPIKeyError(RuntimeError):
    pass


def api_key() -> str:
    key = os.environ.get(config.LLM_API_KEY_ENV, "").strip()
    if not key:
        raise MissingAPIKeyError(
            f"MOCK_LLM=false needs an API key in the {config.LLM_API_KEY_ENV} environment variable "
            f"(endpoint {config.LLM_BASE_URL}). Set it, or run with MOCK_LLM=true."
        )
    return key


def crew_llm():
    """A CrewAI LLM on the configured endpoint (CrewAI's native OpenAI provider)."""
    from crewai.llms.providers.openai.completion import OpenAICompletion

    return OpenAICompletion(model=config.LLM_MODEL, base_url=config.LLM_BASE_URL, api_key=api_key(),
                            temperature=config.LLM_TEMPERATURE, max_retries=config.LLM_MAX_RETRIES)


def autogen_client():
    """An Autogen ChatCompletionClient on the configured endpoint."""
    from autogen_ext.models.openai import OpenAIChatCompletionClient

    return OpenAIChatCompletionClient(
        model=config.LLM_MODEL, base_url=config.LLM_BASE_URL, api_key=api_key(),
        temperature=config.LLM_TEMPERATURE, max_retries=config.LLM_MAX_RETRIES,
        # Non-OpenAI model names need their capabilities declared.
        model_info={"vision": False, "function_calling": True, "json_output": True,
                    "structured_output": True, "family": "unknown", "multiple_system_messages": False},
    )


# Real models write typographic Unicode (gpt-oss: U+202F narrow no-break space in
# "15 days", U+2011 non-breaking hyphen); map it to plain text so substring checks,
# templates and logs see what a customer reads.
_PLAIN = str.maketrans({"\u00a0": " ", "\u2007": " ", "\u2009": " ", "\u202f": " ",
                        "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": " - ",
                        "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"'})


def plain_text(text: str) -> str:
    return re.sub(r" {2,}", " ", text.translate(_PLAIN))


@dataclass
class Completion:
    text: str
    tokens: int


_client = None


def complete(prompt: str, json_mode: bool = False) -> Completion:
    """One chat completion for a single-prompt call site (generation, judge)."""
    global _client
    if _client is None:
        from openai import OpenAI

        _client = OpenAI(base_url=config.LLM_BASE_URL, api_key=api_key(), max_retries=config.LLM_MAX_RETRIES)
    extra = {"response_format": {"type": "json_object"}} if json_mode else {}
    r = _client.chat.completions.create(model=config.LLM_MODEL, temperature=config.LLM_TEMPERATURE,
                                        messages=[{"role": "user", "content": prompt}], **extra)
    return Completion(text=plain_text(r.choices[0].message.content or "").strip(),
                      tokens=r.usage.total_tokens if r.usage else 0)
