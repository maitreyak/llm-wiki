"""Thin wrapper around the Anthropic SDK.

All compiler/error-book calls use structured outputs (a JSON schema via
``output_config.format``) so responses are guaranteed parseable; the
retrieval agent uses the tool-use loop in ``agent/loop.py`` directly.
Streaming is used everywhere so long generations don't hit HTTP timeouts.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import anthropic


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    calls: int = 0

    def add(self, usage) -> None:
        self.calls += 1
        self.input_tokens += usage.input_tokens or 0
        self.output_tokens += usage.output_tokens or 0
        self.cache_read_input_tokens += getattr(usage, "cache_read_input_tokens", 0) or 0
        self.cache_creation_input_tokens += (
            getattr(usage, "cache_creation_input_tokens", 0) or 0
        )

    def __str__(self) -> str:
        return (
            f"{self.calls} calls, {self.input_tokens} in "
            f"({self.cache_read_input_tokens} cached), {self.output_tokens} out"
        )


class RefusalError(RuntimeError):
    pass


class TruncatedOutputError(RuntimeError):
    pass


@dataclass
class LLM:
    client: anthropic.Anthropic = field(default_factory=anthropic.Anthropic)
    usage: Usage = field(default_factory=Usage)
    max_attempts: int = 8

    def message(
        self,
        *,
        model: str,
        system: str | list[dict],
        messages: list[dict],
        max_tokens: int,
        tools: list[dict] | None = None,
        schema: dict | None = None,
        effort: str | None = None,
    ) -> anthropic.types.Message:
        """One streamed Messages call with retry on transient failures."""
        kwargs: dict = {
            "model": model,
            "system": system,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
        if schema:
            kwargs["output_config"] = {
                "format": {"type": "json_schema", "schema": schema}
            }
        # Haiku-tier models reject the effort parameter (400).
        if effort and not model.startswith("claude-haiku"):
            kwargs.setdefault("output_config", {})["effort"] = effort

        last_exc: Exception | None = None
        for attempt in range(self.max_attempts):
            try:
                with self.client.messages.stream(**kwargs) as stream:
                    response = stream.get_final_message()
                self.usage.add(response.usage)
                if response.stop_reason == "refusal":
                    raise RefusalError(
                        f"model declined the request "
                        f"(category: {getattr(response.stop_details, 'category', None)})"
                    )
                return response
            except anthropic.APIStatusError as exc:
                # Retry transient statuses: 429 rate limit, >=500 server
                # errors, and 529 overloaded (raised as bare APIStatusError).
                if exc.status_code != 429 and exc.status_code < 500:
                    raise
                last_exc = exc
                # Up to ~4.5 min of total patience — overload incidents flap
                # for minutes at a time; skipping is costlier than waiting.
                time.sleep(min(2**attempt * 2.0, 90.0))
            except anthropic.APIConnectionError as exc:
                last_exc = exc
                time.sleep(2.0)
        raise last_exc  # type: ignore[misc]

    def structured(
        self,
        *,
        model: str,
        system: str | list[dict],
        user: str,
        schema: dict,
        max_tokens: int,
        effort: str | None = None,
    ) -> dict:
        # A response that hits max_tokens is truncated mid-JSON; retry once
        # with double the budget before giving up.
        for attempt_tokens in (max_tokens, max_tokens * 2):
            response = self.message(
                model=model,
                system=system,
                messages=[{"role": "user", "content": user}],
                max_tokens=attempt_tokens,
                schema=schema,
                effort=effort,
            )
            if response.stop_reason == "max_tokens":
                continue
            text = next(b.text for b in response.content if b.type == "text")
            return json.loads(text)
        raise TruncatedOutputError(
            f"structured output still truncated at {max_tokens * 2} tokens"
        )


def cached_system(text: str) -> list[dict]:
    """System prompt block with a cache breakpoint (stable across a batch)."""
    return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]


def make_llm(config):
    """Build the LLM backend selected by config.provider."""
    if config.provider == "ollama":
        from .ollama_llm import OllamaLLM

        return OllamaLLM(base_url=config.ollama_base_url, num_ctx=config.ollama_num_ctx)
    if config.provider == "claude-cli":
        from .claude_cli_llm import ClaudeCliLLM

        return ClaudeCliLLM()
    if config.provider == "anthropic":
        return LLM()
    raise ValueError(
        f"unknown provider {config.provider!r} (expected anthropic, claude-cli, or ollama)"
    )
