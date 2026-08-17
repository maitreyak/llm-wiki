"""Ollama backend: the same interface as ``llm.LLM``, served by a local model.

Translates between the Anthropic-style shapes the rest of the codebase uses
(system blocks, tool_use/tool_result content blocks, input_schema tools) and
Ollama's ``/api/chat``. Structured outputs use Ollama's ``format=<schema>``;
tool calling requires a tools-capable model (qwen2.5, llama3.1+, etc.).

Expect noticeably weaker compilation quality and agentic behavior than the
Claude backend — useful for local development and demos, not benchmarks.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from types import SimpleNamespace

import httpx

from .llm import Usage


class OllamaError(RuntimeError):
    pass


def _system_text(system: str | list[dict]) -> str:
    if isinstance(system, str):
        return system
    return "\n\n".join(b.get("text", "") for b in system)


def anthropic_tools_to_ollama(tools: list[dict]) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t["input_schema"],
            },
        }
        for t in tools
    ]


def anthropic_messages_to_ollama(system: str, messages: list[dict]) -> list[dict]:
    """Flatten Anthropic-style messages (with tool_use / tool_result content
    blocks) into Ollama chat messages (assistant tool_calls + role=tool)."""
    out: list[dict] = [{"role": "system", "content": system}]
    tool_names: dict[str, str] = {}  # tool_use id -> tool name

    for message in messages:
        role, content = message["role"], message["content"]
        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue

        texts: list[str] = []
        tool_calls: list[dict] = []
        tool_results: list[dict] = []
        for block in content:
            btype = block["type"] if isinstance(block, dict) else block.type
            if btype == "text":
                texts.append(block["text"] if isinstance(block, dict) else block.text)
            elif btype == "tool_use":
                bid = block["id"] if isinstance(block, dict) else block.id
                name = block["name"] if isinstance(block, dict) else block.name
                binput = block["input"] if isinstance(block, dict) else block.input
                tool_names[bid] = name
                tool_calls.append({"function": {"name": name, "arguments": binput}})
            elif btype == "tool_result":
                bid = block["tool_use_id"] if isinstance(block, dict) else block.tool_use_id
                result = block["content"] if isinstance(block, dict) else block.content
                tool_results.append(
                    {
                        "role": "tool",
                        "tool_name": tool_names.get(bid, ""),
                        "content": result if isinstance(result, str) else json.dumps(result),
                    }
                )
            # thinking or other block types are dropped

        if role == "assistant":
            entry: dict = {"role": "assistant", "content": "\n".join(texts)}
            if tool_calls:
                entry["tool_calls"] = tool_calls
            out.append(entry)
        else:
            if texts:
                out.append({"role": "user", "content": "\n".join(texts)})
            out.extend(tool_results)
    return out


def ollama_response_to_blocks(message: dict) -> tuple[list, str]:
    """Ollama response message -> (Anthropic-style blocks, stop_reason)."""
    blocks = []
    text = (message.get("content") or "").strip()
    if text:
        blocks.append(SimpleNamespace(type="text", text=text))
    calls = message.get("tool_calls") or []
    for i, call in enumerate(calls):
        fn = call.get("function", {})
        args = fn.get("arguments") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        blocks.append(
            SimpleNamespace(
                type="tool_use",
                id=f"call_{i}_{fn.get('name', 'tool')}",
                name=fn.get("name", ""),
                input=args,
            )
        )
    return blocks, ("tool_use" if calls else "end_turn")


@dataclass
class OllamaLLM:
    base_url: str = "http://localhost:11434"
    num_ctx: int = 16384
    timeout: float = 600.0
    usage: Usage = field(default_factory=Usage)

    def _chat(self, payload: dict) -> dict:
        payload["stream"] = False
        try:
            response = httpx.post(
                f"{self.base_url.rstrip('/')}/api/chat",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise OllamaError(
                f"Ollama error {exc.response.status_code}: {exc.response.text[:300]}"
            ) from exc
        except httpx.HTTPError as exc:
            raise OllamaError(
                f"cannot reach Ollama at {self.base_url} ({exc}); is `ollama serve` running?"
            ) from exc
        data = response.json()
        self.usage.calls += 1
        self.usage.input_tokens += data.get("prompt_eval_count") or 0
        self.usage.output_tokens += data.get("eval_count") or 0
        return data

    def message(
        self,
        *,
        model: str,
        system: str | list[dict],
        messages: list[dict],
        max_tokens: int,
        tools: list[dict] | None = None,
        schema: dict | None = None,
        effort: str | None = None,  # ignored
    ) -> SimpleNamespace:
        payload: dict = {
            "model": model,
            "messages": anthropic_messages_to_ollama(_system_text(system), messages),
            "options": {"num_ctx": self.num_ctx, "num_predict": max_tokens, "temperature": 0},
        }
        if tools:
            payload["tools"] = anthropic_tools_to_ollama(tools)
        if schema:
            payload["format"] = schema
        data = self._chat(payload)
        blocks, stop_reason = ollama_response_to_blocks(data.get("message", {}))
        if tools and not blocks:
            # Known Ollama failure mode: the model emits a malformed tool call
            # and the parser swallows it — tokens were generated (eval_count>0)
            # but the message has neither content nor tool_calls. Re-ask the
            # same context without tools to force a plain-text response; the
            # agent loop re-offers tools on the next turn.
            retry = {k: v for k, v in payload.items() if k != "tools"}
            data = self._chat(retry)
            blocks, stop_reason = ollama_response_to_blocks(data.get("message", {}))
        return SimpleNamespace(content=blocks, stop_reason=stop_reason)

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
        response = self.message(
            model=model,
            system=system,
            messages=[{"role": "user", "content": user}],
            max_tokens=max_tokens,
            schema=schema,
        )
        text = "".join(b.text for b in response.content if b.type == "text")
        return json.loads(text)
