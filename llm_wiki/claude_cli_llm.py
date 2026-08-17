"""Claude Code CLI backend: same interface as ``llm.LLM``, served by
``claude -p`` (headless print mode).

Why: headless CLI calls authenticate via your Claude login rather than
API credits. The subprocess environment deliberately drops
ANTHROPIC_API_KEY so the CLI cannot silently fall back to API billing.

The CLI has no tool-calling API, so the agent's tools are emulated with
a text protocol: the conversation is rendered as a transcript, and the
model is instructed to reply with either a single JSON tool call or its
final answer. Structured outputs are prompted JSON, parsed with a retry.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from types import SimpleNamespace

from .llm import Usage
from .ollama_llm import extract_textual_tool_calls

FENCE_RE = re.compile(r"```(?:json|python)?\s*(.*?)```", re.DOTALL)


class ClaudeCliError(RuntimeError):
    pass


def _system_text(system: str | list[dict]) -> str:
    if isinstance(system, str):
        return system
    return "\n\n".join(b.get("text", "") for b in system)


def strip_json_fences(text: str) -> str:
    m = FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    return text.strip()


def render_transcript(messages: list[dict]) -> str:
    """Flatten Anthropic-style messages into a plain-text transcript."""
    lines: list[str] = []
    tool_names: dict[str, str] = {}
    for message in messages:
        role, content = message["role"], message["content"]
        if isinstance(content, str):
            lines.append(f"[{role.upper()}]\n{content}")
            continue
        for block in content:
            btype = block["type"] if isinstance(block, dict) else block.type
            if btype == "text":
                text = block["text"] if isinstance(block, dict) else block.text
                if text.strip():
                    lines.append(f"[{role.upper()}]\n{text}")
            elif btype == "tool_use":
                bid = block["id"] if isinstance(block, dict) else block.id
                name = block["name"] if isinstance(block, dict) else block.name
                binput = block["input"] if isinstance(block, dict) else block.input
                tool_names[bid] = name
                lines.append(
                    "[ASSISTANT TOOL CALL]\n"
                    + json.dumps({"name": name, "arguments": binput})
                )
            elif btype == "tool_result":
                bid = block["tool_use_id"] if isinstance(block, dict) else block.tool_use_id
                result = block["content"] if isinstance(block, dict) else block.content
                if not isinstance(result, str):
                    result = json.dumps(result)
                lines.append(f"[TOOL RESULT: {tool_names.get(bid, 'tool')}]\n{result}")
    return "\n\n".join(lines)


def tool_protocol(tools: list[dict]) -> str:
    specs = []
    for t in tools:
        specs.append(
            f"- {t['name']}: {t.get('description', '')}\n"
            f"  arguments schema: {json.dumps(t['input_schema'])}"
        )
    return (
        "\n\nTOOLS\nYou can use these tools:\n"
        + "\n".join(specs)
        + "\n\nTo call a tool, reply with ONLY one JSON object on a single "
        'line, exactly: {"name": "<tool name>", "arguments": {...}}\n'
        "One tool call per reply. When you have what you need, reply with "
        "your final answer as plain text instead (never JSON)."
    )


@dataclass
class ClaudeCliLLM:
    cli_path: str = "claude"
    timeout: float = 600.0
    usage: Usage = field(default_factory=Usage)
    cost_usd: float = 0.0

    def _run(self, *, model: str, system: str, prompt: str) -> str:
        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
        cmd = [
            self.cli_path,
            "-p",
            "--output-format",
            "json",
            "--model",
            model,
            "--system-prompt",
            system,
            "--tools",
            "",
        ]
        try:
            proc = subprocess.run(
                cmd,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=env,
            )
        except FileNotFoundError as exc:
            raise ClaudeCliError(
                f"claude CLI not found at {self.cli_path!r}; is Claude Code installed?"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise ClaudeCliError(f"claude CLI timed out after {self.timeout}s") from exc
        if proc.returncode != 0:
            raise ClaudeCliError(
                f"claude CLI exited {proc.returncode}: {proc.stderr.strip()[:300]}"
            )
        try:
            data = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise ClaudeCliError(
                f"unparseable CLI output: {proc.stdout[:200]!r}"
            ) from exc
        if data.get("is_error"):
            raise ClaudeCliError(f"claude CLI error: {str(data.get('result'))[:300]}")
        u = data.get("usage") or {}
        self.usage.calls += 1
        self.usage.input_tokens += u.get("input_tokens") or 0
        self.usage.output_tokens += u.get("output_tokens") or 0
        self.usage.cache_read_input_tokens += u.get("cache_read_input_tokens") or 0
        self.usage.cache_creation_input_tokens += u.get("cache_creation_input_tokens") or 0
        self.cost_usd += data.get("total_cost_usd") or 0.0
        return data.get("result") or ""

    def message(
        self,
        *,
        model: str,
        system: str | list[dict],
        messages: list[dict],
        max_tokens: int,  # advisory only; the CLI has no output cap flag
        tools: list[dict] | None = None,
        schema: dict | None = None,
        effort: str | None = None,  # ignored
    ) -> SimpleNamespace:
        sys_text = _system_text(system)
        if tools:
            sys_text += tool_protocol(tools)
        if schema:
            sys_text += (
                "\n\nRespond with ONLY valid JSON matching this schema "
                f"(no prose, no code fences):\n{json.dumps(schema)}"
            )
        prompt = render_transcript(messages) + "\n\n[ASSISTANT]\n"
        text = self._run(model=model, system=sys_text, prompt=prompt)

        blocks: list = []
        if tools:
            calls = extract_textual_tool_calls(text) or _bare_json_call(text)
            if calls:
                for i, call in enumerate(calls[:1]):  # one call per turn
                    fn = call["function"]
                    blocks.append(
                        SimpleNamespace(
                            type="tool_use",
                            id=f"cli_{self.usage.calls}_{i}",
                            name=fn.get("name", ""),
                            input=fn.get("arguments") or {},
                        )
                    )
                return SimpleNamespace(content=blocks, stop_reason="tool_use")
        if text.strip():
            blocks.append(SimpleNamespace(type="text", text=text.strip()))
        return SimpleNamespace(content=blocks, stop_reason="end_turn")

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
        sys_text = (
            _system_text(system)
            + "\n\nRespond with ONLY valid JSON matching this schema "
            + f"(no prose, no code fences):\n{json.dumps(schema)}"
        )
        text = self._run(model=model, system=sys_text, prompt=user)
        for attempt in range(2):
            try:
                return json.loads(strip_json_fences(text))
            except json.JSONDecodeError as exc:
                if attempt == 1:
                    raise ClaudeCliError(
                        f"invalid JSON from CLI after retry: {text[:200]!r}"
                    ) from exc
                text = self._run(
                    model=model,
                    system=sys_text,
                    prompt=(
                        f"{user}\n\nYour previous reply was not valid JSON "
                        f"({exc}). Reply again with ONLY the JSON object."
                    ),
                )
        raise AssertionError("unreachable")


def _bare_json_call(text: str) -> list[dict]:
    try:
        obj = json.loads(strip_json_fences(text))
    except json.JSONDecodeError:
        return []
    if isinstance(obj, dict) and "name" in obj and "arguments" in obj:
        return [{"function": {"name": obj["name"], "arguments": obj["arguments"]}}]
    return []
