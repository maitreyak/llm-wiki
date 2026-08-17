from types import SimpleNamespace

from llm_wiki.claude_cli_llm import (
    ClaudeCliLLM,
    render_transcript,
    strip_json_fences,
    tool_protocol,
)
from llm_wiki.config import WikiConfig
from llm_wiki.llm import make_llm


def test_render_transcript():
    messages = [
        {"role": "user", "content": "Who is Ada?"},
        {
            "role": "assistant",
            "content": [
                SimpleNamespace(type="text", text="Searching."),
                SimpleNamespace(type="tool_use", id="t1", name="wiki_search", input={"query": "ada"}),
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "- [[people/Ada]]"}],
        },
    ]
    out = render_transcript(messages)
    assert "[USER]\nWho is Ada?" in out
    assert '[ASSISTANT TOOL CALL]\n{"name": "wiki_search"' in out
    assert "[TOOL RESULT: wiki_search]\n- [[people/Ada]]" in out


def test_strip_json_fences():
    assert strip_json_fences('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert strip_json_fences('{"a": 1}') == '{"a": 1}'


def test_tool_protocol_mentions_tools():
    text = tool_protocol(
        [{"name": "wiki_read", "description": "read pages", "input_schema": {"type": "object"}}]
    )
    assert "wiki_read" in text and "ONLY one JSON object" in text


def test_message_parses_tool_call_and_text(monkeypatch):
    llm = ClaudeCliLLM()
    outputs = ['{"name": "wiki_read", "arguments": {"paths": ["people/Ada"]}}', "ANSWER: 1815"]

    monkeypatch.setattr(llm, "_run", lambda **kw: outputs.pop(0))
    tools = [{"name": "wiki_read", "description": "", "input_schema": {"type": "object"}}]

    resp = llm.message(model="m", system="s", messages=[{"role": "user", "content": "q"}],
                       max_tokens=100, tools=tools)
    assert resp.stop_reason == "tool_use"
    assert resp.content[0].name == "wiki_read"
    assert resp.content[0].input == {"paths": ["people/Ada"]}

    resp = llm.message(model="m", system="s", messages=[{"role": "user", "content": "q"}],
                       max_tokens=100, tools=tools)
    assert resp.stop_reason == "end_turn"
    assert resp.content[0].text == "ANSWER: 1815"


def test_structured_retries_bad_json(monkeypatch):
    llm = ClaudeCliLLM()
    outputs = ["not json at all", '```json\n{"ok": true}\n```']
    prompts = []

    def fake_run(**kw):
        prompts.append(kw["prompt"])
        return outputs.pop(0)

    monkeypatch.setattr(llm, "_run", fake_run)
    out = llm.structured(model="m", system="s", user="u", schema={}, max_tokens=100)
    assert out == {"ok": True}
    assert len(prompts) == 2 and "not valid JSON" in prompts[1]


def test_factory_builds_cli_backend(tmp_path):
    config = WikiConfig(root=tmp_path, provider="claude-cli")
    assert isinstance(make_llm(config), ClaudeCliLLM)
