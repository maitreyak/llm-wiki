from types import SimpleNamespace

from llm_wiki.config import WikiConfig
from llm_wiki.llm import make_llm
from llm_wiki.ollama_llm import (
    OllamaLLM,
    anthropic_messages_to_ollama,
    anthropic_tools_to_ollama,
    ollama_response_to_blocks,
)


def test_tools_conversion():
    tools = [
        {
            "name": "wiki_search",
            "description": "d",
            "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}},
            "strict": True,
        }
    ]
    out = anthropic_tools_to_ollama(tools)
    assert out[0]["type"] == "function"
    assert out[0]["function"]["name"] == "wiki_search"
    assert out[0]["function"]["parameters"]["properties"]["query"]["type"] == "string"


def test_message_translation_roundtrip():
    assistant_blocks = [
        SimpleNamespace(type="text", text="Searching."),
        SimpleNamespace(type="tool_use", id="t1", name="wiki_search", input={"query": "ada"}),
    ]
    messages = [
        {"role": "user", "content": "Who is Ada?"},
        {"role": "assistant", "content": assistant_blocks},
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "- [[people/Ada]]"}
            ],
        },
    ]
    out = anthropic_messages_to_ollama("SYS", messages)
    assert out[0] == {"role": "system", "content": "SYS"}
    assert out[1] == {"role": "user", "content": "Who is Ada?"}
    assert out[2]["role"] == "assistant"
    assert out[2]["tool_calls"][0]["function"]["name"] == "wiki_search"
    assert out[3] == {"role": "tool", "tool_name": "wiki_search", "content": "- [[people/Ada]]"}


def test_response_to_blocks_with_tool_calls():
    message = {
        "content": "Let me look.",
        "tool_calls": [
            {"function": {"name": "wiki_read", "arguments": {"paths": ["people/Ada"]}}}
        ],
    }
    blocks, stop = ollama_response_to_blocks(message)
    assert stop == "tool_use"
    assert blocks[0].type == "text"
    assert blocks[1].type == "tool_use"
    assert blocks[1].input == {"paths": ["people/Ada"]}

    blocks, stop = ollama_response_to_blocks({"content": "ANSWER: x"})
    assert stop == "end_turn"
    assert blocks[0].text == "ANSWER: x"

    # string-encoded arguments are parsed
    blocks, _ = ollama_response_to_blocks(
        {"content": "", "tool_calls": [{"function": {"name": "f", "arguments": '{"a": 1}'}}]}
    )
    assert blocks[0].input == {"a": 1}


def test_make_llm_factory(tmp_path):
    config = WikiConfig(root=tmp_path, provider="ollama")
    llm = make_llm(config)
    assert isinstance(llm, OllamaLLM)
    config.provider = "nope"
    try:
        make_llm(config)
        assert False, "expected ValueError"
    except ValueError:
        pass
