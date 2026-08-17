from types import SimpleNamespace

import pytest

from llm_wiki.agent.loop import ask
from llm_wiki.config import WikiConfig
from llm_wiki.wiki.page import WikiPage
from llm_wiki.wiki.store import WikiStore


def text_block(text):
    return SimpleNamespace(type="text", text=text)


def tool_block(id, name, input):
    return SimpleNamespace(type="tool_use", id=id, name=name, input=input)


def response(content, stop_reason):
    return SimpleNamespace(content=content, stop_reason=stop_reason)


class ScriptedLLM:
    """Replays a fixed sequence of agent responses; records requests."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def message(self, **kwargs):
        self.requests.append(kwargs)
        return self.responses.pop(0)


@pytest.fixture
def store(tmp_path):
    config = WikiConfig(root=tmp_path / "wiki")
    s = WikiStore(config)
    s.init()
    page = WikiPage(name="people/Ada-Lovelace", title="Ada Lovelace", summary="Mathematician.")
    page.set_section("Key Facts", "- Born in 1815. [src:s1]")
    page.set_section("Related Pages", "")
    page.set_section("Related Sources", "- [src:s1]")
    s.save(page)
    return s


def test_happy_path_search_read_answer(store):
    llm = ScriptedLLM(
        [
            response([tool_block("t1", "wiki_search", {"query": "Ada Lovelace"})], "tool_use"),
            response(
                [tool_block("t2", "wiki_read", {"paths": ["people/Ada-Lovelace"]})], "tool_use"
            ),
            response([text_block("ANSWER: 1815\nEVIDENCE: her page.")], "end_turn"),
        ]
    )
    result = ask(llm, store.config, store, "When was Ada born?")
    assert result.answer == "1815"
    assert result.trace.searches == 1
    assert result.trace.reads == 1
    assert result.trace.pages_read == ["people/Ada-Lovelace"]
    # tool results flowed back as user messages
    second_request = llm.requests[1]
    assert second_request["messages"][-1]["content"][0]["type"] == "tool_result"
    assert "Ada Lovelace" in second_request["messages"][-1]["content"][0]["content"]


def test_nudge_when_answering_without_reading(store):
    llm = ScriptedLLM(
        [
            response([text_block("ANSWER: guess")], "end_turn"),
            response(
                [tool_block("t1", "wiki_read", {"paths": ["people/Ada-Lovelace"]})], "tool_use"
            ),
            response([text_block("ANSWER: 1815")], "end_turn"),
        ]
    )
    result = ask(llm, store.config, store, "When was Ada born?")
    assert result.answer == "1815"
    assert result.trace.reads == 1
    # the nudge message was inserted
    assert any(
        "must read at least one" in str(m.get("content"))
        for req in llm.requests
        for m in req["messages"]
        if m["role"] == "user"
    )


def test_empty_response_stall_is_retried(store):
    llm = ScriptedLLM(
        [
            response([tool_block("t1", "wiki_search", {"query": "Ada"})], "tool_use"),
            response([], "end_turn"),  # model stalls with empty content
            response(
                [tool_block("t2", "wiki_read", {"paths": ["people/Ada-Lovelace"]})], "tool_use"
            ),
            response([text_block("ANSWER: 1815")], "end_turn"),
        ]
    )
    result = ask(llm, store.config, store, "q")
    assert result.answer == "1815"
    assert result.trace.reads == 1


def test_stall_gives_up_after_max_nudges(store):
    llm = ScriptedLLM(
        [
            response([], "end_turn"),
            response([], "end_turn"),
            response([], "end_turn"),
        ]
    )
    result = ask(llm, store.config, store, "q")
    assert result.answer == ""
    assert len(llm.requests) == 3  # 1 initial + 2 nudges, then give up


def test_budget_termination(store):
    config = store.config
    config.agent_max_tool_calls = 2
    llm = ScriptedLLM(
        [
            response([tool_block("t1", "wiki_read", {"paths": ["people/Ada-Lovelace"]})], "tool_use"),
            response([tool_block("t2", "wiki_search", {"query": "x"})], "tool_use"),
            response([text_block("ANSWER: 1815")], "end_turn"),  # forced final
        ]
    )
    result = ask(llm, config, store, "q")
    assert result.trace.stop_reason == "budget"
    assert result.answer == "1815"
    # forced-final request carries no tools
    assert "tools" not in llm.requests[-1] or not llm.requests[-1].get("tools")


def test_patience_prompts_redirect(store):
    config = store.config
    config.agent_patience = 2
    llm = ScriptedLLM(
        [
            response([tool_block("t1", "wiki_search", {"query": "zz1"})], "tool_use"),
            response([tool_block("t2", "wiki_search", {"query": "zz2"})], "tool_use"),
            response(
                [tool_block("t3", "wiki_read", {"paths": ["people/Ada-Lovelace"]})], "tool_use"
            ),
            response([text_block("ANSWER: 1815")], "end_turn"),
        ]
    )
    result = ask(llm, config, store, "q")
    assert result.answer == "1815"
    joined = " ".join(
        str(m.get("content"))
        for req in llm.requests
        for m in req["messages"]
        if m["role"] == "user"
    )
    assert "Repeated searches found nothing" in joined
