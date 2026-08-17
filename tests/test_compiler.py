import pytest

from llm_wiki.compiler.compile import compile_passage
from llm_wiki.compiler.pipeline import Document, split_passages
from llm_wiki.config import WikiConfig
from llm_wiki.wiki.store import WikiStore


class FakeLLM:
    """Stands in for llm.LLM: returns queued structured responses."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def structured(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


@pytest.fixture
def store(tmp_path):
    config = WikiConfig(root=tmp_path / "wiki")
    s = WikiStore(config)
    s.init()
    return s


def page_op(title, ptype, facts, existing_name=None, related=(), aliases=(), tags=(), summary="s"):
    return {
        "title": title,
        "type": ptype,
        "existing_name": existing_name,
        "aliases": list(aliases),
        "tags": list(tags),
        "summary": summary,
        "key_facts": facts,
        "related_pages": [{"name": n, "note": ""} for n in related],
    }


def test_compile_creates_pages_and_sources(store):
    llm = FakeLLM(
        [
            {
                "digest": "A film and its director.",
                "pages": [
                    page_op(
                        "What's Eating Gilbert Grape",
                        "media",
                        ["Directed by [[people/Lasse-Hallstrom]]. [src:doc-1]"],
                        related=["people/Lasse-Hallstrom"],
                    ),
                    page_op(
                        "Lasse Hallström",
                        "people",
                        ["Directed [[media/Whats-Eating-Gilbert-Grape]]. [src:doc-1]"],
                        aliases=["Lasse Hallstrom"],
                    ),
                ],
            }
        ]
    )
    result = compile_passage(
        llm,
        store.config,
        store,
        source_id="doc-1",
        passage="text",
        selected=[],
        constraints=[],
    )
    assert sorted(result.written) == [
        "media/Whats-Eating-Gilbert-Grape",
        "people/Lasse-Hallstrom",
    ]
    page = store.load("people/Lasse-Hallstrom")
    assert page.related_sources == ["doc-1"]
    assert store.load_source("doc-1") == "A film and its director."
    assert store.index_entries("media") == ["media/Whats-Eating-Gilbert-Grape"]


def test_compile_update_merges_and_keeps_old_facts(store):
    llm1 = FakeLLM(
        [
            {
                "digest": "d1",
                "pages": [
                    page_op("Ada Lovelace", "people", ["Born in 1815. [src:a]"], tags=["math"])
                ],
            }
        ]
    )
    compile_passage(llm1, store.config, store, source_id="a", passage="p", selected=[], constraints=[])

    # Update via existing_name; model "forgets" the old fact — merge must keep it.
    llm2 = FakeLLM(
        [
            {
                "digest": "d2",
                "pages": [
                    page_op(
                        "Ada Lovelace",
                        "people",
                        ["Wrote the first algorithm. [src:b]"],
                        existing_name="people/Ada-Lovelace",
                        tags=["computing"],
                    )
                ],
            }
        ]
    )
    compile_passage(llm2, store.config, store, source_id="b", passage="p", selected=["people/Ada-Lovelace"], constraints=[])

    page = store.load("people/Ada-Lovelace")
    facts = "\n".join(page.key_facts)
    assert "Born in 1815" in facts and "first algorithm" in facts
    assert page.tags == ["math", "computing"]
    assert page.related_sources == ["a", "b"]
    assert page.created  # preserved


def test_compile_resolves_title_to_existing_page(store):
    llm1 = FakeLLM(
        [{"digest": "d", "pages": [page_op("Ada Lovelace", "people", ["f1. [src:a]"], aliases=["Ada"])]}]
    )
    compile_passage(llm1, store.config, store, source_id="a", passage="p", selected=[], constraints=[])
    # Second compile creates "Ada" without existing_name — alias resolution
    # must route it onto the same page instead of forking the entity.
    llm2 = FakeLLM([{"digest": "d", "pages": [page_op("Ada", "people", ["f2. [src:b]"])]}])
    compile_passage(llm2, store.config, store, source_id="b", passage="p", selected=[], constraints=[])
    assert store.all_names() == ["people/Ada-Lovelace"]
    assert len(store.load("people/Ada-Lovelace").key_facts) == 2


def test_type_and_title_cleaning(store):
    llm = FakeLLM(
        [
            {
                "digest": "d",
                "pages": [
                    page_op("media/film Titanic 1997", "media/film", ["f. [src:a]"]),
                    page_op("work/My-Life-as-a-Dog", "work", ["f. [src:a]"]),
                ],
            }
        ]
    )
    result = compile_passage(llm, store.config, store, source_id="a", passage="p", selected=[], constraints=[])
    assert "media/Titanic-1997" in result.written
    assert "works/My-Life-as-a-Dog" in result.written
    assert store.load("media/Titanic-1997").title == "Titanic 1997"
    assert store.load("works/My-Life-as-a-Dog").title == "My Life as a Dog"


def test_update_with_mismatched_existing_name_becomes_create(store):
    llm1 = FakeLLM(
        [{"digest": "d", "pages": [page_op("Lasse Hallstrom", "people", ["f1. [src:a]"])]}]
    )
    compile_passage(llm1, store.config, store, source_id="a", passage="p", selected=[], constraints=[])
    # Model wrongly claims James Cameron is an update of Lasse's page.
    llm2 = FakeLLM(
        [
            {
                "digest": "d",
                "pages": [
                    page_op(
                        "James Cameron",
                        "people",
                        ["Born 1954. [src:b]"],
                        existing_name="people/Lasse-Hallstrom",
                    )
                ],
            }
        ]
    )
    compile_passage(llm2, store.config, store, source_id="b", passage="p", selected=["people/Lasse-Hallstrom"], constraints=[])
    assert sorted(store.all_names()) == ["people/James-Cameron", "people/Lasse-Hallstrom"]
    assert "Born 1954" not in " ".join(store.load("people/Lasse-Hallstrom").key_facts)


def test_structured_retries_on_truncation():
    from types import SimpleNamespace

    from llm_wiki.llm import LLM, TruncatedOutputError

    calls = []

    class FakeMessagesLLM(LLM):
        def message(self, **kwargs):
            calls.append(kwargs["max_tokens"])
            if len(calls) == 1:
                return SimpleNamespace(stop_reason="max_tokens", content=[])
            return SimpleNamespace(
                stop_reason="end_turn",
                content=[SimpleNamespace(type="text", text='{"ok": true}')],
            )

    llm = FakeMessagesLLM()
    out = llm.structured(model="m", system="s", user="u", schema={}, max_tokens=100)
    assert out == {"ok": True}
    assert calls == [100, 200]

    calls.clear()

    class AlwaysTruncated(LLM):
        def message(self, **kwargs):
            return SimpleNamespace(stop_reason="max_tokens", content=[])

    try:
        AlwaysTruncated().structured(model="m", system="s", user="u", schema={}, max_tokens=100)
        assert False, "expected TruncatedOutputError"
    except TruncatedOutputError:
        pass


def test_pipeline_skips_failing_passage(store):
    from llm_wiki.compiler.pipeline import Document, ingest

    class ExplodingLLM:
        def structured(self, **kwargs):
            raise RuntimeError("boom")

    report = ingest(
        ExplodingLLM(),
        store.config,
        store,
        [Document(doc_id="d1", text="some text"), Document(doc_id="d2", text="more")],
    )
    assert report.skipped == ["d1", "d2"]
    assert report.articles == 2


def test_pipeline_aborts_on_consecutive_failures(store):
    import pytest as _pytest

    from llm_wiki.compiler.pipeline import Document, ingest

    class ExplodingLLM:
        def structured(self, **kwargs):
            raise RuntimeError("credit balance too low")

    docs = [Document(doc_id=f"d{i}", text="t") for i in range(5)]
    with _pytest.raises(RuntimeError, match="consecutive passage failures"):
        ingest(ExplodingLLM(), store.config, store, docs)


def test_split_passages_packs_paragraphs():
    doc = Document(doc_id="d", text="a\n\nb\n\nc", title="T")
    parts = split_passages(doc, max_chars=1)
    assert [sid for sid, _ in parts] == ["d-p001", "d-p002", "d-p003"]
    assert all(p.startswith("[Document: T]") for _, p in parts)

    small = Document(doc_id="d", text="only one paragraph")
    assert split_passages(small)[0][0] == "d"
