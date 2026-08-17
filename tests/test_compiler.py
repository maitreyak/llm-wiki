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


def test_split_passages_packs_paragraphs():
    doc = Document(doc_id="d", text="a\n\nb\n\nc", title="T")
    parts = split_passages(doc, max_chars=1)
    assert [sid for sid, _ in parts] == ["d-p001", "d-p002", "d-p003"]
    assert all(p.startswith("[Document: T]") for _, p in parts)

    small = Document(doc_id="d", text="only one paragraph")
    assert split_passages(small)[0][0] == "d"
