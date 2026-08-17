import pytest

from llm_wiki.config import WikiConfig
from llm_wiki.wiki.page import WikiPage
from llm_wiki.wiki.store import WikiStore, normalize_name


@pytest.fixture
def store(tmp_path):
    config = WikiConfig(root=tmp_path / "wiki")
    s = WikiStore(config)
    s.init()
    return s


def make_page(name, title, **kw):
    page = WikiPage(name=name, title=title, **kw)
    page.set_section("Key Facts", "- a fact [src:s1]")
    page.set_section("Related Pages", "")
    page.set_section("Related Sources", "- [src:s1]")
    return page


def test_save_load_roundtrip(store):
    page = make_page("people/Ada-Lovelace", "Ada Lovelace", type="people", aliases=["Ada"])
    store.save(page)
    loaded = store.load("people/Ada-Lovelace")
    assert loaded.title == "Ada Lovelace"
    assert loaded.aliases == ["Ada"]
    assert store.all_names() == ["people/Ada-Lovelace"]


def test_index_maintenance(store):
    store.save(make_page("people/Ada-Lovelace", "Ada Lovelace", type="people"))
    store.save(make_page("people/Alan-Turing", "Alan Turing", type="people"))
    entries = store.index_entries("people")
    assert entries == ["people/Ada-Lovelace", "people/Alan-Turing"]
    root = store.root_index_path.read_text()
    assert "people/ — 2 pages" in root


def test_resolve_by_alias_and_title(store):
    store.save(make_page("people/Ada-Lovelace", "Ada Lovelace", aliases=["Countess of Lovelace"]))
    assert store.resolve("people/Ada-Lovelace") == "people/Ada-Lovelace"
    assert store.resolve("ada lovelace") == "people/Ada-Lovelace"
    assert store.resolve("countess of lovelace") == "people/Ada-Lovelace"
    assert store.resolve("nobody") is None


def test_backlinks(store):
    a = make_page("a/One", "One")
    a.set_section("Related Pages", "- [[b/Two]]")
    store.save(a)
    store.save(make_page("b/Two", "Two"))
    assert store.backlinks("b/Two") == ["a/One"]


def test_sources(store):
    store.save_source("doc-1", "full text", digest="short digest")
    assert store.source_exists("doc-1")
    assert store.load_source("doc-1") == "short digest"  # digest preferred
    assert not store.source_exists("doc-2")


def test_normalize_name():
    assert normalize_name("pages/people/X.md") == "people/X"
    assert normalize_name("/people//X/") == "people/X"
