import pytest

from llm_wiki.agent.tools import WikiTools
from llm_wiki.config import WikiConfig
from llm_wiki.search.index import WikiSearchIndex
from llm_wiki.wiki.page import WikiPage
from llm_wiki.wiki.store import WikiStore


@pytest.fixture
def store(tmp_path):
    config = WikiConfig(root=tmp_path / "wiki")
    s = WikiStore(config)
    s.init()

    director = WikiPage(
        name="people/Lasse-Hallstrom",
        title="Lasse Hallström",
        type="people",
        aliases=["Lasse Hallstrom"],
        tags=["director", "sweden"],
        summary="Swedish film director.",
    )
    director.set_section("Key Facts", "- Directed [[media/Whats-Eating-Gilbert-Grape]]. [src:s1]")
    director.set_section("Related Pages", "- [[media/Whats-Eating-Gilbert-Grape]]")
    director.set_section("Related Sources", "- [src:s1]")
    s.save(director)

    film = WikiPage(
        name="media/Whats-Eating-Gilbert-Grape",
        title="What's Eating Gilbert Grape",
        type="media",
        tags=["film", "1993"],
        summary="1993 American drama film.",
    )
    film.set_section("Key Facts", "- Directed by [[people/Lasse-Hallstrom]]. [src:s1]")
    film.set_section("Related Pages", "- [[people/Lasse-Hallstrom]]")
    film.set_section("Related Sources", "- [src:s1]")
    s.save(film)
    return s


def test_search_ranks_title_match_first(store):
    index = WikiSearchIndex(store)
    results = index.search("gilbert grape")
    assert results
    assert results[0].name == "media/Whats-Eating-Gilbert-Grape"


def test_exact_alias_match_boost(store):
    index = WikiSearchIndex(store)
    results = index.search("Lasse Hallstrom")
    assert results[0].name == "people/Lasse-Hallstrom"


def test_content_match_finds_page(store):
    index = WikiSearchIndex(store)
    # "directed" only appears in body content
    names = [r.name for r in index.search("directed")]
    assert "people/Lasse-Hallstrom" in names


def test_empty_query_and_no_results(store):
    index = WikiSearchIndex(store)
    assert index.search("") == []
    assert index.search("zzzqqq nonexistent") == []


def test_wiki_read_page_and_indexes(store):
    tools = WikiTools(store)
    out = tools.wiki_read(["index"])
    assert "people/" in out and "media/" in out
    out = tools.wiki_read(["people/_index"])
    assert "[[people/Lasse-Hallstrom]]" in out
    out = tools.wiki_read(["people/Lasse-Hallstrom", "media/Whats-Eating-Gilbert-Grape"])
    assert "# Lasse Hallström" in out
    assert "# What's Eating Gilbert Grape" in out


def test_wiki_read_resolves_alias_and_reports_missing(store):
    tools = WikiTools(store)
    out = tools.wiki_read(["Lasse Hallstrom"])
    assert "# Lasse Hallström" in out
    out = tools.wiki_read(["people/Nobody-Here"])
    assert "ERROR: page not found" in out
    assert "Did you mean" in out


def test_wiki_search_tool_output(store):
    tools = WikiTools(store)
    out = tools.wiki_search("gilbert grape")
    assert out.startswith("- [[media/Whats-Eating-Gilbert-Grape]]")
    assert tools.wiki_search("zzzqqq") == "No results."
