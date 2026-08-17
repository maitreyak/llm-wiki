from llm_wiki.wiki.page import (
    WikiPage,
    extract_source_refs,
    extract_wikilinks,
    slugify,
)

SAMPLE = """\
---
type: people
created: 2026-08-17
updated: 2026-08-17
aliases:
- Lasse Hallstrom
tags:
- director
- sweden
---

# Lasse Hallström

> Swedish film director best known for character-driven dramas.

## Key Facts

- Born 2 June 1946 in Stockholm, Sweden. [src:doc-0012]
- Directed [[media/Whats-Eating-Gilbert-Grape]] (1993). [src:doc-0012]

## Related Pages

- [[media/Whats-Eating-Gilbert-Grape]] — directed the film

## Related Sources

- [src:doc-0012]
"""


def test_parse_fields():
    page = WikiPage.from_markdown("people/Lasse-Hallstrom", SAMPLE)
    assert page.title == "Lasse Hallström"
    assert page.type == "people"
    assert page.aliases == ["Lasse Hallstrom"]
    assert page.tags == ["director", "sweden"]
    assert page.summary.startswith("Swedish film director")
    assert len(page.key_facts) == 2
    assert page.related_pages == ["media/Whats-Eating-Gilbert-Grape"]
    assert page.related_sources == ["doc-0012"]
    assert page.missing_sections() == []


def test_roundtrip():
    page = WikiPage.from_markdown("people/Lasse-Hallstrom", SAMPLE)
    again = WikiPage.from_markdown(page.name, page.to_markdown())
    assert again == page


def test_links_and_refs():
    text = "See [[a/B|B page]] and [[c/D]]. [src:s1] [src:s2]"
    assert extract_wikilinks(text) == ["a/B", "c/D"]
    assert extract_source_refs(text) == ["s1", "s2"]


def test_missing_sections():
    page = WikiPage(name="x/Y", title="Y")
    assert page.missing_sections() == ["Key Facts", "Related Pages", "Related Sources"]
    page.set_section("Key Facts", "- fact")
    assert "Key Facts" not in page.missing_sections()


def test_slugify():
    assert slugify("What's Eating Gilbert Grape") == "Whats-Eating-Gilbert-Grape"
    assert slugify("  A  B  ") == "A-B"
    assert slugify("!!!") == "untitled"
