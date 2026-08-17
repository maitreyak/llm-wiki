import pytest

from llm_wiki.config import WikiConfig
from llm_wiki.errorbook.autofix import autofix
from llm_wiki.errorbook.book import ErrorBook
from llm_wiki.errorbook.llm_checks import remove_unsupported_facts
from llm_wiki.errorbook.models import CONSTRAINT_RULES, Finding
from llm_wiki.errorbook.validators import (
    PassageAudit,
    check_unseen_overwrites,
    validate_structure,
)
from llm_wiki.wiki.page import WikiPage
from llm_wiki.wiki.store import WikiStore


@pytest.fixture
def store(tmp_path):
    config = WikiConfig(root=tmp_path / "wiki")
    s = WikiStore(config)
    s.init()
    return s


def save_page(store, name, title, facts, related="", aliases=(), summary="A summary."):
    page = WikiPage(name=name, title=title, aliases=list(aliases), summary=summary)
    page.set_section("Key Facts", "\n".join(f"- {f}" for f in facts))
    page.set_section("Related Pages", related)
    page.set_section("Related Sources", "")
    store.save(page)
    return page


def categories(findings):
    return {f.category for f in findings}


def test_validators_catch_dangling_link_and_bad_refs(store):
    store.save_source("s1", "text")
    save_page(
        store,
        "a/One",
        "One",
        [
            "Links to [[b/Missing]]. [src:s1]",
            "Bad ref form. [source: s1]",
            "Unknown source. [src:nope]",
            "No source at all.",
        ],
    )
    findings = validate_structure(store)
    cats = categories(findings)
    assert "dangling_link" in cats
    assert "malformed_ref" in cats
    details = " | ".join(f.detail for f in findings)
    assert "[[b/Missing]]" in details
    assert "unknown source" in details
    assert "no [src:] reference" in details


def test_validators_catch_incomplete_and_index_drift(store):
    page = WikiPage(name="a/Stub", title="Stub")  # no sections, no summary
    store.save(page)
    # create index drift: write a page file without updating indexes
    rogue = WikiPage(name="a/Rogue", title="Rogue", summary="s")
    rogue.set_section("Key Facts", "- f [src:s1]")
    rogue.set_section("Related Pages", "")
    rogue.set_section("Related Sources", "- [src:s1]")
    store.save(rogue, update_index=False)

    findings = validate_structure(store)
    cats = categories(findings)
    assert "incomplete_page" in cats
    assert "index_inconsistency" in cats


def test_unseen_overwrite(store):
    audits = [
        PassageAudit(
            source_id="s1",
            selected=["a/One"],
            written=["a/One", "a/Two", "a/New"],
            preexisting=["a/One", "a/Two"],
        )
    ]
    findings = check_unseen_overwrites(audits)
    assert [f.page for f in findings] == ["a/Two"]


def test_autofix_repairs_structure(store):
    store.save_source("s1", "text")
    save_page(store, "people/Ada-Lovelace", "Ada Lovelace", ["A fact. [src:s1]"], aliases=["Ada"])
    save_page(
        store,
        "a/One",
        "One",
        [
            "Links to [[people/Ada]] by alias. [src:s1]",
            "Dangling [[b/Missing|the missing one]]. [src:s1]",
            "Bad ref. [source: s1]",
        ],
        related="- [[b/Missing]] — gone",
    )
    stub = WikiPage(name="a/Stub", title="Stub", summary="s")
    store.save(stub, update_index=False)  # missing sections + index drift

    log = autofix(store)
    page = store.load("a/One")
    text = page.all_text()
    assert "[[people/Ada-Lovelace|Ada]]" in text or "[[people/Ada-Lovelace]]" in text
    assert "[[b/Missing" not in text
    assert "the missing one" in text  # unwrapped to display text
    assert "[source: s1]" not in text and "[src:s1]" in text
    assert store.load("a/Stub").missing_sections() == []
    assert "a/Stub" in store.index_entries("a")
    assert log

    remaining = validate_structure(store)
    assert "dangling_link" not in categories(remaining)
    assert "index_inconsistency" not in categories(remaining)


def test_error_book_lifecycle(tmp_path):
    book = ErrorBook(tmp_path / "error_book.yaml")
    findings = [Finding("dangling_link", "a/One", "link to missing page [[b/X]]")]
    created = book.record_findings(findings)
    assert len(created) == 1
    # repeat -> dedup + occurrence bump
    book.record_findings(findings)
    assert book.records[0].occurrences == 2

    assert CONSTRAINT_RULES["dangling_link"] in book.active_constraints()

    closed = book.verify_and_close([])
    assert closed == 1
    assert book.records[0].status == "closed"
    # constraints stay active after close
    assert book.active_constraints()

    book.save()
    reloaded = ErrorBook(tmp_path / "error_book.yaml")
    assert reloaded.records[0].id == book.records[0].id
    assert reloaded.records[0].status == "closed"


def test_remove_unsupported_facts(store):
    save_page(store, "a/One", "One", ["Keep me. [src:s1]", "Drop me. [src:s1]", "Keep too. [src:s1]"])
    log = remove_unsupported_facts(store, {"a/One": [2]})
    assert log
    facts = store.load("a/One").key_facts
    assert facts == ["Keep me. [src:s1]", "Keep too. [src:s1]"]
