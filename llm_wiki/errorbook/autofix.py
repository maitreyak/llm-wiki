"""Layer 1 repair: deterministic code fixes, run after every article.

Covers the bulk of the paper's observed error mass (dangling links 55.8%,
malformed refs 22.3%, index inconsistency 8.2%, incomplete pages 2.1%):

- dangling links: retarget via alias resolution when the entity exists
  under another name; otherwise unwrap the link to plain text.
- malformed source refs: normalize recognizable variants to [src:id],
  drop empty ones.
- incomplete pages: add missing (empty) required sections; resync the
  Related Sources section with the refs actually used in the page.
- index inconsistency: rebuild all directory indexes from the filesystem.
"""

from __future__ import annotations

import re

from ..wiki.page import WIKILINK_RE, WikiPage, extract_source_refs
from ..wiki.store import WikiStore, normalize_name

_FIXABLE_REF_RE = re.compile(r"\[(?:source|Source|SRC|src|ref)\s*[:=]\s*([^\]\s]+)\s*\]")
_EMPTY_REF_RE = re.compile(r"\[src:\s*\]|\[(?:source|Source|SRC|ref)\s*[:=]\s*\]")


def autofix(store: WikiStore) -> list[str]:
    """Apply all deterministic fixes; returns human-readable fix log."""
    log: list[str] = []
    names = set(store.all_names())
    alias_map = store.alias_map()

    for name in sorted(names):
        page = store.load(name)
        original = page.to_markdown()
        _fix_links(page, names, alias_map, log)
        _fix_refs(page, log)
        _fix_sections(page, log)
        if page.to_markdown() != original:
            store.save(page, update_index=False)

    store.rebuild_all_indexes()
    return log


def _fix_links(page: WikiPage, names: set[str], alias_map: dict, log: list[str]) -> None:
    def repl(m: re.Match) -> str:
        target = normalize_name(m.group(1))
        display = m.group(2)
        if target in names:
            return m.group(0)
        resolved = alias_map.get(target.strip().lower()) or alias_map.get(
            target.rsplit("/", 1)[-1].replace("-", " ").strip().lower()
        )
        if resolved and resolved in names:
            log.append(f"{page.name}: retargeted [[{target}]] -> [[{resolved}]]")
            return f"[[{resolved}|{display}]]" if display else f"[[{resolved}]]"
        text = display or target.rsplit("/", 1)[-1].replace("-", " ")
        log.append(f"{page.name}: unwrapped dangling link [[{target}]] -> {text!r}")
        return text

    page.summary = WIKILINK_RE.sub(repl, page.summary)
    new_sections = []
    for heading, body in page.sections:
        if heading.lower() == "related pages":
            # A related-pages bullet whose link got unwrapped is just noise; drop it.
            kept = []
            for line in body.splitlines():
                fixed = WIKILINK_RE.sub(repl, line)
                if WIKILINK_RE.search(fixed) or not re.match(r"\s*[-*]\s+", line):
                    kept.append(fixed)
                else:
                    log.append(f"{page.name}: dropped dead Related Pages entry {line.strip()!r}")
            body = "\n".join(kept)
        else:
            body = WIKILINK_RE.sub(repl, body)
        new_sections.append((heading, body))
    page.sections = new_sections


def _fix_refs(page: WikiPage, log: list[str]) -> None:
    def apply(text: str) -> str:
        before = text
        text = _EMPTY_REF_RE.sub("", text)
        text = _FIXABLE_REF_RE.sub(lambda m: f"[src:{m.group(1)}]", text)
        if text != before:
            log.append(f"{page.name}: normalized source refs")
        return text

    page.summary = apply(page.summary)
    page.sections = [(h, apply(b)) for h, b in page.sections]


def _fix_sections(page: WikiPage, log: list[str]) -> None:
    for missing in page.missing_sections():
        page.set_section(missing, "")
        log.append(f"{page.name}: added missing section '{missing}'")
    refs = sorted(set(extract_source_refs(page.section("Key Facts") + page.summary)))
    if refs:
        current = page.related_sources
        if set(current) != set(refs):
            page.set_section("Related Sources", "\n".join(f"- [src:{r}]" for r in refs))
            log.append(f"{page.name}: resynced Related Sources")
