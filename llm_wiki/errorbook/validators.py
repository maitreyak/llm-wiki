"""Deterministic structural validators (the Error Book's discovery layer
for structural-validity categories)."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..wiki.page import MALFORMED_SOURCE_REF_RE, SOURCE_REF_RE
from ..wiki.store import WikiStore, normalize_name
from .models import Finding


@dataclass
class PassageAudit:
    """What one passage's compilation did (for the unseen-overwrite check)."""

    source_id: str
    selected: list[str] = field(default_factory=list)
    written: list[str] = field(default_factory=list)
    preexisting: list[str] = field(default_factory=list)  # written pages that already existed


def validate_structure(store: WikiStore) -> list[Finding]:
    findings: list[Finding] = []
    names = set(store.all_names())
    alias_map = store.alias_map()
    pages = store.all_pages()

    for page in pages:
        # dangling links
        for link in set(page.outgoing_links()):
            target = normalize_name(link)
            if target not in names and target.strip().lower() not in alias_map:
                findings.append(
                    Finding("dangling_link", page.name, f"link to missing page [[{link}]]")
                )
        # malformed refs
        text = page.all_text()
        for m in set(m.group(0) for m in MALFORMED_SOURCE_REF_RE.finditer(text)):
            findings.append(Finding("malformed_ref", page.name, f"malformed source ref {m!r}"))
        for ref in set(SOURCE_REF_RE.findall(text)):
            if not store.source_exists(ref):
                findings.append(
                    Finding("malformed_ref", page.name, f"ref to unknown source [src:{ref}]")
                )
        for i, fact in enumerate(page.key_facts):
            if not SOURCE_REF_RE.search(fact):
                findings.append(
                    Finding("malformed_ref", page.name, f"fact {i + 1} has no [src:] reference")
                )
        # incomplete pages
        missing = page.missing_sections()
        if missing:
            findings.append(
                Finding("incomplete_page", page.name, f"missing sections: {', '.join(missing)}")
            )
        if not page.summary:
            findings.append(Finding("incomplete_page", page.name, "missing summary"))
        if "Key Facts" not in missing and not page.key_facts:
            findings.append(Finding("incomplete_page", page.name, "empty Key Facts"))

    # index inconsistency: bidirectional diff per directory
    directories = store.directories() + [""]
    for directory in directories:
        in_dir = {
            n for n in names if (n.rsplit("/", 1)[0] if "/" in n else "") == directory
        }
        indexed = {normalize_name(e) for e in store.index_entries(directory)}
        if not in_dir and not indexed:
            continue
        for missing_entry in sorted(in_dir - indexed):
            findings.append(
                Finding(
                    "index_inconsistency",
                    missing_entry,
                    f"page not listed in {directory or '.'}/_index",
                )
            )
        for stale in sorted(indexed - in_dir):
            findings.append(
                Finding(
                    "index_inconsistency",
                    stale,
                    f"stale index entry in {directory or '.'}/_index",
                )
            )
    return findings


def check_unseen_overwrites(audits: list[PassageAudit]) -> list[Finding]:
    findings = []
    for audit in audits:
        selected = set(audit.selected)
        for name in audit.preexisting:
            if name not in selected:
                findings.append(
                    Finding(
                        "unseen_overwrite",
                        name,
                        f"modified while compiling {audit.source_id} without being selected",
                    )
                )
    return findings
