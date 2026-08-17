"""WikiPage: parse/serialize the paper's Wiki page format.

A page is a Markdown file with YAML frontmatter:

    ---
    type: people
    created: 2026-08-17
    updated: 2026-08-17
    aliases: [Lasse Hallstrom]
    tags: [director, sweden]
    ---

    # Lasse Hallström

    > Swedish film director best known for ...

    ## Key Facts

    - Born 2 June 1946 in Stockholm, Sweden. [src:doc-0012]
    - Directed [[media/Whats-Eating-Gilbert-Grape]] (1993). [src:doc-0012]

    ## Related Pages

    - [[media/Whats-Eating-Gilbert-Grape]] — directed the film

    ## Related Sources

    - [src:doc-0012]

Pages are identified by their logical name: the path relative to the pages
root without the ``.md`` extension, e.g. ``people/Lasse-Hallstrom``.
Wikilinks use that logical name: ``[[people/Lasse-Hallstrom]]`` or
``[[people/Lasse-Hallstrom|display text]]``. Source references use
``[src:<source-id>]``.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date

import yaml

WIKILINK_RE = re.compile(r"\[\[([^\]|]+?)(?:\|([^\]]+?))?\]\]")
SOURCE_REF_RE = re.compile(r"\[src:([A-Za-z0-9_.\-/]+)\]")
# Things that look like a broken attempt at a source ref (used by validators).
MALFORMED_SOURCE_REF_RE = re.compile(
    r"\[(?:source|Source|SRC|ref)\s*[:=]\s*[^\]]*\]|\[src:\s*\]|\[src\s+[^\]]*\]"
)

REQUIRED_SECTIONS = ("Key Facts", "Related Pages", "Related Sources")


def slugify(name: str) -> str:
    """Turn an entity name into a filesystem/wikilink-safe ASCII basename.

    Diacritics are folded (Hallström -> Hallstrom) so links written with or
    without accents land on the same page."""
    slug = unicodedata.normalize("NFKD", name.strip())
    slug = "".join(c for c in slug if not unicodedata.combining(c))
    slug = re.sub(r"['’\"“”]", "", slug)
    slug = re.sub(r"[^A-Za-z0-9]+", "-", slug)
    slug = re.sub(r"-{2,}", "-", slug).strip("-")
    return slug or "untitled"


def extract_wikilinks(text: str) -> list[str]:
    """Return logical page names referenced by ``[[...]]`` links in text."""
    return [m.group(1).strip() for m in WIKILINK_RE.finditer(text)]


def extract_source_refs(text: str) -> list[str]:
    return [m.group(1) for m in SOURCE_REF_RE.finditer(text)]


@dataclass
class WikiPage:
    name: str  # logical name, e.g. "people/Lasse-Hallstrom"
    title: str
    type: str = "topics"
    created: str = ""
    updated: str = ""
    aliases: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    summary: str = ""
    # Ordered (heading, body) pairs; body is markdown without the heading line.
    sections: list[tuple[str, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        today = date.today().isoformat()
        self.created = self.created or today
        self.updated = self.updated or today

    # --- section helpers ----------------------------------------------------

    def section(self, heading: str) -> str:
        for h, body in self.sections:
            if h.lower() == heading.lower():
                return body
        return ""

    def set_section(self, heading: str, body: str) -> None:
        body = body.strip("\n")
        for i, (h, _) in enumerate(self.sections):
            if h.lower() == heading.lower():
                self.sections[i] = (h, body)
                return
        self.sections.append((heading, body))

    @property
    def key_facts(self) -> list[str]:
        return _bullet_items(self.section("Key Facts"))

    @property
    def related_pages(self) -> list[str]:
        """Logical names linked from the Related Pages section."""
        return extract_wikilinks(self.section("Related Pages"))

    @property
    def related_sources(self) -> list[str]:
        return extract_source_refs(self.section("Related Sources"))

    def all_text(self) -> str:
        """Full body text (summary + all sections) for link/ref scans and search."""
        parts = [self.summary]
        parts.extend(body for _, body in self.sections)
        return "\n".join(parts)

    def outgoing_links(self) -> list[str]:
        return extract_wikilinks(self.all_text())

    def source_refs(self) -> list[str]:
        return extract_source_refs(self.all_text())

    def missing_sections(self) -> list[str]:
        present = {h.lower() for h, _ in self.sections}
        return [s for s in REQUIRED_SECTIONS if s.lower() not in present]

    # --- serialization ------------------------------------------------------

    def to_markdown(self) -> str:
        fm = {
            "type": self.type,
            "created": self.created,
            "updated": self.updated,
            "aliases": self.aliases,
            "tags": self.tags,
        }
        out = ["---", yaml.safe_dump(fm, sort_keys=False, allow_unicode=True).strip(), "---", ""]
        out.append(f"# {self.title}")
        out.append("")
        if self.summary:
            for line in self.summary.splitlines():
                out.append(f"> {line}" if line.strip() else ">")
            out.append("")
        for heading, body in self.sections:
            out.append(f"## {heading}")
            out.append("")
            if body:
                out.append(body)
                out.append("")
        return "\n".join(out).rstrip("\n") + "\n"

    @classmethod
    def from_markdown(cls, name: str, text: str) -> "WikiPage":
        fm, body = _split_frontmatter(text)
        title = ""
        summary_lines: list[str] = []
        sections: list[tuple[str, str]] = []
        current: str | None = None
        buf: list[str] = []

        def flush() -> None:
            nonlocal buf, current
            if current is not None:
                sections.append((current, "\n".join(buf).strip("\n")))
            buf = []

        for line in body.splitlines():
            h2 = re.match(r"##\s+(.*)", line)
            h1 = re.match(r"#\s+(?!#)(.*)", line)
            if h2:
                flush()
                current = h2.group(1).strip()
            elif h1 and not title:
                title = h1.group(1).strip()
            elif current is None:
                if line.startswith(">"):
                    summary_lines.append(line.lstrip(">").strip())
            else:
                buf.append(line)
        flush()

        aliases = fm.get("aliases") or []
        tags = fm.get("tags") or []
        return cls(
            name=name,
            title=title or name.rsplit("/", 1)[-1].replace("-", " "),
            type=str(fm.get("type", "topics")),
            created=str(fm.get("created", "")),
            updated=str(fm.get("updated", "")),
            aliases=[str(a) for a in aliases] if isinstance(aliases, list) else [str(aliases)],
            tags=[str(t) for t in tags] if isinstance(tags, list) else [str(tags)],
            summary="\n".join(summary_lines).strip(),
            sections=sections,
        )


def _bullet_items(body: str) -> list[str]:
    items: list[str] = []
    for line in body.splitlines():
        m = re.match(r"\s*[-*]\s+(.*)", line)
        if m:
            items.append(m.group(1).strip())
        elif items and line.strip() and re.match(r"\s{2,}", line):
            items[-1] += " " + line.strip()
    return items


def _split_frontmatter(text: str) -> tuple[dict, str]:
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            try:
                fm = yaml.safe_load(parts[1]) or {}
                if isinstance(fm, dict):
                    return fm, parts[2]
            except yaml.YAMLError:
                pass
    return {}, text
