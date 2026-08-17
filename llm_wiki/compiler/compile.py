"""CompileWikiPages: turn one passage into Wiki page create/update operations.

The LLM emits structured page content (facts, links, metadata); this module
owns everything structural — logical names are derived in code, Related
Sources sections are computed from the actual [src:...] references, and
updates are applied as full-page merges the model returned after seeing the
current page content.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field as dataclass_field
from datetime import date

from ..config import WikiConfig
from ..llm import LLM, cached_system
from ..wiki.page import WikiPage, extract_source_refs, slugify
from ..wiki.store import WikiStore, normalize_name
from .prompts import CANONICAL_TYPES, COMPILE_SCHEMA, compile_system


@dataclass
class CompileResult:
    digest: str
    written: list[str] = dataclass_field(default_factory=list)  # page names written
    selected: list[str] = dataclass_field(default_factory=list)  # SelectPages output


def compile_passage(
    llm: LLM,
    config: WikiConfig,
    store: WikiStore,
    *,
    source_id: str,
    passage: str,
    selected: list[str],
    constraints: list[str],
) -> CompileResult:
    selected_blocks = []
    for name in selected:
        page_md = store.load(name).to_markdown()
        selected_blocks.append(f"--- existing page: {name} ---\n{page_md}")
    existing_context = (
        "\n\n".join(selected_blocks) if selected_blocks else "(no relevant existing pages)"
    )

    user = (
        f"SOURCE_ID: {source_id}\n\n"
        f"PASSAGE:\n{passage}\n\n"
        f"RELEVANT EXISTING PAGES:\n{existing_context}"
    )
    result = llm.structured(
        model=config.compiler_model,
        system=cached_system(compile_system(constraints)),
        user=user,
        schema=COMPILE_SCHEMA,
        max_tokens=config.max_output_tokens,
    )

    # Save source + digest first so [src:...] refs are resolvable immediately.
    digest = result.get("digest", "").strip()
    store.save_source(source_id, passage, digest=digest or None)

    # Derive logical names in code: updates target existing pages, creates
    # get type/slugified-title (resolving aliases so we never fork an entity).
    ops = []
    for op in result.get("pages", []):
        page_type = _clean_type(op.get("type", "topics"))
        title = _clean_title(op.get("title", ""), op.get("type", ""), page_type)
        op["title"] = title
        existing = op.get("existing_name")
        name = None
        if existing and store.exists(normalize_name(existing)):
            candidate = normalize_name(existing)
            old = store.load(candidate)
            # Guard against entity conflation: an "update" whose title shares no
            # tokens with the target page is a misdirected create, not an update.
            known = [old.title, *old.aliases, candidate.rsplit("/", 1)[-1].replace("-", " ")]
            if _titles_overlap(title, known):
                name = candidate
        if name is None:
            resolved = store.resolve(title)
            name = resolved or f"{page_type}/{slugify(title or 'untitled')}"
        ops.append((name, page_type, op))

    written: list[str] = []
    for name, page_type, op in ops:
        page = _build_page(store, name, page_type, op, source_id)
        store.save(page, update_index=False)
        written.append(name)

    for directory in {n.rsplit("/", 1)[0] for n in written if "/" in n}:
        store.rebuild_dir_index(directory)
    store.rebuild_root_index()
    return CompileResult(digest=digest, written=written, selected=list(selected))


def _build_page(
    store: WikiStore, name: str, page_type: str, op: dict, source_id: str
) -> WikiPage:
    is_update = store.exists(name)
    old = store.load(name) if is_update else None
    today = date.today().isoformat()

    page = WikiPage(
        name=name,
        title=op.get("title") or (old.title if old else name.rsplit("/", 1)[-1]),
        type=page_type if not old else old.type,
        created=old.created if old else today,
        updated=today,
        aliases=_merge_lists(old.aliases if old else [], op.get("aliases", [])),
        tags=_merge_lists(old.tags if old else [], op.get("tags", [])),
        summary=(op.get("summary") or (old.summary if old else "")).strip(),
    )

    facts = [f.strip() for f in op.get("key_facts", []) if f.strip()]
    if old:
        # The model was instructed to return the merged fact set; guard against
        # dropped facts by re-adding any old fact that disappeared entirely.
        facts = _merge_facts(old.key_facts, facts)
    page.set_section("Key Facts", "\n".join(f"- {f}" for f in facts))

    related = {}
    if old:
        for line in old.section("Related Pages").splitlines():
            m = re.match(r"\s*[-*]\s+\[\[([^\]|]+)\]\](?:\s*—\s*(.*))?", line)
            if m:
                related[normalize_name(m.group(1))] = (m.group(2) or "").strip()
    for entry in op.get("related_pages", []):
        link = normalize_name(entry.get("name", ""))
        if link and link != name:
            related[link] = entry.get("note", "").strip() or related.get(link, "")
    page.set_section(
        "Related Pages",
        "\n".join(
            f"- [[{link}]]" + (f" — {note}" if note else "")
            for link, note in sorted(related.items())
        ),
    )

    sources = sorted(set(extract_source_refs("\n".join(facts))) | {source_id} | set(old.related_sources if old else []))
    page.set_section("Related Sources", "\n".join(f"- [src:{s}]" for s in sources))
    return page


def _clean_type(raw: str) -> str:
    """Clamp the model-provided type onto the canonical set (small models
    emit things like 'media/film' or 'work')."""
    t = slugify(raw or "topics").lower()
    for c in CANONICAL_TYPES:
        if t == c or t.rstrip("s") == c.rstrip("s") or c in t.split("-"):
            return c
    return "topics"


def _clean_title(raw: str, raw_type: str, page_type: str) -> str:
    """Strip path-like noise from a model-provided title ('media/film Titanic'
    -> 'Titanic'); titles must be entity names, never logical paths."""
    title = (raw or "").strip()
    if "/" in title:
        title = title.split("/")[-1].strip()
        if "-" in title and " " not in title and slugify(title) == title:
            title = title.replace("-", " ")
    type_tokens = set(slugify(raw_type).lower().split("-")) | {
        page_type,
        page_type.rstrip("s"),
    }
    words = title.split()
    while len(words) > 1 and words[0].lower().strip("-") in type_tokens:
        words = words[1:]
    return " ".join(words).strip() or "Untitled"


def _titles_overlap(title: str, known: list[str]) -> bool:
    tokens = set(re.findall(r"[a-z0-9]+", title.lower()))
    for candidate in known:
        if tokens & set(re.findall(r"[a-z0-9]+", candidate.lower())):
            return True
    return False


def _merge_lists(old: list[str], new: list[str]) -> list[str]:
    out = list(old)
    seen = {x.strip().lower() for x in old}
    for x in new:
        x = x.strip()
        if x and x.lower() not in seen:
            out.append(x)
            seen.add(x.lower())
    return out


def _fact_fingerprint(fact: str) -> str:
    text = re.sub(r"\[src:[^\]]+\]", "", fact)
    text = re.sub(r"\[\[([^\]|]+?)(?:\|[^\]]+?)?\]\]", r"\1", text)
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _merge_facts(old_facts: list[str], new_facts: list[str]) -> list[str]:
    fingerprints = {_fact_fingerprint(f) for f in new_facts}
    merged = list(new_facts)
    for f in old_facts:
        if _fact_fingerprint(f) not in fingerprints:
            merged.append(f)
    return merged


__all__ = ["compile_passage", "CompileResult", "CANONICAL_TYPES"]
