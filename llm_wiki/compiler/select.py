"""SelectPages: pick up to k existing pages relevant to a new passage."""

from __future__ import annotations

from ..config import WikiConfig
from ..llm import LLM, cached_system
from ..search.index import WikiSearchIndex
from ..wiki.store import WikiStore
from .prompts import SELECT_PAGES_SCHEMA, SELECT_PAGES_SYSTEM


def select_pages(
    llm: LLM,
    config: WikiConfig,
    store: WikiStore,
    index: WikiSearchIndex,
    passage: str,
) -> list[str]:
    """Return up to ``config.select_pages_k`` existing page names relevant
    to the passage. Skips the LLM call entirely when the wiki is empty or
    lexical search surfaces no candidates."""
    if not store.all_names():
        return []
    candidates = index.search(passage, limit=20)
    if not candidates:
        return []

    lines = []
    for r in candidates:
        meta = []
        if r.aliases:
            meta.append("aliases: " + ", ".join(r.aliases))
        if r.tags:
            meta.append("tags: " + ", ".join(r.tags))
        suffix = f" ({'; '.join(meta)})" if meta else ""
        lines.append(f"- {r.name} — {r.summary}{suffix}")

    user = (
        f"NEW PASSAGE:\n{passage}\n\n"
        f"CANDIDATE EXISTING PAGES (from search):\n" + "\n".join(lines) + "\n\n"
        f"Select up to {config.select_pages_k} of these pages that this passage "
        f"is relevant to. Return their exact logical names."
    )
    result = llm.structured(
        model=config.compiler_model,
        system=cached_system(SELECT_PAGES_SYSTEM),
        user=user,
        schema=SELECT_PAGES_SCHEMA,
        max_tokens=1024,
        effort="low",
    )
    selected = []
    for name in result.get("pages", []):
        resolved = store.resolve(name)
        if resolved and resolved not in selected:
            selected.append(resolved)
    return selected[: config.select_pages_k]
