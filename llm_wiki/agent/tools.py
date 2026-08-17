"""The agent's two retrieval tools: wiki_search and wiki_read.

Per the paper, retrieval decomposes into atomic operations the agent
composes itself:

- ``wiki_search(query)`` ranks pages by structured signals first
  (names, aliases, tags) then content, returning candidates + metadata.
- ``wiki_read(paths)`` batch-reads directory indexes or full pages;
  returned pages keep their ``[[wikilinks]]`` intact as traversal
  affordances.
"""

from __future__ import annotations

from ..search.index import WikiSearchIndex
from ..wiki.store import WikiStore, normalize_name

WIKI_SEARCH_TOOL = {
    "name": "wiki_search",
    "description": (
        "Search the Wiki for pages matching a query. Ranking prioritizes page "
        "names, aliases, and tags over body content, so entity-style queries "
        "(names of people, works, places) work best. Returns ranked candidate "
        "pages with their path, one-line summary, aliases, and tags. Call this "
        "when you need to locate a page for an entity or topic; if a search "
        "returns nothing useful, try a shorter or reworded query."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Free-text search query"}
        },
        "required": ["query"],
        "additionalProperties": False,
    },
    "strict": True,
}

WIKI_READ_TOOL = {
    "name": "wiki_read",
    "description": (
        "Read one or more Wiki pages or directory indexes in a single call. "
        "Paths are logical page names like 'people/Lasse-Hallstrom'. Special "
        "paths: 'index' (root index of directories) and '<dir>/_index' (list "
        "of pages in a directory). Returned pages contain [[wikilinks]] to "
        "related pages — follow them with further wiki_read calls to traverse "
        "multi-hop evidence. Batch related reads into one call."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "paths": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Logical page names or index paths to read",
            }
        },
        "required": ["paths"],
        "additionalProperties": False,
    },
    "strict": True,
}


class WikiTools:
    def __init__(self, store: WikiStore, index: WikiSearchIndex | None = None):
        self.store = store
        self.index = index or WikiSearchIndex(store)

    def wiki_search(self, query: str, limit: int = 8) -> str:
        results = self.index.search(query, limit=limit)
        if not results:
            return "No results."
        lines = []
        for r in results:
            meta = []
            if r.aliases:
                meta.append("aliases: " + ", ".join(r.aliases))
            if r.tags:
                meta.append("tags: " + ", ".join(r.tags))
            suffix = f" ({'; '.join(meta)})" if meta else ""
            lines.append(f"- [[{r.name}]] — {r.summary}{suffix}")
        return "\n".join(lines)

    def wiki_read(self, paths: list[str]) -> str:
        chunks = []
        for raw in paths:
            chunks.append(self._read_one(raw))
        return "\n\n".join(chunks)

    def _read_one(self, raw: str) -> str:
        path = normalize_name(raw)
        if path in ("index", "index.md", ""):
            return _titled(raw, self.store.root_index_path.read_text())
        if path.endswith("_index"):
            directory = path[: -len("_index")].strip("/")
            index_path = (
                self.store.dir_index_path(directory)
                if directory
                else self.store.config.pages_dir / "_index.md"
            )
            if index_path.exists():
                return _titled(raw, index_path.read_text())
            return f"=== {raw} ===\nERROR: no such directory index."
        resolved = self.store.resolve(path)
        if resolved is None:
            hint = ""
            results = self.index.search(path.replace("/", " ").replace("-", " "), limit=3)
            if results:
                hint = " Did you mean: " + ", ".join(f"[[{r.name}]]" for r in results)
            return f"=== {raw} ===\nERROR: page not found.{hint}"
        page = self.store.load(resolved)
        return _titled(resolved, page.to_markdown())


def _titled(path: str, content: str) -> str:
    return f"=== {path} ===\n{content.strip()}"
