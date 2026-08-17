"""WikiStore: the on-disk Wiki.

Layout under a wiki root:

    wiki.json                  # config
    index.md                   # root index: one line per directory
    pages/
      people/_index.md         # directory index: one line per page
      people/Lasse-Hallstrom.md
      media/...
    sources/
      digests/<source-id>.md   # paragraph digest kept for grounding checks
      articles/<source-id>.md  # original passage text
    error_book.yaml

Pages are addressed by logical name (``people/Lasse-Hallstrom``). Directory
indexes are regenerated from page files, so the filesystem is the source of
truth; a divergence between the two is an "index inconsistency" error the
Error Book validators catch (we still expose rebuild for autofix).
"""

from __future__ import annotations

import re
from pathlib import Path

from ..config import WikiConfig
from .page import WikiPage, extract_wikilinks


class WikiStore:
    def __init__(self, config: WikiConfig):
        self.config = config

    # --- setup ---------------------------------------------------------------

    def init(self) -> None:
        self.config.pages_dir.mkdir(parents=True, exist_ok=True)
        self.config.digests_dir.mkdir(parents=True, exist_ok=True)
        self.config.articles_dir.mkdir(parents=True, exist_ok=True)
        self.config.save()
        if not self.root_index_path.exists():
            self.rebuild_root_index()

    # --- paths ---------------------------------------------------------------

    @property
    def root_index_path(self) -> Path:
        return self.config.root / "index.md"

    def page_path(self, name: str) -> Path:
        name = normalize_name(name)
        return self.config.pages_dir / f"{name}.md"

    def dir_index_path(self, directory: str) -> Path:
        return self.config.pages_dir / directory / "_index.md"

    # --- page CRUD -----------------------------------------------------------

    def exists(self, name: str) -> bool:
        return self.page_path(name).exists()

    def load(self, name: str) -> WikiPage:
        name = normalize_name(name)
        return WikiPage.from_markdown(name, self.page_path(name).read_text())

    def save(self, page: WikiPage, update_index: bool = True) -> None:
        page.name = normalize_name(page.name)
        path = self.page_path(page.name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(page.to_markdown())
        if update_index:
            directory = page.name.rsplit("/", 1)[0] if "/" in page.name else ""
            self.rebuild_dir_index(directory)
            self.rebuild_root_index()

    def delete(self, name: str) -> None:
        path = self.page_path(name)
        if path.exists():
            path.unlink()

    def all_names(self) -> list[str]:
        names = []
        for path in sorted(self.config.pages_dir.rglob("*.md")):
            if path.name == "_index.md":
                continue
            rel = path.relative_to(self.config.pages_dir)
            names.append(str(rel.with_suffix("")))
        return names

    def all_pages(self) -> list[WikiPage]:
        return [self.load(n) for n in self.all_names()]

    def directories(self) -> list[str]:
        dirs = {n.rsplit("/", 1)[0] for n in self.all_names() if "/" in n}
        return sorted(dirs)

    # --- name resolution -----------------------------------------------------

    def alias_map(self) -> dict[str, str]:
        """lowercased title/alias/basename -> logical name."""
        out: dict[str, str] = {}
        for page in self.all_pages():
            keys = [page.title, page.name.rsplit("/", 1)[-1].replace("-", " ")]
            keys.extend(page.aliases)
            for k in keys:
                out.setdefault(k.strip().lower(), page.name)
        return out

    def resolve(self, ref: str) -> str | None:
        """Resolve a page reference (logical name, title, or alias) to a name."""
        ref = normalize_name(ref)
        if self.exists(ref):
            return ref
        return self.alias_map().get(ref.strip().lower())

    # --- indexes -------------------------------------------------------------

    def rebuild_dir_index(self, directory: str) -> None:
        dir_path = self.config.pages_dir / directory if directory else self.config.pages_dir
        if not dir_path.exists():
            return
        entries = []
        for path in sorted(dir_path.glob("*.md")):
            if path.name == "_index.md":
                continue
            rel = path.relative_to(self.config.pages_dir)
            name = str(rel.with_suffix(""))
            page = WikiPage.from_markdown(name, path.read_text())
            meta = []
            if page.aliases:
                meta.append("aliases: " + ", ".join(page.aliases))
            if page.tags:
                meta.append("tags: " + ", ".join(page.tags))
            suffix = f" ({'; '.join(meta)})" if meta else ""
            summary = page.summary.splitlines()[0] if page.summary else ""
            entries.append(f"- [[{name}]] — {summary}{suffix}")
        header = f"# Index: {directory or 'pages'}\n"
        index_path = dir_path / "_index.md"
        if entries:
            index_path.write_text(header + "\n" + "\n".join(entries) + "\n")
        elif index_path.exists():
            index_path.unlink()

    def rebuild_all_indexes(self) -> None:
        for directory in self.directories():
            self.rebuild_dir_index(directory)
        self.rebuild_dir_index("")
        self.rebuild_root_index()

    def rebuild_root_index(self) -> None:
        lines = ["# Wiki Index", ""]
        dirs = self.directories()
        counts = {d: 0 for d in dirs}
        rootless = 0
        for name in self.all_names():
            if "/" in name:
                counts[name.rsplit("/", 1)[0]] += 1
            else:
                rootless += 1
        for d in dirs:
            lines.append(f"- {d}/ — {counts[d]} pages (read `{d}/_index` for the list)")
        if rootless:
            lines.append(f"- ./ — {rootless} pages (read `_index` for the list)")
        if not dirs and not rootless:
            lines.append("(empty wiki)")
        self.root_index_path.write_text("\n".join(lines) + "\n")

    def index_entries(self, directory: str) -> list[str]:
        """Page names listed in a directory's _index.md (for validators)."""
        path = self.dir_index_path(directory) if directory else self.config.pages_dir / "_index.md"
        if not path.exists():
            return []
        return extract_wikilinks(path.read_text())

    # --- sources -------------------------------------------------------------

    def save_source(self, source_id: str, text: str, digest: str | None = None) -> None:
        safe = _safe_source_id(source_id)
        (self.config.articles_dir / f"{safe}.md").write_text(text)
        if digest is not None:
            (self.config.digests_dir / f"{safe}.md").write_text(digest)

    def load_source(self, source_id: str) -> str | None:
        safe = _safe_source_id(source_id)
        for base in (self.config.digests_dir, self.config.articles_dir):
            path = base / f"{safe}.md"
            if path.exists():
                return path.read_text()
        return None

    def source_exists(self, source_id: str) -> bool:
        safe = _safe_source_id(source_id)
        return (self.config.articles_dir / f"{safe}.md").exists() or (
            self.config.digests_dir / f"{safe}.md"
        ).exists()

    # --- link graph ----------------------------------------------------------

    def backlinks(self, name: str) -> list[str]:
        name = normalize_name(name)
        return [p.name for p in self.all_pages() if name in p.outgoing_links()]


def normalize_name(name: str) -> str:
    name = name.strip().strip("/")
    if name.endswith(".md"):
        name = name[:-3]
    if name.startswith("pages/"):
        name = name[len("pages/"):]
    return re.sub(r"/{2,}", "/", name)


def _safe_source_id(source_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.\-]", "-", source_id)
