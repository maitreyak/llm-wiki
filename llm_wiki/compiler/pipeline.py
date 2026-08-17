"""The ingest pipeline: documents -> passages -> Wiki, with repair hooks.

Per the paper's Algorithm 1: passages compile sequentially against the
current Wiki state (SelectPages -> CompileWikiPages -> apply). After each
article: structural validation + code autofix + Error Book update. Every
N=10 articles: LLM periodic fix for semantic errors. At the end of an
ingest run: a finalization loop alternating code fixes and LLM fixes.

The repair hooks are provided by ``errorbook.manager.ErrorBookManager``;
the pipeline runs without them (hooks=None) for compile-only use.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

from ..config import WikiConfig
from ..errorbook.validators import PassageAudit
from ..llm import LLM
from ..search.index import WikiSearchIndex
from ..wiki.store import WikiStore
from .compile import compile_passage
from .select import select_pages


class RepairHooks(Protocol):
    def after_article(self, audits: list[PassageAudit]) -> None:
        """Run validators + code autofix after one article's passages."""

    def periodic_fix(self) -> None:
        """LLM-based repair pass (every N articles)."""

    def finalize(self) -> None:
        """Closing repair rounds after an ingest run."""


@dataclass
class Document:
    doc_id: str
    text: str
    title: str | None = None


@dataclass
class IngestReport:
    articles: int = 0
    passages: int = 0
    pages_written: set[str] = field(default_factory=set)

    def __str__(self) -> str:
        return (
            f"{self.articles} articles / {self.passages} passages -> "
            f"{len(self.pages_written)} pages touched"
        )


def split_passages(doc: Document, max_chars: int = 4000) -> list[tuple[str, str]]:
    """Split a document into (source_id, passage) pairs on paragraph
    boundaries, packing paragraphs up to ~max_chars per passage."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", doc.text) if p.strip()]
    passages: list[str] = []
    buf: list[str] = []
    size = 0
    for para in paragraphs:
        if buf and size + len(para) > max_chars:
            passages.append("\n\n".join(buf))
            buf, size = [], 0
        buf.append(para)
        size += len(para)
    if buf:
        passages.append("\n\n".join(buf))
    if len(passages) == 1:
        return [(doc.doc_id, _with_title(doc, passages[0]))]
    return [
        (f"{doc.doc_id}-p{i:03d}", _with_title(doc, p))
        for i, p in enumerate(passages, start=1)
    ]


def _with_title(doc: Document, passage: str) -> str:
    if doc.title and doc.title not in passage.splitlines()[0]:
        return f"[Document: {doc.title}]\n{passage}"
    return passage


def ingest(
    llm: LLM,
    config: WikiConfig,
    store: WikiStore,
    documents: list[Document],
    *,
    hooks: RepairHooks | None = None,
    constraints_fn: Callable[[], list[str]] | None = None,
    progress: Callable[[str], None] | None = None,
) -> IngestReport:
    index = WikiSearchIndex(store)
    report = IngestReport()
    say = progress or (lambda _msg: None)

    for doc_num, doc in enumerate(documents, start=1):
        audits: list[PassageAudit] = []
        for source_id, passage in split_passages(doc):
            say(f"[{doc_num}/{len(documents)}] compiling {source_id}")
            constraints = constraints_fn() if constraints_fn else []
            selected = select_pages(llm, config, store, index, passage)
            existing_before = set(store.all_names())
            result = compile_passage(
                llm,
                config,
                store,
                source_id=source_id,
                passage=passage,
                selected=selected,
                constraints=constraints,
            )
            audits.append(
                PassageAudit(
                    source_id=source_id,
                    selected=result.selected,
                    written=result.written,
                    preexisting=[w for w in result.written if w in existing_before],
                )
            )
            report.passages += 1
            report.pages_written.update(result.written)
            index.rebuild()
        report.articles += 1
        if hooks:
            hooks.after_article(audits)
            if doc_num % config.repair_every_n_articles == 0:
                say(f"periodic LLM repair after article {doc_num}")
                hooks.periodic_fix()
            index.rebuild()

    if hooks:
        say("finalization repair rounds")
        hooks.finalize()
    store.rebuild_all_indexes()
    return report


def load_documents(paths: list[Path]) -> list[Document]:
    docs = []
    for path in paths:
        text = path.read_text()
        docs.append(Document(doc_id=path.stem, text=text, title=path.stem.replace("_", " ")))
    return docs
