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
import time
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
    skipped: list[str] = field(default_factory=list)  # source_ids that failed

    def __str__(self) -> str:
        skipped = f" ({len(self.skipped)} passages skipped)" if self.skipped else ""
        return (
            f"{self.articles} articles / {self.passages} passages -> "
            f"{len(self.pages_written)} pages touched{skipped}"
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
    resume: bool = False,
) -> IngestReport:
    index = WikiSearchIndex(store)
    report = IngestReport()
    say = progress or (lambda _msg: None)
    consecutive_failures = 0
    max_consecutive_failures = 3
    retry_queue: list[tuple[str, str]] = []

    for doc_num, doc in enumerate(documents, start=1):
        audits: list[PassageAudit] = []
        for source_id, passage in split_passages(doc):
            if resume and store.source_exists(source_id):
                # Already compiled by an interrupted earlier run; its pages
                # are on disk. (A kill between source-save and page-writes is
                # a sub-second window; validators catch any inconsistency.)
                report.passages += 1
                continue
            say(f"[{doc_num}/{len(documents)}] compiling {source_id}")
            constraints = constraints_fn() if constraints_fn else []
            try:
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
            except Exception as exc:  # noqa: BLE001 — one passage must not kill a build
                say(f"SKIPPED {source_id}: {type(exc).__name__}: {exc}")
                report.skipped.append(source_id)
                retry_queue.append((source_id, passage))
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    # A streak is either truly systemic (billing, auth, API
                    # down — a tiny probe also fails: abort) or throughput
                    # shedding of sustained traffic (probe succeeds: cool
                    # down and push on; the retry queue recovers the skips).
                    if _probe(llm, config):
                        say(
                            f"{consecutive_failures} consecutive failures but "
                            f"probe OK — throughput shedding; cooling down "
                            f"{config.shed_cooldown_seconds}s"
                        )
                        time.sleep(config.shed_cooldown_seconds)
                        consecutive_failures = 0
                        continue
                    raise RuntimeError(
                        f"aborting ingest: {consecutive_failures} consecutive "
                        f"passage failures and probe failed — last: {exc}"
                    ) from exc
                continue
            consecutive_failures = 0
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
        # During resume fast-forward nothing was compiled: audits is empty and
        # running validators/autofix per replayed article is pure overhead.
        if hooks and audits:
            _run_hook(hooks.after_article, say, "after-article repair", audits)
            if doc_num % config.repair_every_n_articles == 0:
                say(f"periodic LLM repair after article {doc_num}")
                _run_hook(hooks.periodic_fix, say, "periodic repair")
            index.rebuild()

    # Second chance for passages skipped on transient failures (API flaps):
    # one retry each, at the end, when the outage has likely passed.
    if retry_queue:
        say(f"retrying {len(retry_queue)} skipped passage(s)")
        for source_id, passage in retry_queue:
            try:
                constraints = constraints_fn() if constraints_fn else []
                selected = select_pages(llm, config, store, index, passage)
                result = compile_passage(
                    llm,
                    config,
                    store,
                    source_id=source_id,
                    passage=passage,
                    selected=selected,
                    constraints=constraints,
                )
                report.skipped.remove(source_id)
                report.passages += 1
                report.pages_written.update(result.written)
                index.rebuild()
                say(f"recovered {source_id}")
            except Exception as exc:  # noqa: BLE001
                say(f"still failing {source_id}: {type(exc).__name__}: {exc}")

    if hooks:
        say("finalization repair rounds")
        _run_hook(hooks.finalize, say, "finalization repair")
    store.rebuild_all_indexes()
    return report


def _probe(llm, config) -> bool:
    """Minimal single-attempt call to tell 'API is down' from 'API is
    shedding our sustained traffic'."""
    saved = getattr(llm, "max_attempts", None)
    try:
        if saved is not None:
            llm.max_attempts = 1
        llm.message(
            model=config.compiler_model,
            system="Reply with OK.",
            messages=[{"role": "user", "content": "ok"}],
            max_tokens=4,
        )
        return True
    except Exception:  # noqa: BLE001
        return False
    finally:
        if saved is not None:
            llm.max_attempts = saved


def _run_hook(fn, say, label: str, *args) -> None:
    """Repair passes are best-effort: a failure (e.g. API overload during an
    LLM check) is logged and deferred, never fatal to the build — validators
    re-run at the next checkpoint and the error book persists findings."""
    try:
        fn(*args)
    except Exception as exc:  # noqa: BLE001
        say(f"HOOK FAILED ({label}): {type(exc).__name__}: {exc}")


def load_documents(paths: list[Path]) -> list[Document]:
    docs = []
    for path in paths:
        text = path.read_text()
        docs.append(Document(doc_id=path.stem, text=text, title=path.stem.replace("_", " ")))
    return docs
