"""ErrorBookManager: wires validators, autofix, LLM checks, and the error
book into the ingest pipeline's repair hooks.

Two-layer repair per the paper:
- Layer 1 (after every article): deterministic validators -> code autofix
  -> re-validate -> record + verify/close in the error book.
- Layer 2 (every N articles + finalization): LLM source-grounded fact
  check on recently touched pages, removing unsupported facts; sampled
  cross-page contradiction check (recorded; constraints injected).
Finalization runs `finalization_rounds` alternations of both layers.
"""

from __future__ import annotations

from ..config import WikiConfig
from ..llm import LLM
from ..wiki.store import WikiStore
from .autofix import autofix
from .book import ErrorBook
from .llm_checks import (
    check_contradictions,
    check_unsupported_facts,
    remove_unsupported_facts,
)
from .validators import PassageAudit, check_unseen_overwrites, validate_structure


class ErrorBookManager:
    def __init__(self, llm: LLM, config: WikiConfig, store: WikiStore):
        self.llm = llm
        self.config = config
        self.store = store
        self.book = ErrorBook(config.error_book_path)
        self.fix_log: list[str] = []
        self._touched_since_llm_fix: set[str] = set()

    # --- pipeline hooks ------------------------------------------------------

    def after_article(self, audits: list[PassageAudit]) -> None:
        for audit in audits:
            self._touched_since_llm_fix.update(audit.written)
        findings = validate_structure(self.store) + check_unseen_overwrites(audits)
        self.book.record_findings(findings)
        self.fix_log.extend(autofix(self.store))
        remaining = validate_structure(self.store)
        self.book.verify_and_close(remaining)
        self.book.save()

    def periodic_fix(self) -> None:
        pages = sorted(self._touched_since_llm_fix & set(self.store.all_names()))
        if pages:
            findings, to_remove = check_unsupported_facts(
                self.llm, self.config, self.store, pages
            )
            self.book.record_findings(findings)
            self.fix_log.extend(remove_unsupported_facts(self.store, to_remove))
        self._touched_since_llm_fix.clear()
        self.book.save()

    def finalize(self) -> None:
        for _ in range(self.config.finalization_rounds):
            findings = validate_structure(self.store)
            if not findings:
                break
            self.book.record_findings(findings)
            self.fix_log.extend(autofix(self.store))
            self.book.verify_and_close(validate_structure(self.store))
        self.periodic_fix()
        contradiction_findings = check_contradictions(self.llm, self.config, self.store)
        self.book.record_findings(contradiction_findings)
        self.book.save()

    # --- constraint injection ------------------------------------------------

    def active_constraints(self) -> list[str]:
        return self.book.active_constraints()
