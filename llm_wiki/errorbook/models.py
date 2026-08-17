"""Error Book data model.

Seven error categories from the paper — structural validity:
dangling_link, malformed_ref, index_inconsistency, incomplete_page,
unseen_overwrite; content consistency: unsupported_fact,
cross_page_contradiction.

Each distinct error becomes an ErrorRecord with a lifecycle
(discover -> attribute -> constrain -> inject -> verify & close). The
constraint text is what gets injected into future compilation prompts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

CATEGORIES = (
    "dangling_link",
    "malformed_ref",
    "index_inconsistency",
    "incomplete_page",
    "unseen_overwrite",
    "unsupported_fact",
    "cross_page_contradiction",
)

# Deterministic categories are re-checkable by validators on every pass,
# so their records can be auto-closed when the error no longer reproduces.
DETERMINISTIC_CATEGORIES = frozenset(
    ["dangling_link", "malformed_ref", "index_inconsistency", "incomplete_page"]
)

# Canned constraint rules per category, in the spirit of the paper's
# natural-language rules. (The paper attributes errors with an LLM; we use
# fixed per-category rules — same injection mechanism, zero extra calls.)
CONSTRAINT_RULES = {
    "dangling_link": (
        "NEVER write a [[wikilink]] to a page that does not already exist and is "
        "not being created in this same response. If unsure whether a page "
        "exists, mention the entity as plain text instead of linking it."
    ),
    "malformed_ref": (
        "Source references must use exactly the form [src:SOURCE_ID] with the "
        "given source id — no spaces, no other keywords like 'source' or 'ref'. "
        "Every key fact must end with one."
    ),
    "index_inconsistency": (
        "Do not write index or listing content inside pages; directory indexes "
        "are maintained automatically."
    ),
    "incomplete_page": (
        "Every page must include a one-line summary and non-empty Key Facts; "
        "do not emit placeholder or stub pages without factual content."
    ),
    "unseen_overwrite": (
        "Only update pages that were provided to you as RELEVANT EXISTING "
        "PAGES; for any other entity, either create a new page or leave it "
        "untouched."
    ),
    "unsupported_fact": (
        "Do not add any attribute or claim about an entity unless it is stated "
        "in the given passage or already present on the page with its original "
        "source reference. Never generalize, infer, or embellish."
    ),
    "cross_page_contradiction": (
        "When updating a page, keep its facts consistent with what the passage "
        "says; if the passage contradicts an existing fact, keep the version "
        "supported by the cited sources rather than duplicating both claims."
    ),
}


@dataclass
class Finding:
    """One concrete error occurrence discovered by a validator or LLM check."""

    category: str
    page: str
    detail: str  # human-readable phenomenon, also used for dedup

    @property
    def key(self) -> str:
        return f"{self.category}|{self.page}|{self.detail}"


@dataclass
class ErrorRecord:
    id: str
    category: str
    page: str
    phenomenon: str
    constraint: str
    status: str = "open"  # open | closed
    occurrences: int = 1
    first_seen: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    last_seen: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    @property
    def key(self) -> str:
        return f"{self.category}|{self.page}|{self.phenomenon}"
