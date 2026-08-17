"""error_book.yaml persistence and the error lifecycle.

discover -> (attribute/constrain via per-category rules) -> inject
(active_constraints feeds compilation prompts) -> verify & close
(deterministic categories auto-close when validators stop reproducing them).
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .models import CONSTRAINT_RULES, DETERMINISTIC_CATEGORIES, ErrorRecord, Finding


class ErrorBook:
    def __init__(self, path: Path):
        self.path = path
        self.records: list[ErrorRecord] = []
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        data = yaml.safe_load(self.path.read_text()) or {}
        self.records = [ErrorRecord(**r) for r in data.get("errors", [])]

    def save(self) -> None:
        data = {"errors": [asdict(r) for r in self.records]}
        self.path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))

    # --- lifecycle -----------------------------------------------------------

    def record_findings(self, findings: list[Finding]) -> list[ErrorRecord]:
        """Discover: merge findings into the book (dedup on category|page|detail)."""
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        by_key = {r.key: r for r in self.records}
        new_records = []
        for f in findings:
            existing = by_key.get(f.key)
            if existing:
                existing.occurrences += 1
                existing.last_seen = now
                existing.status = "open"
            else:
                record = ErrorRecord(
                    id=f"E{len(self.records) + len(new_records) + 1:04d}",
                    category=f.category,
                    page=f.page,
                    phenomenon=f.detail,
                    constraint=CONSTRAINT_RULES.get(f.category, ""),
                )
                new_records.append(record)
                by_key[record.key] = record
        self.records.extend(new_records)
        return new_records

    def verify_and_close(self, current_findings: list[Finding]) -> int:
        """Close deterministic-category records that no longer reproduce."""
        live = {f.key for f in current_findings}
        closed = 0
        for r in self.records:
            if (
                r.status == "open"
                and r.category in DETERMINISTIC_CATEGORIES
                and r.key not in live
            ):
                r.status = "closed"
                closed += 1
        return closed

    # --- inject --------------------------------------------------------------

    def active_constraints(self) -> list[str]:
        """One constraint rule per category that has ever occurred, ordered by
        total occurrence count. Rules stay active after errors close — the
        point is to stop the compiler repeating the mistake."""
        counts: dict[str, int] = {}
        for r in self.records:
            counts[r.category] = counts.get(r.category, 0) + r.occurrences
        ordered = sorted(counts, key=counts.get, reverse=True)
        return [CONSTRAINT_RULES[c] for c in ordered if c in CONSTRAINT_RULES]

    # --- reporting -----------------------------------------------------------

    def open_records(self) -> list[ErrorRecord]:
        return [r for r in self.records if r.status == "open"]

    def summary(self) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        for r in self.records:
            bucket = out.setdefault(r.category, {"open": 0, "closed": 0, "occurrences": 0})
            bucket[r.status] += 1
            bucket["occurrences"] += r.occurrences
        return out
