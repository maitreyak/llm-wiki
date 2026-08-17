"""Content-consistency checks that need a judge model (Layer 2 discovery):
unsupported facts (source-grounded verification) and cross-page
contradictions (sampled linked pairs)."""

from __future__ import annotations

import random

from ..config import WikiConfig
from ..llm import LLM, cached_system
from ..wiki.page import SOURCE_REF_RE
from ..wiki.store import WikiStore
from .models import Finding

FACT_CHECK_SYSTEM = """\
You verify that Wiki page facts are grounded in their cited sources. For each
numbered fact, decide whether the cited source text supports it. A fact is
unsupported if the source does not state it (paraphrase is fine; inference,
embellishment, or facts about entities the source never mentions are not).
Only judge against the provided source texts."""

FACT_CHECK_SCHEMA = {
    "type": "object",
    "properties": {
        "unsupported": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "fact_number": {"type": "integer"},
                    "reason": {"type": "string"},
                },
                "required": ["fact_number", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["unsupported"],
    "additionalProperties": False,
}

CONTRADICTION_SYSTEM = """\
You check two related Wiki pages for factual contradictions — statements that
cannot both be true (different dates for the same event, incompatible
relationships, conflicting attributes). Differences in coverage or emphasis
are not contradictions."""

CONTRADICTION_SCHEMA = {
    "type": "object",
    "properties": {
        "contradictions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"description": {"type": "string"}},
                "required": ["description"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["contradictions"],
    "additionalProperties": False,
}


def check_unsupported_facts(
    llm: LLM, config: WikiConfig, store: WikiStore, page_names: list[str]
) -> tuple[list[Finding], dict[str, list[int]]]:
    """Returns (findings, {page: [1-based fact numbers judged unsupported]})."""
    findings: list[Finding] = []
    to_remove: dict[str, list[int]] = {}
    for name in page_names:
        if not store.exists(name):
            continue
        page = store.load(name)
        facts = page.key_facts
        if not facts:
            continue
        source_ids = sorted(
            {ref for fact in facts for ref in SOURCE_REF_RE.findall(fact)}
        )
        sources = []
        for sid in source_ids:
            text = store.load_source(sid)
            if text:
                sources.append(f"--- source {sid} ---\n{text}")
        if not sources:
            continue
        numbered = "\n".join(f"{i}. {f}" for i, f in enumerate(facts, start=1))
        user = (
            f"PAGE: {page.title}\n\nFACTS:\n{numbered}\n\n"
            f"SOURCES:\n" + "\n\n".join(sources)
        )
        result = llm.structured(
            model=config.judge_model,
            system=cached_system(FACT_CHECK_SYSTEM),
            user=user,
            schema=FACT_CHECK_SCHEMA,
            max_tokens=2048,
            effort="low",
        )
        bad = []
        for item in result.get("unsupported", []):
            n = item.get("fact_number", 0)
            if 1 <= n <= len(facts):
                bad.append(n)
                findings.append(
                    Finding(
                        "unsupported_fact",
                        name,
                        f"fact {n} unsupported: {item.get('reason', '')[:120]}",
                    )
                )
        if bad:
            to_remove[name] = sorted(set(bad))
    return findings, to_remove


def remove_unsupported_facts(store: WikiStore, to_remove: dict[str, list[int]]) -> list[str]:
    log = []
    for name, numbers in to_remove.items():
        page = store.load(name)
        facts = page.key_facts
        kept = [f for i, f in enumerate(facts, start=1) if i not in numbers]
        page.set_section("Key Facts", "\n".join(f"- {f}" for f in kept))
        store.save(page, update_index=False)
        log.append(f"{name}: removed {len(numbers)} unsupported fact(s)")
    if to_remove:
        store.rebuild_all_indexes()
    return log


def check_contradictions(
    llm: LLM, config: WikiConfig, store: WikiStore, rng: random.Random | None = None
) -> list[Finding]:
    rng = rng or random.Random(0)
    pairs: list[tuple[str, str]] = []
    names = set(store.all_names())
    for name in sorted(names):
        for link in store.load(name).related_pages:
            if link in names and name < link:
                pairs.append((name, link))
    rng.shuffle(pairs)
    pairs = pairs[: config.contradiction_sample_pairs]

    findings = []
    for a, b in pairs:
        page_a, page_b = store.load(a), store.load(b)
        user = (
            f"PAGE A ({a}):\n{page_a.to_markdown()}\n\n"
            f"PAGE B ({b}):\n{page_b.to_markdown()}"
        )
        result = llm.structured(
            model=config.judge_model,
            system=cached_system(CONTRADICTION_SYSTEM),
            user=user,
            schema=CONTRADICTION_SCHEMA,
            max_tokens=1024,
            effort="low",
        )
        for item in result.get("contradictions", []):
            desc = item.get("description", "").strip()
            if desc:
                findings.append(
                    Finding("cross_page_contradiction", a, f"vs {b}: {desc[:160]}")
                )
    return findings
