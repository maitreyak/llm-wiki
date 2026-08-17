"""Prompts and structured-output schemas for the Wiki compiler."""

from __future__ import annotations

CANONICAL_TYPES = [
    "people",
    "organizations",
    "media",
    "works",
    "places",
    "events",
    "concepts",
    "topics",
]

SELECT_PAGES_SYSTEM = """\
You maintain a knowledge Wiki compiled from source documents. Given a new
source passage, identify which EXISTING Wiki pages are relevant — pages this
passage adds facts to, or pages its entities should link to.

You will see search candidates and directory listings from the current Wiki.
Select only pages that actually exist in those listings, up to the stated
maximum. Prefer pages about entities the passage mentions directly. If nothing
existing is relevant, return an empty list.
"""

SELECT_PAGES_SCHEMA = {
    "type": "object",
    "properties": {
        "pages": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Logical names of existing relevant pages",
        }
    },
    "required": ["pages"],
    "additionalProperties": False,
}


COMPILE_SYSTEM_TEMPLATE = """\
You compile source passages into a knowledge Wiki of interlinked entity pages.

For the given passage, produce the set of Wiki pages to create or update so the
Wiki fully captures the passage's factual content. Follow these rules:

ENTITIES AND PAGES
- Create one page per salient entity (person, organization, film/book/work,
  place, event, concept). Choose the page type from: {types} — use the most
  specific fit; "topics" is the fallback.
- If a RELEVANT EXISTING PAGE covers an entity, UPDATE it (you are given its
  current content): keep all its existing facts, merge in the new ones, and
  return the complete merged page. Never drop existing facts or links.
- Do not create near-duplicate pages for the same entity under different names.

FACTS
- Each key fact is one self-contained sentence ending with a source reference
  in exactly the form [src:SOURCE_ID], where SOURCE_ID is the passage's source
  id given in the user message.
- Only state facts supported by the passage (or already present on the page,
  keeping their original [src:...] references). Never invent facts or copy
  unsupported claims.

LINKS
- When a fact mentions another entity that has (or will have after this
  compilation) a Wiki page, write it as a wikilink: [[type/Page-Name]].
  Use the exact logical names of existing pages, or the name that will be
  derived for pages you create in this same response (type/slugified-title).
- List each linked page once under related_pages with a short note on the
  relationship.

ALIASES AND TAGS
- aliases: alternative names/spellings a reader might search for.
- tags: a few lowercase keywords for browsing.
{constraints}"""

COMPILE_SCHEMA = {
    "type": "object",
    "properties": {
        "digest": {
            "type": "string",
            "description": "Faithful 2-4 sentence digest of the passage",
        },
        "pages": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "type": {"type": "string"},
                    "existing_name": {
                        "anyOf": [{"type": "string"}, {"type": "null"}],
                        "description": "Logical name if updating an existing page, else null",
                    },
                    "aliases": {"type": "array", "items": {"type": "string"}},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "summary": {"type": "string"},
                    "key_facts": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Each fact ends with [src:...]",
                    },
                    "related_pages": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "note": {"type": "string"},
                            },
                            "required": ["name", "note"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": [
                    "title",
                    "type",
                    "existing_name",
                    "aliases",
                    "tags",
                    "summary",
                    "key_facts",
                    "related_pages",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": ["digest", "pages"],
    "additionalProperties": False,
}


def compile_system(constraints: list[str]) -> str:
    block = ""
    if constraints:
        rules = "\n".join(f"- {c}" for c in constraints)
        block = (
            "\n\nLEARNED CONSTRAINTS (from past compilation errors — follow strictly)\n"
            + rules
        )
    return COMPILE_SYSTEM_TEMPLATE.format(
        types=", ".join(CANONICAL_TYPES), constraints=block
    )
