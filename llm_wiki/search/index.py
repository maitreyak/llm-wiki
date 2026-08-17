"""Field-weighted BM25 search over Wiki pages.

The paper's ``wiki_search`` prioritizes structured signals — page names,
aliases, tags — before page content. We implement that as separate BM25
indexes per field, fused with descending weights, plus an exact-match
boost when the query equals a title or alias.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from ..wiki.store import WikiStore

FIELD_WEIGHTS = {
    "name": 4.0,
    "aliases": 3.0,
    "tags": 2.0,
    "content": 1.0,
}
EXACT_MATCH_BOOST = 5.0


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25:
    """BM25 with the Lucene IDF form, which stays positive on small corpora
    (rank_bm25's BM25Okapi produces negative IDFs when a term appears in
    most documents, which is guaranteed for tiny wikis)."""

    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_freqs = [Counter(d) for d in docs]
        self.doc_lens = [len(d) for d in docs]
        self.avgdl = (sum(self.doc_lens) / len(docs)) if docs else 0.0
        n = len(docs)
        df: Counter[str] = Counter()
        for freq in self.doc_freqs:
            df.update(freq.keys())
        self.idf = {
            term: math.log(1.0 + (n - count + 0.5) / (count + 0.5))
            for term, count in df.items()
        }

    def get_scores(self, query: list[str]) -> list[float]:
        scores = [0.0] * len(self.doc_freqs)
        for term in query:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, freq in enumerate(self.doc_freqs):
                tf = freq.get(term, 0)
                if not tf:
                    continue
                denom = tf + self.k1 * (
                    1 - self.b + self.b * self.doc_lens[i] / (self.avgdl or 1.0)
                )
                scores[i] += idf * tf * (self.k1 + 1) / denom
        return scores


@dataclass
class SearchResult:
    name: str
    title: str
    score: float
    summary: str
    aliases: list[str]
    tags: list[str]


class WikiSearchIndex:
    def __init__(self, store: WikiStore):
        self.store = store
        self._names: list[str] = []
        self._meta: list[dict] = []
        self._indexes: dict[str, BM25 | None] = {}
        self._exact: dict[str, int] = {}
        self.rebuild()

    def rebuild(self) -> None:
        self._names = []
        self._meta = []
        fields: dict[str, list[list[str]]] = {f: [] for f in FIELD_WEIGHTS}
        self._exact = {}
        for page in self.store.all_pages():
            i = len(self._names)
            self._names.append(page.name)
            self._meta.append(
                {
                    "title": page.title,
                    "summary": page.summary.splitlines()[0] if page.summary else "",
                    "aliases": page.aliases,
                    "tags": page.tags,
                }
            )
            title_text = f"{page.title} {page.name.rsplit('/', 1)[-1].replace('-', ' ')}"
            fields["name"].append(tokenize(title_text))
            fields["aliases"].append(tokenize(" ".join(page.aliases)))
            fields["tags"].append(tokenize(" ".join(page.tags)))
            fields["content"].append(tokenize(page.all_text()))
            for key in [page.title, *page.aliases]:
                self._exact.setdefault(key.strip().lower(), i)
        self._indexes = {}
        for field_name, docs in fields.items():
            self._indexes[field_name] = BM25(docs) if any(docs) else None

    def search(self, query: str, limit: int = 8) -> list[SearchResult]:
        if not self._names:
            return []
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = [0.0] * len(self._names)
        for field_name, weight in FIELD_WEIGHTS.items():
            index = self._indexes.get(field_name)
            if index is None:
                continue
            for i, s in enumerate(index.get_scores(tokens)):
                scores[i] += weight * float(s)
        exact_hit = self._exact.get(query.strip().lower())
        if exact_hit is not None:
            scores[exact_hit] += EXACT_MATCH_BOOST
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        results = []
        for i in ranked[:limit]:
            if scores[i] <= 0:
                break
            m = self._meta[i]
            results.append(
                SearchResult(
                    name=self._names[i],
                    title=m["title"],
                    score=round(scores[i], 3),
                    summary=m["summary"],
                    aliases=m["aliases"],
                    tags=m["tags"],
                )
            )
        return results
