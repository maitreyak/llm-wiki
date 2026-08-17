"""Benchmark dataset loaders (HotpotQA, MuSiQue, 2WikiMultiHopQA).

Following the paper's protocol: take the first N validation questions and
build the corpus from the union of their context paragraphs.

Requires the ``eval`` extra (HuggingFace ``datasets``). The MuSiQue and
2Wiki loaders use community mirrors on the Hub; override with the
LLM_WIKI_<DATASET>_HF env vars if a mirror moves.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class EvalItem:
    qid: str
    question: str
    answers: list[str]  # acceptable gold answers (first is canonical)
    paragraphs: list[tuple[str, str]]  # (title, text) context paragraphs
    qtype: str = ""
    hops: int | None = None


@dataclass
class EvalData:
    name: str
    items: list[EvalItem]

    def corpus(self, n_questions: int) -> list[tuple[str, str]]:
        """Union of context paragraphs of the first n questions, deduped by title."""
        seen: dict[str, str] = {}
        for item in self.items[:n_questions]:
            for title, text in item.paragraphs:
                if title not in seen:
                    seen[title] = text
        return sorted(seen.items())


def load_eval_data(dataset: str, limit: int) -> EvalData:
    loaders = {
        "hotpotqa": _load_hotpotqa,
        "musique": _load_musique,
        "2wikimultihopqa": _load_2wiki,
    }
    return loaders[dataset](limit)


def _hf_load(env_key: str, default_path: str, config: str | None, split: str):
    from datasets import load_dataset

    path = os.environ.get(env_key, default_path)
    try:
        if config:
            return load_dataset(path, config, split=split, trust_remote_code=False)
        return load_dataset(path, split=split, trust_remote_code=False)
    except Exception as exc:
        raise RuntimeError(
            f"failed to load HF dataset {path!r} ({exc}); set {env_key} to a "
            f"working dataset id"
        ) from exc


def _load_hotpotqa(limit: int) -> EvalData:
    rows = _hf_load("LLM_WIKI_HOTPOTQA_HF", "hotpot_qa", "distractor", f"validation[:{limit}]")
    items = []
    for row in rows:
        paragraphs = [
            (title, " ".join(sentences))
            for title, sentences in zip(row["context"]["title"], row["context"]["sentences"])
        ]
        items.append(
            EvalItem(
                qid=row["id"],
                question=row["question"],
                answers=[row["answer"]],
                paragraphs=paragraphs,
                qtype=row.get("type", ""),
                hops=2,
            )
        )
    return EvalData("hotpotqa", items)


def _load_musique(limit: int) -> EvalData:
    rows = _hf_load("LLM_WIKI_MUSIQUE_HF", "bdsaglam/musique", "answerable", f"validation[:{limit}]")
    items = []
    for row in rows:
        paragraphs = [
            (p["title"], p["paragraph_text"]) for p in row["paragraphs"]
        ]
        answers = [row["answer"]] + list(row.get("answer_aliases") or [])
        hops = None
        qid = row["id"]
        if isinstance(qid, str) and "hop" in qid:
            try:
                hops = int(qid.split("hop")[0])
            except ValueError:
                pass
        items.append(
            EvalItem(
                qid=str(qid),
                question=row["question"],
                answers=answers,
                paragraphs=paragraphs,
                hops=hops,
            )
        )
    return EvalData("musique", items)


def _load_2wiki(limit: int) -> EvalData:
    rows = _hf_load(
        "LLM_WIKI_2WIKI_HF", "xanhho/2WikiMultihopQA", None, f"validation[:{limit}]"
    )
    items = []
    for row in rows:
        context = row["context"]
        if isinstance(context, dict):  # hotpot-style dict of lists
            paragraphs = [
                (title, " ".join(sentences))
                for title, sentences in zip(context["title"], context["content"])
            ]
        else:  # list of [title, [sentences]]
            paragraphs = [(c[0], " ".join(c[1])) for c in context]
        items.append(
            EvalItem(
                qid=str(row.get("_id") or row.get("id")),
                question=row["question"],
                answers=[row["answer"]],
                paragraphs=paragraphs,
                qtype=row.get("type", ""),
            )
        )
    return EvalData("2wikimultihopqa", items)
