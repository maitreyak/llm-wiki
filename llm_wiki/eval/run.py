"""Benchmark runner: build (or reuse) a Wiki from the corpus, run the
retrieval agent on each question, report F1/EM.

Compiled wikis are cached by corpus content hash — compilation is the
expensive step and is amortized across eval runs (paper §limitations).
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from ..agent.loop import ask
from ..compiler.pipeline import Document, ingest
from ..config import WikiConfig
from ..errorbook.manager import ErrorBookManager
from ..llm import make_llm
from ..search.index import WikiSearchIndex
from ..wiki.page import slugify
from ..wiki.store import WikiStore
from .data import load_eval_data
from .metrics import best_scores

DEFAULT_CACHE = Path.home() / ".cache" / "llm-wiki"


@dataclass
class QuestionResult:
    qid: str
    question: str
    gold: str
    prediction: str
    f1: float
    em: float
    qtype: str
    hops: int | None
    tool_calls: int
    stop_reason: str
    seconds: float


def corpus_hash(paragraphs: list[tuple[str, str]]) -> str:
    h = hashlib.sha256()
    for title, text in paragraphs:
        h.update(title.encode())
        h.update(b"\x00")
        h.update(text.encode())
        h.update(b"\x01")
    return h.hexdigest()[:12]


def build_or_load_wiki(
    llm,
    dataset: str,
    paragraphs: list[tuple[str, str]],
    cache_dir: Path,
    rebuild: bool,
    echo: Callable[[str], None],
) -> tuple[WikiConfig, WikiStore]:
    root = cache_dir / f"{dataset}-{corpus_hash(paragraphs)}"
    config = WikiConfig.load(root)
    store = WikiStore(config)
    marker = root / ".build-complete"
    if marker.exists() and not rebuild:
        echo(f"reusing cached wiki at {root} ({len(store.all_names())} pages)")
        return config, store

    echo(f"building wiki at {root} from {len(paragraphs)} paragraphs ...")
    if root.exists():
        import shutil

        shutil.rmtree(root)
    store.init()
    docs = [
        Document(doc_id=slugify(title)[:60], text=text, title=title)
        for title, text in paragraphs
    ]
    hooks = ErrorBookManager(llm, config, store)
    report = ingest(
        llm,
        config,
        store,
        docs,
        hooks=hooks,
        constraints_fn=hooks.active_constraints,
        progress=echo,
    )
    echo(f"built: {report}; usage so far: {llm.usage}")
    marker.write_text("ok\n")
    return config, store


def run_eval(
    dataset: str,
    *,
    n: int,
    corpus_questions: int,
    cache_dir: Path | None,
    rebuild: bool,
    echo: Callable[[str], None],
    provider: str = "anthropic",
    model: str | None = None,
) -> dict:
    cache_dir = cache_dir or DEFAULT_CACHE
    cache_dir.mkdir(parents=True, exist_ok=True)
    data = load_eval_data(dataset, limit=max(n, corpus_questions))
    paragraphs = data.corpus(corpus_questions)

    # The wiki root depends only on corpus content; provider/model settings are
    # applied to its config so cached wikis remember what built them.
    root = cache_dir / f"{dataset}-{corpus_hash(paragraphs)}"
    probe = WikiConfig.load(root)
    probe.provider = provider
    if model:
        probe.compiler_model = probe.agent_model = probe.judge_model = model
    llm = make_llm(probe)
    config, store = build_or_load_wiki(llm, dataset, paragraphs, cache_dir, rebuild, echo)
    config.provider = probe.provider
    config.compiler_model = probe.compiler_model
    config.agent_model = probe.agent_model
    config.judge_model = probe.judge_model
    config.save()

    index = WikiSearchIndex(store)
    results: list[QuestionResult] = []
    for i, item in enumerate(data.items[:n], start=1):
        start = time.monotonic()
        try:
            agent_result = ask(llm, config, store, item.question, index=index)
            prediction = agent_result.answer
            tool_calls = len(agent_result.trace.tool_calls)
            stop = agent_result.trace.stop_reason
        except Exception as exc:  # noqa: BLE001 — one bad question shouldn't kill the run
            prediction, tool_calls, stop = "", 0, f"error: {exc}"
        f1, em = best_scores(prediction, item.answers)
        results.append(
            QuestionResult(
                qid=item.qid,
                question=item.question,
                gold=item.answers[0],
                prediction=prediction,
                f1=round(f1, 4),
                em=round(em, 4),
                qtype=item.qtype,
                hops=item.hops,
                tool_calls=tool_calls,
                stop_reason=stop,
                seconds=round(time.monotonic() - start, 1),
            )
        )
        echo(
            f"[{i}/{n}] f1={f1:.2f} em={em:.0f} calls={tool_calls} "
            f"q={item.question[:70]!r} pred={prediction[:50]!r}"
        )

    summary = summarize(results)
    out_path = cache_dir / f"{dataset}-results-{int(time.time())}.json"
    out_path.write_text(
        json.dumps(
            {"dataset": dataset, "n": n, "summary": summary,
             "results": [asdict(r) for r in results]},
            indent=2,
        )
    )
    echo("")
    echo(f"== {dataset} (n={len(results)}) ==")
    echo(f"F1: {summary['f1']:.3f}  EM: {summary['em']:.3f}")
    for key, stats in sorted(summary.get("by_type", {}).items()):
        echo(f"  type={key}: F1 {stats['f1']:.3f} (n={stats['n']})")
    for key, stats in sorted(summary.get("by_hops", {}).items()):
        echo(f"  hops={key}: F1 {stats['f1']:.3f} (n={stats['n']})")
    echo(f"results written to {out_path}")
    echo(f"total LLM usage: {llm.usage}")
    return summary


def summarize(results: list[QuestionResult]) -> dict:
    if not results:
        return {"f1": 0.0, "em": 0.0, "n": 0}
    summary = {
        "f1": sum(r.f1 for r in results) / len(results),
        "em": sum(r.em for r in results) / len(results),
        "n": len(results),
        "avg_tool_calls": sum(r.tool_calls for r in results) / len(results),
        "by_type": _grouped(results, key=lambda r: r.qtype or None),
        "by_hops": _grouped(results, key=lambda r: r.hops),
    }
    return summary


def _grouped(results: list[QuestionResult], key) -> dict:
    groups: dict = {}
    for r in results:
        k = key(r)
        if k is None:
            continue
        groups.setdefault(k, []).append(r)
    return {
        str(k): {
            "f1": sum(r.f1 for r in rs) / len(rs),
            "em": sum(r.em for r in rs) / len(rs),
            "n": len(rs),
        }
        for k, rs in groups.items()
    }
