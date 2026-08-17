"""The retrieval agent: a manual tool-use loop over wiki_search/wiki_read.

Termination per the paper: the agent answers when its evidence suffices,
or is cut off at Tmax=15 tool calls, or after P=3 consecutive empty
searches. At least one wiki_read must precede the final answer — if the
model tries to answer without reading, it gets nudged once.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from ..config import WikiConfig
from ..llm import LLM, cached_system
from ..search.index import WikiSearchIndex
from ..wiki.store import WikiStore
from .tools import WIKI_READ_TOOL, WIKI_SEARCH_TOOL, WikiTools

AGENT_SYSTEM = """\
You answer questions using a knowledge Wiki, via two tools: wiki_search and
wiki_read. The Wiki's pages carry sourced facts and [[wikilinks]] between
related entities.

Strategy:
- For a known entity, search for it, read its page, and follow relevant
  [[wikilinks]] to reach connected evidence (bridge entities, attributes on
  linked pages). Multi-hop questions usually require reading several pages —
  identify the bridge entity on one page, then read its page.
- For exploratory questions, read `index` and directory `_index` listings to
  browse what exists, then read the promising pages.
- Batch reads: pass several paths to one wiki_read call when you already know
  what you need.
- You MUST base your answer only on what you read from the Wiki, and you must
  read at least one page before answering. If the Wiki genuinely lacks the
  answer, say so.

When you have sufficient evidence, give your final answer.

Format the final answer as:
ANSWER: <short answer — the entity, date, yes/no, etc.>
EVIDENCE: <one or two sentences citing the pages used>"""


@dataclass
class AgentTrace:
    tool_calls: list[dict] = field(default_factory=list)
    pages_read: list[str] = field(default_factory=list)
    searches: int = 0
    reads: int = 0
    stop_reason: str = "answered"


@dataclass
class AgentResult:
    answer: str
    full_text: str
    trace: AgentTrace


def _extract_answer(text: str) -> str:
    for line in text.splitlines():
        if line.strip().upper().startswith("ANSWER:"):
            return line.split(":", 1)[1].strip()
    return text.strip()


def ask(
    llm: LLM,
    config: WikiConfig,
    store: WikiStore,
    question: str,
    *,
    index: WikiSearchIndex | None = None,
    tools_impl: WikiTools | None = None,
) -> AgentResult:
    tools_impl = tools_impl or WikiTools(store, index=index)
    trace = AgentTrace()
    messages: list[dict] = [{"role": "user", "content": question}]
    empty_streak = 0
    nudged = False

    while True:
        response = llm.message(
            model=config.agent_model,
            system=cached_system(AGENT_SYSTEM),
            messages=messages,
            max_tokens=config.max_output_tokens,
            tools=[WIKI_SEARCH_TOOL, WIKI_READ_TOOL],
        )
        tool_uses = [b for b in response.content if b.type == "tool_use"]

        if response.stop_reason != "tool_use" or not tool_uses:
            text = "".join(b.text for b in response.content if b.type == "text")
            if trace.reads == 0 and not nudged and store.all_names():
                nudged = True
                messages.append({"role": "assistant", "content": response.content})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "You must read at least one Wiki page with wiki_read "
                            "before giving a final answer. Continue."
                        ),
                    }
                )
                continue
            return AgentResult(
                answer=_extract_answer(text), full_text=text, trace=trace
            )

        messages.append({"role": "assistant", "content": response.content})
        results = []
        for block in tool_uses:
            budget_left = config.agent_max_tool_calls - len(trace.tool_calls)
            if budget_left <= 0:
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": "Tool budget exhausted. Answer now from what you have read.",
                        "is_error": True,
                    }
                )
                continue
            output, is_error = _run_tool(tools_impl, block.name, block.input, trace)
            if block.name == "wiki_search":
                trace.searches += 1
                empty_streak = empty_streak + 1 if output.strip() == "No results." else 0
            elif block.name == "wiki_read":
                trace.reads += 1
            trace.tool_calls.append({"tool": block.name, "input": block.input})
            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": output,
                    "is_error": is_error,
                }
            )
        messages.append({"role": "user", "content": results})

        if len(trace.tool_calls) >= config.agent_max_tool_calls:
            trace.stop_reason = "budget"
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Tool budget reached. Give your final answer now based on "
                        "what you have read, in the required format."
                    ),
                }
            )
            final = llm.message(
                model=config.agent_model,
                system=cached_system(AGENT_SYSTEM),
                messages=messages,
                max_tokens=config.max_output_tokens,
            )
            text = "".join(b.text for b in final.content if b.type == "text")
            return AgentResult(answer=_extract_answer(text), full_text=text, trace=trace)

        if empty_streak >= config.agent_patience:
            trace.stop_reason = "patience"
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Repeated searches found nothing. Either read the directory "
                        "indexes to browse, or give your final answer now."
                    ),
                }
            )
            empty_streak = 0


def _run_tool(tools_impl: WikiTools, name: str, tool_input, trace: AgentTrace):
    try:
        if isinstance(tool_input, str):  # defensive; SDK normally parses
            tool_input = json.loads(tool_input)
        if name == "wiki_search":
            return tools_impl.wiki_search(tool_input["query"]), False
        if name == "wiki_read":
            paths = tool_input["paths"]
            trace.pages_read.extend(paths)
            return tools_impl.wiki_read(paths), False
        return f"Unknown tool: {name}", True
    except Exception as exc:  # noqa: BLE001 — report tool failure to the model
        return f"Tool error: {exc}", True
