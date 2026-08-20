"""Iterative wiki-build demo: compile one HotpotQA paragraph per invocation,
showing each internal stage (search candidates -> SelectPages -> compile ops
-> validators). State persists in the demo wiki directory.

Usage: uv run python step_build.py <step-number>
"""

import sys
from pathlib import Path

from llm_wiki.compiler.compile import compile_passage
from llm_wiki.compiler.select import select_pages
from llm_wiki.config import WikiConfig
from llm_wiki.errorbook.autofix import autofix
from llm_wiki.errorbook.validators import validate_structure
from llm_wiki.eval.data import load_eval_data
from llm_wiki.llm import make_llm
from llm_wiki.search.index import WikiSearchIndex
from llm_wiki.wiki.store import WikiStore

DEMO = Path(__file__).parent / "demo-wiki"
# Build order chosen to showcase create -> unrelated create -> merge/link -> link
TITLES = ["Scott Derrickson", "Ed Wood", "Sinister (film)", "Adam Collis"]

step = int(sys.argv[1])
title = TITLES[step - 1]

config = WikiConfig.load(DEMO)
config.provider = "claude-cli"
config.compiler_model = config.agent_model = config.judge_model = "claude-haiku-4-5"
store = WikiStore(config)
if step == 1:
    store.init()
config.save()

data = load_eval_data("hotpotqa", 20)
text = dict(data.corpus(20))[title]
source_id = title.replace(" ", "-").replace("(", "").replace(")", "")
passage = f"[Document: {title}]\n{text}"

print(f"### STEP {step}: compiling {title!r}")
print(f"--- passage ({len(text)} chars) ---")
print(text[:400] + ("..." if len(text) > 400 else ""))

llm = make_llm(config)
index = WikiSearchIndex(store)

print("\n--- stage 1: BM25 candidates for this passage ---")
candidates = index.search(passage, limit=20)
if not candidates:
    print("(wiki empty or no hits — SelectPages will be skipped)")
else:
    for r in candidates[:8]:
        print(f"  score={r.score:>7} {r.name}")

print("\n--- stage 2: SelectPages (LLM picks <=5 relevant existing pages) ---")
before = set(store.all_names())
selected = select_pages(llm, config, store, index, passage)
print(f"  selected: {selected or '(none)'}")

print("\n--- stage 3: CompileWikiPages ---")
result = compile_passage(
    llm, config, store,
    source_id=source_id, passage=passage,
    selected=selected, constraints=[],
)
for name in result.written:
    action = "UPDATED" if name in before else "created"
    page = store.load(name)
    print(f"  {action}: {name}  ({len(page.key_facts)} facts, "
          f"{len(set(page.outgoing_links()))} links)")

print("\n--- stage 4: validators ---")
findings = validate_structure(store)
if findings:
    for f in findings[:8]:
        print(f"  [{f.category}] {f.page}: {f.detail}")
    fixes = autofix(store)
    for line in fixes[:8]:
        print(f"  autofix: {line}")
    remaining = validate_structure(store)
    print(f"  {len(findings)} finding(s) -> {len(fixes)} fix(es) -> {len(remaining)} remaining")
else:
    print("  clean")

print(f"\n--- wiki now: {len(store.all_names())} pages ---")
for name in store.all_names():
    print(f"  {name}")
print(f"\nLLM usage this step: {llm.usage}")
