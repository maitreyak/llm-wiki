# llm-wiki

An implementation of **"Retrieval as Reasoning: Self-Evolving Agent-Native Retrieval via LLM-Wiki"** (arXiv 2605.25480), powered by the Claude API.

Instead of chunk-and-embed RAG, `llm-wiki`:

1. **Compiles** documents into a persistent, interlinked Markdown Wiki — pages with YAML frontmatter (aliases, tags), sourced Key Facts, and bidirectional `[[wikilinks]]`, organized under per-directory indexes.
2. **Retrieves agentically** — a Claude tool-use loop with `wiki_search` and `wiki_read` iteratively searches, reads, and follows links until it has the evidence to answer (budget Tmax=15 tool calls, patience P=3).
3. **Self-corrects** with an **Error Book** — deterministic validators and LLM checks find compilation errors (dangling links, malformed refs, index drift, unsupported facts, contradictions), repair them, and distill them into constraint rules injected into future compilation prompts.

## Install

```sh
uv sync            # library + CLI
uv sync --extra eval   # + benchmark eval harness (HuggingFace datasets)
```

Authentication uses the standard Anthropic SDK resolution (`ANTHROPIC_API_KEY` or `ant auth login`).

## Usage

```sh
llm-wiki init ./mywiki                       # create an empty wiki
llm-wiki ingest ./mywiki docs/*.md           # compile documents into the wiki
llm-wiki ask ./mywiki "Which director is older, X or Y?"
llm-wiki search ./mywiki "gilbert grape"     # inspect BM25 retrieval
llm-wiki fix ./mywiki                        # run validators + repairs
llm-wiki status ./mywiki                     # pages, links, open errors
llm-wiki eval hotpotqa --n 50                # benchmark run (cached wiki builds)
```

## Smoke test

With credentials set, this exercises the full loop — compile a tiny corpus, then answer a multi-hop comparison question that requires traversing film → director → birth-date evidence across pages:

```sh
llm-wiki init /tmp/smokewiki
llm-wiki ingest /tmp/smokewiki examples/smoke/*.md
llm-wiki status /tmp/smokewiki
llm-wiki ask /tmp/smokewiki --trace \
  "Which film's director was born first: What's Eating Gilbert Grape or Titanic?"
```

Expected: the trace shows `wiki_search`/`wiki_read` calls walking the link structure, and the answer is *What's Eating Gilbert Grape* (Hallström, 1946, vs Cameron, 1954).

## Development

```sh
uv run pytest            # offline unit tests
uv run pytest -m api     # integration tests that call the Claude API
```
