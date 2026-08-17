"""llm-wiki command line interface."""

from __future__ import annotations

import sys
from pathlib import Path

import click

from .config import WikiConfig
from .wiki.store import WikiStore


def _open(root: str) -> tuple[WikiConfig, WikiStore]:
    config = WikiConfig.load(Path(root))
    store = WikiStore(config)
    if not config.pages_dir.exists():
        raise click.ClickException(
            f"{root} is not an initialized wiki (run: llm-wiki init {root})"
        )
    return config, store


@click.group()
def main() -> None:
    """LLM-Wiki: compile documents into a Wiki and answer questions over it."""


@main.command()
@click.argument("root", type=click.Path())
def init(root: str) -> None:
    """Create an empty wiki at ROOT."""
    config = WikiConfig.load(Path(root))
    WikiStore(config).init()
    click.echo(f"Initialized empty wiki at {root}")


@main.command()
@click.argument("root", type=click.Path(exists=True))
@click.argument("files", nargs=-1, required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--no-repair", is_flag=True, help="Skip Error Book validation/repair.")
def ingest(root: str, files: tuple[Path, ...], no_repair: bool) -> None:
    """Compile documents into the wiki."""
    from .compiler.pipeline import ingest as run_ingest, load_documents
    from .errorbook.manager import ErrorBookManager
    from .llm import LLM

    config, store = _open(root)
    llm = LLM()
    hooks = None if no_repair else ErrorBookManager(llm, config, store)
    docs = load_documents(list(files))
    report = run_ingest(
        llm,
        config,
        store,
        docs,
        hooks=hooks,
        constraints_fn=(hooks.active_constraints if hooks else None),
        progress=lambda msg: click.echo(msg, err=True),
    )
    click.echo(f"Ingested: {report}")
    if hooks and hooks.fix_log:
        click.echo(f"Repairs applied: {len(hooks.fix_log)} (see error_book.yaml)")
    click.echo(f"LLM usage: {llm.usage}")


@main.command()
@click.argument("root", type=click.Path(exists=True))
@click.argument("question")
@click.option("--trace", is_flag=True, help="Show the tool-call trace.")
def ask(root: str, question: str, trace: bool) -> None:
    """Answer a question using the wiki."""
    from .agent.loop import ask as run_ask
    from .llm import LLM

    config, store = _open(root)
    llm = LLM()
    result = run_ask(llm, config, store, question)
    click.echo(result.full_text)
    if trace:
        click.echo("\n--- trace ---", err=True)
        for call in result.trace.tool_calls:
            click.echo(f"{call['tool']}: {call['input']}", err=True)
        click.echo(
            f"stop: {result.trace.stop_reason}; "
            f"{result.trace.searches} searches, {result.trace.reads} reads; "
            f"usage: {llm.usage}",
            err=True,
        )


@main.command()
@click.argument("root", type=click.Path(exists=True))
@click.argument("query")
def search(root: str, query: str) -> None:
    """Run a BM25 search against the wiki (no LLM)."""
    from .agent.tools import WikiTools

    _config, store = _open(root)
    click.echo(WikiTools(store).wiki_search(query))


@main.command()
@click.argument("root", type=click.Path(exists=True))
@click.option("--llm-checks", is_flag=True, help="Also run LLM fact/contradiction checks.")
def fix(root: str, llm_checks: bool) -> None:
    """Run validators and repairs on the wiki."""
    from .errorbook.autofix import autofix
    from .errorbook.book import ErrorBook
    from .errorbook.validators import validate_structure

    config, store = _open(root)
    book = ErrorBook(config.error_book_path)
    findings = validate_structure(store)
    book.record_findings(findings)
    log = autofix(store)
    remaining = validate_structure(store)
    book.verify_and_close(remaining)

    if llm_checks:
        from .errorbook.llm_checks import (
            check_contradictions,
            check_unsupported_facts,
            remove_unsupported_facts,
        )
        from .llm import LLM

        llm = LLM()
        fact_findings, to_remove = check_unsupported_facts(
            llm, config, store, store.all_names()
        )
        book.record_findings(fact_findings)
        log.extend(remove_unsupported_facts(store, to_remove))
        book.record_findings(check_contradictions(llm, config, store))
    book.save()

    for line in log:
        click.echo(f"fixed: {line}")
    click.echo(
        f"{len(findings)} finding(s), {len(log)} fix(es), "
        f"{len(remaining)} structural issue(s) remaining"
    )


@main.command()
@click.argument("root", type=click.Path(exists=True))
def status(root: str) -> None:
    """Show wiki statistics and open errors."""
    from .errorbook.book import ErrorBook

    config, store = _open(root)
    names = store.all_names()
    links = sum(len(set(p.outgoing_links())) for p in store.all_pages())
    click.echo(f"pages: {len(names)}  directories: {len(store.directories())}  links: {links}")
    book = ErrorBook(config.error_book_path)
    if book.records:
        for category, counts in sorted(book.summary().items()):
            click.echo(
                f"errors[{category}]: {counts['open']} open, "
                f"{counts['closed']} closed, {counts['occurrences']} occurrences"
            )
    else:
        click.echo("error book: empty")


@main.command(name="eval")
@click.argument("dataset", type=click.Choice(["hotpotqa", "musique", "2wikimultihopqa"]))
@click.option("--n", default=50, show_default=True, help="Number of questions.")
@click.option("--corpus-questions", default=None, type=int,
              help="Questions whose contexts form the corpus (default: same as --n).")
@click.option("--cache-dir", default=None, type=click.Path(), help="Wiki cache directory.")
@click.option("--rebuild", is_flag=True, help="Force wiki rebuild even if cached.")
def eval_cmd(dataset: str, n: int, corpus_questions: int | None, cache_dir: str | None, rebuild: bool) -> None:
    """Run a benchmark evaluation."""
    try:
        from .eval.run import run_eval
    except ImportError as exc:
        raise click.ClickException(
            f"eval extras not installed ({exc}); run: uv sync --extra eval"
        ) from exc

    run_eval(
        dataset,
        n=n,
        corpus_questions=corpus_questions or n,
        cache_dir=Path(cache_dir) if cache_dir else None,
        rebuild=rebuild,
        echo=click.echo,
    )


if __name__ == "__main__":
    sys.exit(main())
