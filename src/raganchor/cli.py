"""RAGAnchor CLI."""

from __future__ import annotations

import typer

app = typer.Typer(add_completion=False, help="RAGAnchor — agentic RAG eval harness")


def _tasks(s: str | None) -> list[str] | None:
    return [t.strip() for t in s.split(",")] if s else None


@app.command()
def baseline(
    split: str = "test",
    tasks: str = typer.Option(None, help="comma list: QA,Summary,Data2txt"),
    limit: int = typer.Option(60, help="number of sources (0 = all)"),
    tag: str = "vanilla",
):
    """Run the vanilla RAG baseline over a RAGTruth slice."""
    from raganchor.eval.harness import run_baseline

    run_baseline(split=split, task_types=_tasks(tasks), limit=limit or None, tag=tag)


@app.command()
def smoke():
    """Load the model and generate once — sanity check."""
    from raganchor.models.llm import LocalLLM

    llm = LocalLLM()
    msgs = [
        {"role": "system", "content": "Answer only from the provided context."},
        {"role": "user", "content": "Context: The shop is at 826 B Street.\n\nWhere is the shop?"},
    ]
    r = llm.generate(msgs, max_new_tokens=48)
    typer.echo(f"{r.text}\n[{r.completion_tokens} tok, {r.latency_s:.2f}s]")


if __name__ == "__main__":
    app()
