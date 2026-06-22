"""Diagnostic for the surprising CAD result (CAD hurt faithfulness, contra the paper).
For a few sources, print baseline vs CAD(0.5) vs CAD(1.0) answers + judge verdicts, so we
can tell whether CAD output is (a) garbled => implementation bug, or (b) coherent but less
grounded => the method genuinely doesn't transfer to a 3B 4-bit model.

  uv run python experiments/inspect_cad.py --tasks Data2txt,Summary --limit 4

Run from the repo root."""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from raganchor.eval.harness import free_gpu, select_sources
from raganchor.eval.judge import FaithfulnessJudge
from raganchor.llm import LocalLLM
from raganchor.rag import VanillaRAG
from raganchor.retrieval import HybridRetriever


def short(s: str, n: int = 500) -> str:
    s = " ".join(s.split())
    return textwrap.fill(s[:n] + ("…" if len(s) > n else ""), 100)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="Data2txt,Summary")
    ap.add_argument("--limit", type=int, default=4)
    args = ap.parse_args()
    tasks = [t.strip() for t in args.tasks.split(",")] if args.tasks else None
    sources = select_sources("test", tasks, args.limit)

    llm = LocalLLM()
    retriever = HybridRetriever()
    rags = {
        "baseline": VanillaRAG(llm, retriever, cad_alpha=None),
        "cad_0.5": VanillaRAG(llm, retriever, cad_alpha=0.5),
        "adacad": VanillaRAG(llm, retriever, cad_adaptive=True),
    }
    rows = [(s, {tag: rag.run(s) for tag, rag in rags.items()}) for s in sources]
    free_gpu(llm, retriever)
    del llm, retriever, rags

    judge = FaithfulnessJudge()
    for i, (src, outs) in enumerate(rows, 1):
        print("=" * 100)
        print(f"[{i}] task={src.task_type}  source_id={src.source_id}")
        ctx = " ".join(next(iter(outs.values())).contexts)
        print(f"CONTEXT:\n{short(ctx, 500)}")
        for tag, out in outs.items():
            j = judge.score(out.answer, out.contexts, question=src.question)
            verdict = "FAITHFUL" if j.is_faithful else f"UNFAITHFUL({j.n_spans})"
            a = f" mean_alpha={out.gen.mean_alpha}" if out.gen.mean_alpha is not None else ""
            print(f"\n-- {tag} [{verdict}]{a} --\n{short(out.answer, 500)}")


if __name__ == "__main__":
    main()
