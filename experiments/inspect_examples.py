"""Diagnostic: eyeball whether the pipeline is doing the right thing. For a few sources
(mixed tasks), generate the baseline answer, run the judge, and dump everything —
context, answer, judge verdict + flagged spans — so a human can check:
  - are generations sensible?
  - is the judge's faithful/unfaithful call correct on OUR (Qwen) outputs, not just RAGTruth's?
  - is QA faithfulness near-ceiling (little to gain)?

  uv run python experiments/inspect_examples.py --limit 8

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


def short(s: str, n: int = 600) -> str:
    s = " ".join(s.split())
    return textwrap.fill(s[:n] + ("…" if len(s) > n else ""), 100)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--tasks", default="", help="comma list; default all")
    args = ap.parse_args()
    tasks = [t.strip() for t in args.tasks.split(",")] if args.tasks else None
    sources = select_sources("test", tasks, args.limit)

    llm = LocalLLM()
    rag = VanillaRAG(llm)
    rows = [(s, rag.run(s)) for s in sources]
    free_gpu(llm, rag.retriever)
    del llm, rag

    judge = FaithfulnessJudge()
    for i, (src, out) in enumerate(rows, 1):
        j = judge.score(out.answer, out.contexts, question=src.question)
        print("=" * 100)
        print(f"[{i}] task={src.task_type}  source_id={src.source_id}  "
              f"judge: {'FAITHFUL' if j.is_faithful else 'UNFAITHFUL'} ({j.n_spans} spans)")
        if src.question:
            print(f"Q: {src.question}")
        print(f"CONTEXT:\n{short(' '.join(out.contexts), 700)}")
        print(f"ANSWER:\n{short(out.answer, 700)}")
        for sp in j.spans:
            print(f"  FLAGGED (conf {sp.get('confidence', 0):.2f}): {sp.get('text','').strip()[:200]}")


if __name__ == "__main__":
    main()
