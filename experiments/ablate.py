"""Ablation runner. One config per row: baseline, then each method toggled on its own
(individually — we don't test the power set). Loads the LLM once and the judge once and
reuses them across configs (6GB-friendly), then prints a raw comparison table.

First method: reranking. On RAGTruth QA there are only 3 passages, so a reranker can't
improve recall — instead we use it to *prune*: rerank the 3 passages, keep the top
rerank_keep, and watch faithfulness vs token cost. Hence QA-only by default.

  uv run python experiments/ablate.py --limit 50
  uv run python experiments/ablate.py --tasks QA --limit 0     # full QA test slice

Run from the repo root."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from raganchor.config import RUNS_DIR
from raganchor.eval.harness import generate_records, judge_records, select_sources
from raganchor.eval.judge import FaithfulnessJudge
from raganchor.eval.metrics import aggregate
from raganchor.llm import LocalLLM
from raganchor.rag import VanillaRAG
from raganchor.retrieval import HybridRetriever, Reranker

# (tag, rerank_keep). rerank_keep None => no reranker (baseline). Each row toggles ONE
# thing vs baseline. Retrieval params are identical across rows, so one retriever +
# one reranker are shared (load each model once — 6GB-friendly).
CONFIGS: list[tuple[str, int | None]] = [
    ("baseline", None),
    ("rerank_keep2", 2),
    ("rerank_keep1", 1),
]

ROW = "{:<14}{:>8}{:>10}{:>12}{:>10}{:>10}{:>10}{:>9}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--tasks", default="QA", help="comma list; reranking only bites on QA")
    ap.add_argument("--limit", type=int, default=50, help="0 = all")
    args = ap.parse_args()

    task_types = [t.strip() for t in args.tasks.split(",")] if args.tasks else None
    sources = select_sources(args.split, task_types, args.limit or None)
    print(f"[ablate] {len(sources)} sources | split={args.split} | tasks={task_types or 'all'}")

    # --- phase 1: generate per config (LLM + embedder + reranker resident) ---
    llm = LocalLLM()
    retriever = HybridRetriever()  # shared (one embedder)
    reranker = Reranker()  # shared model; per-config keep is passed explicitly
    by_config: dict[str, list[dict]] = {}
    for tag, keep in CONFIGS:
        print(f"\n[ablate] generating: {tag} (rerank_keep={keep})")
        rag = VanillaRAG(
            llm, retriever, reranker=(reranker if keep else None), rerank_keep=keep
        )
        by_config[tag] = generate_records(rag, sources, tag=tag)
    del llm, retriever, reranker, rag  # release every GPU ref before the judge loads
    gc.collect()
    torch.cuda.empty_cache()

    # --- phase 2: judge every config (judge resident) ---
    judge = FaithfulnessJudge()
    for tag, records in by_config.items():
        judge_records(judge, records, tag=tag)
    del judge
    gc.collect()
    torch.cuda.empty_cache()

    # --- report ---
    summaries = {tag: aggregate(records) for tag, records in by_config.items()}
    print("\n=== ablation (raw per-config numbers) ===")
    print(ROW.format("config", "n", "faithful", "halluc_sp", "ttft_s", "lat_s", "prompt_t", "compl_t"))
    for tag, _ in CONFIGS:
        s = summaries[tag]
        print(ROW.format(
            tag, s["n"], s["faithful_rate"], s["mean_hallucination_spans"],
            s["mean_ttft_s"], s["mean_latency_s"], s["mean_prompt_tokens"], s["mean_completion_tokens"],
        ))

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out = RUNS_DIR / "ablation_rerank.json"
    out.write_text(json.dumps(
        {"split": args.split, "tasks": task_types, "n": len(sources), "summaries": summaries}, indent=2
    ))
    print(f"\n[ablate] wrote {out}")


if __name__ == "__main__":
    main()
