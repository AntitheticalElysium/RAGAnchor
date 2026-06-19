"""Ablation: Provence query-aware context pruning (Cat 1). baseline vs conservative
(threshold 0.1) vs aggressive (0.5) sentence-level pruning. QA-only — Provence is
query-conditioned, so like reranking it doesn't apply to Summary/Data2txt.

This is the sentence-level counterpart to the passage-level rerank-then-prune ablation;
compare the cost (prompt tokens) / faithfulness tradeoff between the two.

  uv run python experiments/ablate_prune.py --limit 150

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
from raganchor.eval.harness import free_gpu, generate_records, judge_records, select_sources
from raganchor.eval.judge import FaithfulnessJudge
from raganchor.eval.metrics import aggregate
from raganchor.llm import LocalLLM
from raganchor.rag import VanillaRAG
from raganchor.retrieval import ContextPruner, HybridRetriever

# (tag, prune_threshold). None => no pruning (baseline).
CONFIGS: list[tuple[str, float | None]] = [
    ("baseline", None),
    ("provence_t0.1", 0.1),
    ("provence_t0.5", 0.5),
]

ROW = "{:<16}{:>6}{:>10}{:>12}{:>10}{:>10}{:>10}{:>9}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=150, help="0 = all")
    args = ap.parse_args()

    sources = select_sources(args.split, ["QA"], args.limit or None)
    print(f"[prune-ablate] {len(sources)} QA sources | split={args.split}")

    # --- phase 1: generate per config (LLM + embedder + Provence resident) ---
    llm = LocalLLM()
    retriever = HybridRetriever()  # shared (one embedder)
    pruner = ContextPruner()  # shared; threshold passed per config
    by_config: dict[str, list[dict]] = {}
    for tag, thr in CONFIGS:
        print(f"\n[prune-ablate] generating: {tag} (threshold={thr})")
        rag = VanillaRAG(
            llm, retriever, pruner=(pruner if thr is not None else None), prune_threshold=thr
        )
        by_config[tag] = generate_records(rag, sources, tag=tag)
    free_gpu(llm, retriever, pruner)
    del llm, retriever, pruner, rag

    # --- phase 2: judge every config ---
    judge = FaithfulnessJudge()
    for tag, records in by_config.items():
        judge_records(judge, records, tag=tag)
    del judge
    gc.collect()
    torch.cuda.empty_cache()

    # --- report ---
    summaries = {tag: aggregate(records) for tag, records in by_config.items()}
    print("\n=== Provence pruning ablation (raw per-config numbers) ===")
    print(ROW.format("config", "n", "faithful", "halluc_sp", "ttft_s", "lat_s", "prompt_t", "compl_t"))
    for tag, _ in CONFIGS:
        s = summaries[tag]
        print(ROW.format(
            tag, s["n"], s["faithful_rate"], s["mean_hallucination_spans"],
            s["mean_ttft_s"], s["mean_latency_s"], s["mean_prompt_tokens"], s["mean_completion_tokens"],
        ))

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out = RUNS_DIR / "ablation_prune.json"
    out.write_text(json.dumps(
        {"split": args.split, "tasks": ["QA"], "n": len(sources), "summaries": summaries}, indent=2
    ))
    print(f"\n[prune-ablate] wrote {out}")


if __name__ == "__main__":
    main()
