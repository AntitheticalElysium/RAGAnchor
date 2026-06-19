"""Ablation: post-generation NLI faithfulness gate (Cat 1). baseline vs gate-abstain vs
gate-retry. Runs on ALL task types (the gate is generation-side, not retrieval — unlike
reranking it isn't QA-bound). LLM + NLI scorer share the GPU in phase 1; the LettuceDetect
judge grades in phase 2.

Read the abstention trap: abstaining makes faithful_rate go up trivially (the judge scores
"I don't know" as faithful), so the honest columns are faithful_rate_answered (faithfulness
of answers actually given) and abstention_rate.

  uv run python experiments/ablate_gate.py --threshold 0.75 --limit 150

Run from the repo root."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from raganchor.config import RUNS_DIR, GateConfig
from raganchor.eval.harness import free_gpu, generate_records, judge_records, select_sources
from raganchor.eval.judge import FaithfulnessJudge
from raganchor.eval.metrics import aggregate
from raganchor.gate import NLIGate
from raganchor.llm import LocalLLM
from raganchor.rag import VanillaRAG

ROW = "{:<14}{:>6}{:>10}{:>11}{:>11}{:>9}{:>9}{:>8}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--tasks", default="", help="comma list; default all tasks")
    ap.add_argument("--limit", type=int, default=150, help="0 = all")
    ap.add_argument("--threshold", type=float, default=0.75, help="from tune_gate.py")
    args = ap.parse_args()

    task_types = [t.strip() for t in args.tasks.split(",")] if args.tasks else None
    sources = select_sources(args.split, task_types, args.limit or None)
    print(f"[gate-ablate] {len(sources)} sources | tasks={task_types or 'all'} | T={args.threshold}")

    configs = [
        ("baseline", None),
        ("gate_abstain", GateConfig(support_threshold=args.threshold, action="abstain")),
        ("gate_retry", GateConfig(support_threshold=args.threshold, action="retry")),
    ]

    # --- phase 1: generate per config (LLM + NLI scorer resident) ---
    llm = LocalLLM()
    rag = VanillaRAG(llm)
    scorer = None
    by_config: dict[str, list[dict]] = {}
    for tag, gcfg in configs:
        print(f"\n[gate-ablate] generating: {tag}")
        gate = None
        if gcfg is not None:
            from raganchor.eval.nli import NLIScorer

            scorer = scorer or NLIScorer()  # load once, shared across gate configs
            gate = NLIGate(scorer, gcfg)
        by_config[tag] = generate_records(rag, sources, tag=tag, gate=gate)
    free_gpu(llm, rag.retriever, scorer)
    del llm, rag, scorer

    # --- phase 2: judge every config ---
    judge = FaithfulnessJudge()
    for tag, records in by_config.items():
        judge_records(judge, records, tag=tag)
    del judge
    gc.collect()
    torch.cuda.empty_cache()

    # --- report ---
    summaries = {tag: aggregate(records) for tag, records in by_config.items()}
    print("\n=== gate ablation (raw per-config numbers) ===")
    print(ROW.format("config", "n", "faith_all", "faith_ans", "abstain", "retries", "lat_s", "compl_t"))
    for tag, _ in configs:
        s = summaries[tag]
        print(ROW.format(
            tag, s["n"], s["faithful_rate"], s.get("faithful_rate_answered", "-"),
            s.get("abstention_rate", "-"), s.get("mean_retries", "-"),
            s["mean_latency_s"], s["mean_completion_tokens"],
        ))

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out = RUNS_DIR / "ablation_gate.json"
    out.write_text(json.dumps(
        {"split": args.split, "tasks": task_types, "threshold": args.threshold,
         "n": len(sources), "summaries": summaries}, indent=2
    ))
    print(f"\n[gate-ablate] wrote {out}")


if __name__ == "__main__":
    main()
