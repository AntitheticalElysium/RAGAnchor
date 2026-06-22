"""Ablation: Context-Aware Decoding (CAD, Shi et al. 2023). baseline vs alpha=0.5
(paper default for summarization) vs alpha=1.0. Runs ALL tasks — CAD is generation-side
and its value should show on the low-faithfulness slices (Summary/Data2txt ~0.72), where
there's headroom, unlike the near-ceiling QA slice.

CAD doubles decode cost (two forward passes/step), so watch latency vs faithfulness.

  uv run python experiments/ablate_cad.py --limit 150

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
from raganchor.retrieval import HybridRetriever

# (tag, cad_alpha, adaptive, warmup). alpha None + adaptive False => baseline greedy.
# adacad_w0.3 = paper's long-form warmup floor; adacad_w0.0 = pure JSD (alpha can ->0 on
# non-conflict, the real test of whether adaptation avoids over-correction here).
CONFIGS: list[tuple[str, float | None, bool, float]] = [
    ("baseline", None, False, 0.0),
    ("adacad_w0.3", None, True, 0.3),
    ("adacad_w0.0", None, True, 0.0),
]

ROW = "{:<12}{:>6}{:>10}{:>12}{:>10}{:>10}{:>10}{:>9}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--tasks", default="", help="comma list; default all tasks")
    ap.add_argument("--limit", type=int, default=150, help="0 = all")
    args = ap.parse_args()

    task_types = [t.strip() for t in args.tasks.split(",")] if args.tasks else None
    sources = select_sources(args.split, task_types, args.limit or None)
    print(f"[cad-ablate] {len(sources)} sources | tasks={task_types or 'all'}")

    # --- phase 1: generate per config (LLM + embedder resident) ---
    llm = LocalLLM()
    retriever = HybridRetriever()  # shared (QA retrieval only)
    by_config: dict[str, list[dict]] = {}
    for tag, alpha, adaptive, warmup in CONFIGS:
        print(f"\n[cad-ablate] generating: {tag} (alpha={alpha}, adaptive={adaptive}, warmup={warmup})")
        rag = VanillaRAG(llm, retriever, cad_alpha=alpha, cad_adaptive=adaptive, cad_warmup=warmup)
        by_config[tag] = generate_records(rag, sources, tag=tag)
    free_gpu(llm, retriever)
    del llm, retriever, rag

    # fair comparison: keep only sources that succeeded across ALL configs (OOM skips differ)
    common = set.intersection(*({r["source_id"] for r in recs} for recs in by_config.values()))
    for tag in by_config:
        by_config[tag] = [r for r in by_config[tag] if r["source_id"] in common]
    if len(common) < len(sources):
        print(f"[cad-ablate] comparing on {len(common)}/{len(sources)} common sources (rest OOM-skipped)")

    # --- phase 2: judge every config ---
    judge = FaithfulnessJudge()
    for tag, records in by_config.items():
        judge_records(judge, records, tag=tag)
    del judge
    gc.collect()
    torch.cuda.empty_cache()

    # --- report (overall + per-task, since CAD's headroom is task-dependent) ---
    summaries = {tag: aggregate(records) for tag, records in by_config.items()}
    print("\n=== CAD ablation (raw per-config numbers, all tasks) ===")
    print(ROW.format("config", "n", "faithful", "halluc_sp", "ttft_s", "lat_s", "prompt_t", "compl_t"))
    for tag, *_ in CONFIGS:
        s = summaries[tag]
        print(ROW.format(
            tag, s["n"], s["faithful_rate"], s["mean_hallucination_spans"],
            s["mean_ttft_s"], s["mean_latency_s"], s["mean_prompt_tokens"], s["mean_completion_tokens"],
        ))

    tasks_seen = sorted({r["task_type"] for r in next(iter(by_config.values()))})
    print("\n=== faithful_rate by task ===")
    print("{:<14}".format("config") + "".join(f"{t:>12}" for t in tasks_seen))
    for tag, records in by_config.items():
        cells = []
        for t in tasks_seen:
            rows = [r for r in records if r["task_type"] == t]
            cells.append(f"{sum(r['is_faithful'] for r in rows)/len(rows):.3f}" if rows else "-")
        print("{:<14}".format(tag) + "".join(f"{c:>12}" for c in cells))

    print("\n=== mean alpha by task (AdaCAD: how hard it pushed) ===")
    print("{:<14}".format("config") + "".join(f"{t:>12}" for t in tasks_seen))
    for tag, records in by_config.items():
        cells = []
        for t in tasks_seen:
            a = [r["mean_alpha"] for r in records if r["task_type"] == t and r.get("mean_alpha") is not None]
            cells.append(f"{sum(a)/len(a):.3f}" if a else "-")
        print("{:<14}".format(tag) + "".join(f"{c:>12}" for c in cells))

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out = RUNS_DIR / "ablation_cad.json"
    out.write_text(json.dumps(
        {"split": args.split, "tasks": task_types, "n": len(sources), "summaries": summaries}, indent=2
    ))
    print(f"\n[cad-ablate] wrote {out}")


if __name__ == "__main__":
    main()
