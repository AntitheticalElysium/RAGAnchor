"""Eval harness. Two phases to respect 6GB: generate everything with the LLM+retriever
resident, free the GPU, then load the NLI scorer and grade. Writes a run JSON."""

from __future__ import annotations

import gc
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from raganchor.config import RUNS_DIR, SETTINGS
from raganchor.data import Source, load_responses, load_sources
from raganchor.eval.metrics import aggregate
from raganchor.eval.nli import NLIScorer
from raganchor.llm import LocalLLM
from raganchor.rag import VanillaRAG
from raganchor.retrieval import HybridRetriever


def select_sources(
    split: str, task_types: list[str] | None, limit: int | None
) -> list[Source]:
    """Sources in a split, deduped in first-seen order. Split lives on responses."""
    srcmap = load_sources(task_types)
    seen: set[str] = set()
    ordered: list[Source] = []
    for r in load_responses(split):
        if r.source_id in seen or r.source_id not in srcmap:
            continue
        seen.add(r.source_id)
        ordered.append(srcmap[r.source_id])
        if limit and len(ordered) >= limit:
            break
    return ordered


def run_baseline(
    split: str = "test",
    task_types: list[str] | None = None,
    limit: int | None = 60,
    tag: str = "vanilla",
) -> dict:
    sources = select_sources(split, task_types, limit)
    print(f"[harness] {len(sources)} sources | split={split} | tasks={task_types or 'all'}")

    # --- phase 1: generate ---
    rag = VanillaRAG(LocalLLM(), HybridRetriever())
    records: list[dict] = []
    t0 = time.perf_counter()
    for i, src in enumerate(sources, 1):
        out = rag.run(src)
        records.append(
            {
                "source_id": src.source_id,
                "task_type": src.task_type,
                "answer": out.answer,
                "contexts": out.contexts,
                "prompt_tokens": out.gen.prompt_tokens,
                "completion_tokens": out.gen.completion_tokens,
                "ttft_s": out.gen.ttft_s,
                "latency_s": out.gen.latency_s,
            }
        )
        if i % 10 == 0 or i == len(sources):
            print(f"[gen] {i}/{len(sources)}  ({time.perf_counter()-t0:.0f}s)")

    del rag
    gc.collect()
    torch.cuda.empty_cache()

    # --- phase 2: score ---
    scorer = NLIScorer()
    for i, rec in enumerate(records, 1):
        f = scorer.score(rec["answer"], rec["contexts"])
        rec.update(
            faithfulness=f.score,
            n_claims=f.n_claims,
            n_supported=f.n_supported,
            contradiction_rate=f.contradiction_rate,
        )
        if i % 20 == 0 or i == len(records):
            print(f"[nli] {i}/{len(records)}")

    summary = aggregate(records)
    run = {
        "tag": tag,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "model": SETTINGS.model.model_id,
            "embedding_model": SETTINGS.retrieval.embedding_model,
            "nli_model": SETTINGS.nli.model_id,
            "top_k": SETTINGS.retrieval.top_k,
            "split": split,
            "task_types": task_types,
        },
        "summary": summary,
        "records": records,
    }
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RUNS_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{tag}.json"
    out_path.write_text(json.dumps(run, indent=2))
    print(f"[harness] wrote {out_path}")
    print(json.dumps(summary, indent=2))
    return run
