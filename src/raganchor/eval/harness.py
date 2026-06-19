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
from raganchor.eval.judge import FaithfulnessJudge
from raganchor.eval.metrics import aggregate
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


def generate_records(rag: VanillaRAG, sources: list[Source], tag: str = "") -> list[dict]:
    """Phase 1: run the RAG over sources with the LLM resident. No judging here."""
    records: list[dict] = []
    t0 = time.perf_counter()
    for i, src in enumerate(sources, 1):
        out = rag.run(src)
        records.append(
            {
                "source_id": src.source_id,
                "task_type": src.task_type,
                "question": src.question,
                "answer": out.answer,
                "contexts": out.contexts,
                "prompt_tokens": out.gen.prompt_tokens,
                "completion_tokens": out.gen.completion_tokens,
                "ttft_s": out.gen.ttft_s,
                "latency_s": out.gen.latency_s,
            }
        )
        if i % 10 == 0 or i == len(sources):
            print(f"[gen{':'+tag if tag else ''}] {i}/{len(sources)}  ({time.perf_counter()-t0:.0f}s)")
    return records


def judge_records(judge: FaithfulnessJudge, records: list[dict], tag: str = "") -> None:
    """Phase 2: grade faithfulness in place with the judge resident."""
    for i, rec in enumerate(records, 1):
        j = judge.score(rec["answer"], rec["contexts"], question=rec["question"])
        rec.update(is_faithful=j.is_faithful, n_hallucination_spans=j.n_spans)
        if i % 20 == 0 or i == len(records):
            print(f"[judge{':'+tag if tag else ''}] {i}/{len(records)}")


def write_run(tag: str, records: list[dict], extra_config: dict | None = None) -> dict:
    run = {
        "tag": tag,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "model": SETTINGS.model.model_id,
            "embedding_model": SETTINGS.retrieval.embedding_model,
            "judge_model": SETTINGS.judge.model_path,
            "top_k": SETTINGS.retrieval.top_k,
            **(extra_config or {}),
        },
        "summary": aggregate(records),
        "records": records,
    }
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RUNS_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{tag}.json"
    out_path.write_text(json.dumps(run, indent=2))
    print(f"[harness] wrote {out_path}")
    return run


def run_baseline(
    split: str = "test",
    task_types: list[str] | None = None,
    limit: int | None = 60,
    tag: str = "vanilla",
) -> dict:
    sources = select_sources(split, task_types, limit)
    print(f"[harness] {len(sources)} sources | split={split} | tasks={task_types or 'all'}")

    rag = VanillaRAG(LocalLLM(), HybridRetriever())
    records = generate_records(rag, sources)
    del rag
    gc.collect()
    torch.cuda.empty_cache()

    judge = FaithfulnessJudge()
    judge_records(judge, records)

    run = write_run(tag, records, {"split": split, "task_types": task_types})
    print(json.dumps(run["summary"], indent=2))
    return run
