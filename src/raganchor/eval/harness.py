"""Eval harness. Two phases to respect 6GB: generate everything with the LLM+retriever
resident, free the GPU, then load the NLI scorer and grade. Writes a run JSON."""

from __future__ import annotations

import gc
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from raganchor.config import RUNS_DIR, SETTINGS
from raganchor.data import Source, load_responses, load_sources
from raganchor.eval.judge import FaithfulnessJudge
from raganchor.eval.metrics import aggregate
from raganchor.gate import NLIGate
from raganchor.llm import LocalLLM
from raganchor.rag import VanillaRAG
from raganchor.retrieval import HybridRetriever


def free_gpu(*objs) -> None:
    """Release phase-1 models before the judge loads. On a 6GB GPU `del + empty_cache`
    isn't enough — the inner model tensors linger (esp. trust_remote_code models), so we
    null the known model-holding attributes first, then collect twice."""
    for o in objs:
        for attr in ("model", "_model", "_embedder", "tokenizer", "detector", "scorer"):
            if hasattr(o, attr):
                setattr(o, attr, None)
    for _ in range(2):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def select_sources(
    split: str, task_types: list[str] | None, limit: int | None
) -> list[Source]:
    """Sources in a split (split lives on responses). Shuffled deterministically so a
    --limit slice is representative across tasks, not task-ordered like the file."""
    srcmap = load_sources(task_types)
    seen: set[str] = set()
    ordered: list[Source] = []
    for r in load_responses(split):
        if r.source_id in seen or r.source_id not in srcmap:
            continue
        seen.add(r.source_id)
        ordered.append(srcmap[r.source_id])
    random.Random(SETTINGS.seed).shuffle(ordered)
    return ordered[:limit] if limit else ordered


def generate_records(
    rag: VanillaRAG, sources: list[Source], tag: str = "", gate: NLIGate | None = None
) -> list[dict]:
    """Phase 1: run the RAG over sources with the LLM (+ NLI gate, if given) resident.
    No LettuceDetect judging here — that's phase 2."""
    records: list[dict] = []
    skipped = 0
    t0 = time.perf_counter()
    for i, src in enumerate(sources, 1):
        try:
            if gate is not None:
                g = gate.run(rag, src)
                rec = {
                    "answer": g.answer,
                    "contexts": g.contexts,
                    "prompt_tokens": g.gen.prompt_tokens,
                    "completion_tokens": g.gen.completion_tokens,
                    "ttft_s": g.gen.ttft_s,
                    "latency_s": g.latency_s,  # whole gate: gen + NLI + retries
                    "nli_support": g.nli_support,
                    "is_abstained": g.is_abstained,
                    "n_retries": g.n_retries,
                }
            else:
                out = rag.run(src)
                rec = {
                    "answer": out.answer,
                    "contexts": out.contexts,
                    "prompt_tokens": out.gen.prompt_tokens,
                    "completion_tokens": out.gen.completion_tokens,
                    "ttft_s": out.gen.ttft_s,
                    "latency_s": out.gen.latency_s,
                    "mean_alpha": out.gen.mean_alpha,  # CAD/AdaCAD only (else None)
                }
        except torch.cuda.OutOfMemoryError:
            # one long example (e.g. CAD's 2x prefill) shouldn't kill the whole run
            skipped += 1
            torch.cuda.empty_cache()
            print(f"[gen{':'+tag if tag else ''}] OOM on source {src.source_id} ({src.task_type}) — skipped")
            continue
        rec = {"source_id": src.source_id, "task_type": src.task_type, "question": src.question, **rec}
        records.append(rec)
        if i % 10 == 0 or i == len(sources):
            print(f"[gen{':'+tag if tag else ''}] {i}/{len(sources)}  ({time.perf_counter()-t0:.0f}s)")
    if skipped:
        print(f"[gen{':'+tag if tag else ''}] skipped {skipped} OOM source(s)")
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
