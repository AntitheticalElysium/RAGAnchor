# RAGAnchor

An ablation of RAG faithfulness/efficiency methods on a small local model, then a
minimal agentic RAG built from *only* the methods that earn their place.

The thesis to test (not assume): a small local model (~3B, 4-bit) + the right subset
of these methods can approach a large model's **faithfulness** on grounded QA at a
fraction of the cost. This is an integration + measurement project, not a new algorithm.

## Stages (strict order)

1. **Ablation** — measure each method's tradeoff with simple scripts that output
   numbers. No services, no APIs, no UI. ← *we are here*
2. **Final RAG** — one clean agentic RAG assembled from the Stage-1 winners.
3. **Deferred** — any productionization, scoped only once Stage 2 works.

See [BUILD.md](BUILD.md) and [CLAUDE.md](CLAUDE.md) for scope and the method catalog.

## Layout

```
src/raganchor/
  config.py      knobs (pydantic Settings)
  data.py        RAGTruth loader
  llm.py         local 4-bit model (TTFT + latency + token counts)
  retrieval.py   hybrid dense + BM25, fused with RRF
  rag.py         vanilla RAG baseline: retrieve -> ground -> generate
  eval/
    nli.py       automatic faithfulness judge (PROVISIONAL — validate vs RAGTruth labels)
    metrics.py   run-level aggregation (raw axes only, no invented composites)
    harness.py   generate -> free GPU -> score -> write run JSON
scripts/         data download, etc.
```

Components are the reusable building blocks; the ablation is thin scripts that toggle
them. The final RAG (Stage 2) will be one more script that composes the winners — no
separate "production" tree.

## Setup

```sh
uv sync
uv run python scripts/download_ragtruth.py   # -> data/ragtruth/
uv run raganchor smoke                        # load model, generate once
uv run raganchor baseline --tasks QA --limit 60
```

## Metrics

Two axes, reported as **raw per-config numbers** — no headline/composite scores:

- **Faithfulness** — an automatic judge (NLI-based for now). Provisional: its
  thresholds are validated against RAGTruth's human hallucination labels before any
  number is trusted.
- **Cost / latency** — TTFT, end-to-end latency, prompt/completion tokens.

Further metrics are added when the ablation shows they matter, not before.

## Benchmark

[RAGTruth](https://arxiv.org/pdf/2401.00396) — span-level hallucination labels across
QA / summarization / data-to-text.
