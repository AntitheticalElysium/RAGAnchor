"""Run-level aggregation. Real, raw axes only — faithfulness (provisional judge) +
production cost/latency (TTFT, end-to-end, tokens). No invented composite scores;
no dollar figures off a made-up rate. Further metrics get added when the ablation
shows they matter, not before."""

from __future__ import annotations

from statistics import mean, median


def aggregate(records: list[dict]) -> dict:
    if not records:
        return {}

    def avg(key: str) -> float:
        return mean(r[key] for r in records)

    return {
        "n": len(records),
        # faithfulness axis (LettuceDetect judge, response-level like RAGTruth)
        "faithful_rate": round(avg("is_faithful"), 4),  # frac of answers with no flagged span
        "mean_hallucination_spans": round(avg("n_hallucination_spans"), 3),
        # cost / latency axis
        "mean_ttft_s": round(avg("ttft_s"), 4),
        "median_ttft_s": round(median(r["ttft_s"] for r in records), 4),
        "mean_latency_s": round(avg("latency_s"), 4),
        "median_latency_s": round(median(r["latency_s"] for r in records), 4),
        "mean_prompt_tokens": round(avg("prompt_tokens"), 1),
        "mean_completion_tokens": round(avg("completion_tokens"), 1),
    }
