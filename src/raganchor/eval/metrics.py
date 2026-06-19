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

    out = {
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

    # gate-only: abstaining inflates faithful_rate (the judge scores "I don't know" as
    # faithful), so also report abstention and faithfulness among *answered* responses.
    if records[0].get("is_abstained") is not None:
        answered = [r for r in records if not r["is_abstained"]]
        out["abstention_rate"] = round(avg("is_abstained"), 4)
        out["mean_retries"] = round(avg("n_retries"), 3)
        out["faithful_rate_answered"] = (
            round(mean(r["is_faithful"] for r in answered), 4) if answered else None
        )
    return out
