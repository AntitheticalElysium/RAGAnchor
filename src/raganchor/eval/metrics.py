"""Cost accounting + run-level aggregation for the Pareto frontier."""

from __future__ import annotations

from statistics import mean

from raganchor.config import SETTINGS


def local_cost_usd(latency_s: float) -> float:
    return latency_s / 3600.0 * SETTINGS.cost.local_gpu_usd_per_hour


def aggregate(records: list[dict]) -> dict:
    if not records:
        return {}
    faith = [r["faithfulness"] for r in records]
    gate = SETTINGS.nli.faithfulness_gate
    total_cost = sum(r["cost_usd"] for r in records)
    mean_faith = mean(faith)
    return {
        "n": len(records),
        "mean_faithfulness": round(mean_faith, 4),
        "unsupported_claim_rate": round(1 - mean_faith, 4),
        "unfaithful_answer_rate": round(mean(f < gate for f in faith), 4),
        "mean_contradiction_rate": round(mean(r["contradiction_rate"] for r in records), 4),
        "mean_latency_s": round(mean(r["latency_s"] for r in records), 3),
        "mean_completion_tokens": round(mean(r["completion_tokens"] for r in records), 1),
        "usd_per_query": round(total_cost / len(records), 6),
        "faithfulness_per_dollar": round(mean_faith / (total_cost / len(records)), 1)
        if total_cost
        else None,
    }
