"""Tune the NLI gate's support threshold on RAGTruth *train* — not invented, not
test-leaked. For each train response we score NLI support of the answer vs its context
(SummaC-ZS-style) and have the human gold label. We pick the threshold T such that
"support < T" best detects hallucination (max F1), and report the curve around it.

Feed the printed T into experiments/ablate_gate.py --threshold.

  uv run python experiments/tune_gate.py --limit 2000

Run from the repo root."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from raganchor.config import SETTINGS
from raganchor.data import load_responses, load_sources


def prf1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=2000, help="0 = all (slow)")
    args = ap.parse_args()

    srcmap = load_sources()
    items = [
        (srcmap[r.source_id], r.response, len(r.labels) > 0)
        for r in load_responses(args.split)
        if r.source_id in srcmap
    ]
    random.Random(SETTINGS.seed).shuffle(items)
    if args.limit:
        items = items[: args.limit]
    print(f"[tune] {len(items)} responses | split={args.split}")
    print(f"[tune] gold hallucination rate = {sum(g for _,_,g in items)/len(items):.1%}")

    from raganchor.eval.nli import NLIScorer

    scorer = NLIScorer()
    scored: list[tuple[float, bool]] = []  # (support, gold_hallucinated)
    for i, (src, answer, gold) in enumerate(items, 1):
        scored.append((scorer.score(answer, src.passages).score, gold))
        if i % 200 == 0 or i == len(items):
            print(f"[nli] {i}/{len(items)}")

    print("\nthreshold  P     R     F1    (predict hallucinated if support < T)")
    best = (0.0, -1.0)
    for thr in [x / 100 for x in range(5, 100, 5)]:
        tp = sum(g and (s < thr) for s, g in scored)
        fp = sum((not g) and (s < thr) for s, g in scored)
        fn = sum(g and (s >= thr) for s, g in scored)
        p, r, f = prf1(tp, fp, fn)
        mark = ""
        if f > best[1]:
            best = (thr, f)
        print(f"  {thr:>4}   {p*100:5.1f} {r*100:5.1f} {f*100:5.1f}")
    print(f"\n[tune] best support_threshold = {best[0]} (F1={best[1]*100:.1f})")
    print(f"       -> uv run python experiments/ablate_gate.py --threshold {best[0]}")


if __name__ == "__main__":
    main()
