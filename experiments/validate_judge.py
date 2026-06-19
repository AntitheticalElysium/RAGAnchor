"""Validate the faithfulness judge against RAGTruth's human labels — the foundation
the whole ablation rests on. No generation here: we run the judge on RAGTruth's own
labeled responses and check it agrees with the humans.

Gold (response-level, RAGTruth convention): response is hallucinated iff it has >=1
labeled span. Prediction: judge flags >=1 unsupported span. We report precision/recall/
F1 for the *hallucinated* class, per task and overall.

Target to reproduce: LettuceDetect-large ~79 example-level F1 (paper Table 2). Hitting
that confirms both the judge and our data pipeline (prompt assembly, label derivation).

  uv run python experiments/validate_judge.py                 # judge only, full test split
  uv run python experiments/validate_judge.py --limit 300     # quick
  uv run python experiments/validate_judge.py --with-nli      # also benchmark the old NLI scorer

Run from the repo root."""

from __future__ import annotations

import argparse
import gc
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from raganchor.config import SETTINGS
from raganchor.data import load_responses, load_sources


def prf1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def _counts(pairs: list[tuple[bool, bool]]) -> dict:
    """pairs = (gold_hallucinated, pred_hallucinated). Positive class = hallucinated."""
    tp = sum(g and p for g, p in pairs)
    fp = sum((not g) and p for g, p in pairs)
    fn = sum(g and (not p) for g, p in pairs)
    tn = sum((not g) and (not p) for g, p in pairs)
    p, r, f = prf1(tp, fp, fn)
    return {"n": len(pairs), "P": round(p * 100, 1), "R": round(r * 100, 1),
            "F1": round(f * 100, 1), "acc": round((tp + tn) / len(pairs) * 100, 1)}


def _report(title: str, by_task: dict[str, list[tuple[bool, bool]]]) -> None:
    print(f"\n=== {title} (positive class = hallucinated) ===")
    print(f"{'task':<12}{'n':>6}{'P':>8}{'R':>8}{'F1':>8}{'acc':>8}")
    allpairs: list[tuple[bool, bool]] = []
    for task in ("QA", "Summary", "Data2txt"):
        if by_task.get(task):
            m = _counts(by_task[task])
            print(f"{task:<12}{m['n']:>6}{m['P']:>8}{m['R']:>8}{m['F1']:>8}{m['acc']:>8}")
            allpairs += by_task[task]
    m = _counts(allpairs)
    print(f"{'OVERALL':<12}{m['n']:>6}{m['P']:>8}{m['R']:>8}{m['F1']:>8}{m['acc']:>8}")


def collect(split: str, limit: int | None):
    """(source, response_text, gold_hallucinated) for every response in the split.
    The file is ordered by task, so shuffle (fixed seed) before limiting — otherwise a
    --limit slice is single-task and unrepresentative."""
    srcmap = load_sources()
    items = [
        (srcmap[r.source_id], r.response, len(r.labels) > 0)
        for r in load_responses(split)
        if r.source_id in srcmap
    ]
    random.Random(SETTINGS.seed).shuffle(items)
    return items[:limit] if limit else items


def eval_judge(items) -> dict[str, list[tuple[bool, bool]]]:
    from raganchor.eval.judge import FaithfulnessJudge

    judge = FaithfulnessJudge()
    by_task: dict[str, list[tuple[bool, bool]]] = {}
    for i, (src, answer, gold) in enumerate(items, 1):
        res = judge.score_prompt(answer, src.prompt)  # raw prompt = training format
        by_task.setdefault(src.task_type, []).append((gold, not res.is_faithful))
        if i % 200 == 0 or i == len(items):
            print(f"[judge] {i}/{len(items)}")
    del judge
    gc.collect()
    torch.cuda.empty_cache()
    return by_task


def eval_nli_best(items) -> dict[str, list[tuple[bool, bool]]]:
    """Old NLI scorer at its *oracle* best-F1 threshold — its best case, for a fair
    upper bound. If even this loses to the judge, the switch is justified."""
    from raganchor.eval.nli import NLIScorer

    scorer = NLIScorer()
    scored: list[tuple[str, float, bool]] = []  # (task, faithfulness, gold)
    for i, (src, answer, gold) in enumerate(items, 1):
        f = scorer.score(answer, src.passages)
        scored.append((src.task_type, f.score, gold))
        if i % 200 == 0 or i == len(items):
            print(f"[nli] {i}/{len(items)}")
    del scorer
    gc.collect()
    torch.cuda.empty_cache()

    best_thr, best_f1 = 0.5, -1.0
    for thr in [x / 100 for x in range(5, 100, 5)]:
        pairs = [(g, fs < thr) for _, fs, g in scored]
        _, _, f1 = prf1(
            sum(g and p for g, p in pairs),
            sum((not g) and p for g, p in pairs),
            sum(g and (not p) for g, p in pairs),
        )
        if f1 > best_f1:
            best_thr, best_f1 = thr, f1
    print(f"[nli] oracle threshold = faithfulness < {best_thr}")
    by_task: dict[str, list[tuple[bool, bool]]] = {}
    for task, fs, g in scored:
        by_task.setdefault(task, []).append((g, fs < best_thr))
    return by_task


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--with-nli", action="store_true", help="also benchmark the old NLI scorer")
    args = ap.parse_args()

    items = collect(args.split, args.limit or None)
    print(f"[validate] {len(items)} responses | split={args.split}")
    gold_rate = sum(g for _, _, g in items) / len(items)
    print(f"[validate] gold hallucination rate = {gold_rate:.1%}")

    _report("LettuceDetect judge", eval_judge(items))
    if args.with_nli:
        _report("Old NLI scorer (oracle threshold)", eval_nli_best(items))


if __name__ == "__main__":
    main()
