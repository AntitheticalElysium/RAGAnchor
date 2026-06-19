"""Post-generation NLI faithfulness gate (Cat 1). Score the answer's claims against the
context with the SummaC-ZS-style NLI scorer; if support is too low (or a claim is
contradicted), either abstain or retry-then-abstain.

Independent of the LettuceDetect *judge* on purpose — the gate uses NLI, the judge uses
ModernBERT, so the judge isn't grading the gate's own signal.

Honest accounting: latency_s covers the *whole* gate (generation + NLI + any retries),
not just the final generation — that's the real cost of the method."""

from __future__ import annotations

import time
from dataclasses import dataclass

from raganchor.config import SETTINGS, GateConfig
from raganchor.data import Source
from raganchor.eval.nli import NLIScorer
from raganchor.llm import GenerationResult
from raganchor.rag import VanillaRAG


@dataclass
class GatedOutput:
    answer: str  # final answer shipped (possibly the abstain text)
    contexts: list[str]
    gen: GenerationResult  # the final generation (token counts, ttft)
    nli_support: float  # NLI support score of the *generated* answer (pre-abstain)
    nli_contradiction: float
    is_abstained: bool
    n_retries: int
    latency_s: float  # end-to-end gate wall-clock


class NLIGate:
    def __init__(self, scorer: NLIScorer, cfg: GateConfig | None = None):
        self.scorer = scorer
        self.cfg = cfg or SETTINGS.gate

    def _fails(self, support: float, contradiction: float) -> bool:
        if support < self.cfg.support_threshold:
            return True
        return self.cfg.fail_on_contradiction and contradiction > 0

    def run(self, rag: VanillaRAG, source: Source) -> GatedOutput:
        t0 = time.perf_counter()
        out = rag.run(source)
        f = self.scorer.score(out.answer, out.contexts)

        retries = 0
        while (
            self._fails(f.score, f.contradiction_rate)
            and self.cfg.action == "retry"
            and retries < self.cfg.max_retries
        ):
            out = rag.run(source, do_sample=True, temperature=self.cfg.retry_temperature)
            f = self.scorer.score(out.answer, out.contexts)
            retries += 1

        abstained = self._fails(f.score, f.contradiction_rate)
        answer = self.cfg.abstain_text if abstained else out.answer
        return GatedOutput(
            answer=answer,
            contexts=out.contexts,
            gen=out.gen,
            nli_support=f.score,
            nli_contradiction=f.contradiction_rate,
            is_abstained=abstained,
            n_retries=retries,
            latency_s=time.perf_counter() - t0,
        )
