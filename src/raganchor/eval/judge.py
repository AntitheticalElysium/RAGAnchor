"""Faithfulness judge = LettuceDetect (ModernBERT token-classifier trained on
RAGTruth). It flags answer spans unsupported by the context; response is faithful
iff no span is flagged (RAGTruth's response-level convention).

Why this and not our NLI scorer: it's RAGTruth-SOTA among lean models (~79 example
F1, 396M, fits 6GB) AND it's independent of NLI — so if we later ablate an NLI-based
post-gen gate, the judge isn't grading its own family. See experiments/validate_judge.py.

  - score():        our own generations — pass retrieved contexts + question.
  - score_prompt(): RAGTruth reproduction — pass the raw RAGTruth prompt verbatim
                    (that's the exact format the model was trained on)."""

from __future__ import annotations

from dataclasses import dataclass

from raganchor.config import SETTINGS


@dataclass
class JudgeResult:
    is_faithful: bool  # no hallucinated span found
    n_spans: int  # number of unsupported spans
    spans: list[dict]  # [{start, end, confidence, text}, ...]


class FaithfulnessJudge:
    def __init__(self, model_path: str | None = None):
        from lettucedetect.models.inference import HallucinationDetector

        self.detector = HallucinationDetector(
            method="transformer", model_path=model_path or SETTINGS.judge.model_path
        )

    @staticmethod
    def _result(spans: list[dict]) -> JudgeResult:
        return JudgeResult(is_faithful=not spans, n_spans=len(spans), spans=spans)

    def score(self, answer: str, contexts: list[str], question: str | None = None) -> JudgeResult:
        spans = self.detector.predict(
            context=contexts, answer=answer, question=question, output_format="spans"
        )
        return self._result(spans)

    def score_prompt(self, answer: str, prompt: str) -> JudgeResult:
        spans = self.detector.predict_prompt(prompt=prompt, answer=answer, output_format="spans")
        return self._result(spans)
