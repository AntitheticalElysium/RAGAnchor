"""Claim-level NLI faithfulness. Split answer into claims, check each against context
windows; supported = max entailment prob over windows >= threshold. Reports support and
contradiction rates separately (a claim with no support != a contradicted claim)."""

from __future__ import annotations

import re
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from raganchor.config import SETTINGS, NLIConfig

_SENT = re.compile(r"(?<=[.!?])\s+")


def split_claims(text: str) -> list[str]:
    return [s.strip() for s in _SENT.split(text.strip()) if len(s.strip()) > 3]


def _windows(context: str, size: int = 2, stride: int = 1, cap: int = 48) -> list[str]:
    sents = [s.strip() for s in _SENT.split(context.strip()) if s.strip()]
    if not sents:
        return [context.strip()] if context.strip() else []
    out = [" ".join(sents[i : i + size]) for i in range(0, len(sents), stride)]
    return out[:cap]


@dataclass
class FaithfulnessResult:
    score: float  # fraction of claims supported
    n_claims: int
    n_supported: int
    contradiction_rate: float
    claim_entailments: list[float]


class NLIScorer:
    def __init__(self, cfg: NLIConfig | None = None, device: str | None = None):
        self.cfg = cfg or SETTINGS.nli
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.cfg.model_id)
        self.model.to(self.device).eval()
        label2id = {k.lower(): v for k, v in self.model.config.label2id.items()}
        self.ent_idx = label2id["entailment"]
        self.con_idx = label2id["contradiction"]

    @torch.inference_mode()
    def _entail_contra(self, premises: list[str], hypotheses: list[str]) -> torch.Tensor:
        probs: list[torch.Tensor] = []
        for i in range(0, len(premises), 32):
            enc = self.tokenizer(
                premises[i : i + 32],
                hypotheses[i : i + 32],
                truncation=True,
                max_length=512,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            logits = self.model(**enc).logits
            probs.append(F.softmax(logits, dim=-1).cpu())
        p = torch.cat(probs)
        return p[:, [self.ent_idx, self.con_idx]]

    def score(self, answer: str, contexts: list[str]) -> FaithfulnessResult:
        claims = split_claims(answer)
        windows = _windows("\n".join(contexts))
        if not claims or not windows:
            return FaithfulnessResult(1.0, 0, 0, 0.0, [])

        premises, hypotheses = [], []
        for claim in claims:
            for w in windows:
                premises.append(w)
                hypotheses.append(claim)
        ec = self._entail_contra(premises, hypotheses).view(len(claims), len(windows), 2)

        max_ent = ec[:, :, 0].max(dim=1).values
        max_con = ec[:, :, 1].max(dim=1).values
        supported = (max_ent >= self.cfg.entailment_threshold)
        # contradicted if best window contradicts more than it entails
        contradicted = (max_con >= self.cfg.entailment_threshold) & (max_con > max_ent)
        return FaithfulnessResult(
            score=float(supported.float().mean()),
            n_claims=len(claims),
            n_supported=int(supported.sum()),
            contradiction_rate=float(contradicted.float().mean()),
            claim_entailments=[round(x, 4) for x in max_ent.tolist()],
        )
