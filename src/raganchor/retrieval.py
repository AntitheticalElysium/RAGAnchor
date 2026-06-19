"""Hybrid retrieval: dense (bge-large) + BM25, fused with Reciprocal Rank Fusion.
Optional cross-encoder reranking (bge-reranker) on top, used here for rerank-then-prune."""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from raganchor.config import SETTINGS, RetrievalConfig

_TOKEN = re.compile(r"\w+")


def _tok(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


@dataclass
class Hit:
    index: int
    text: str
    score: float


class HybridRetriever:
    def __init__(self, cfg: RetrievalConfig | None = None):
        self.cfg = cfg or SETTINGS.retrieval
        self._embedder: SentenceTransformer | None = None
        self._passages: list[str] = []
        self._bm25: BM25Okapi | None = None
        self._emb: np.ndarray | None = None

    @property
    def embedder(self) -> SentenceTransformer:
        if self._embedder is None:
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            self._embedder = SentenceTransformer(
                self.cfg.embedding_model, model_kwargs={"torch_dtype": dtype}
            )
        return self._embedder

    def index(self, passages: list[str]) -> None:
        self._passages = passages
        if self.cfg.use_bm25:
            self._bm25 = BM25Okapi([_tok(p) for p in passages])
        if self.cfg.use_dense:
            self._emb = self.embedder.encode(
                passages,
                normalize_embeddings=self.cfg.normalize_embeddings,
                convert_to_numpy=True,
                show_progress_bar=False,
            )

    def _dense_ranks(self, query: str) -> list[int]:
        q = self.embedder.encode(
            [self.cfg.query_instruction + query],
            normalize_embeddings=self.cfg.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0]
        scores = self._emb @ q
        return list(np.argsort(-scores))

    def _bm25_ranks(self, query: str) -> list[int]:
        scores = self._bm25.get_scores(_tok(query))
        return list(np.argsort(-scores))

    def retrieve(self, query: str, top_k: int | None = None) -> list[Hit]:
        k = top_k or self.cfg.top_k
        n = len(self._passages)
        if n == 0:
            return []
        rankings: list[list[int]] = []
        if self.cfg.use_dense and self._emb is not None:
            rankings.append(self._dense_ranks(query))
        if self.cfg.use_bm25 and self._bm25 is not None:
            rankings.append(self._bm25_ranks(query))

        fused: dict[int, float] = {i: 0.0 for i in range(n)}
        for ranking in rankings:
            for rank, idx in enumerate(ranking):
                fused[int(idx)] += 1.0 / (self.cfg.rrf_k + rank)

        ordered = sorted(fused.items(), key=lambda kv: -kv[1])[:k]
        return [Hit(index=i, text=self._passages[i], score=s) for i, s in ordered]


class Reranker:
    """Cross-encoder reranker (bge-reranker-v2-m3). Scores each (query, passage) pair
    directly, then we keep the top `keep`. On RAGTruth QA (3 passages) this prunes."""

    def __init__(self, cfg: RetrievalConfig | None = None):
        self.cfg = cfg or SETTINGS.retrieval
        self._model: AutoModelForSequenceClassification | None = None
        self._tok: AutoTokenizer | None = None

    def _load(self) -> None:
        if self._model is not None:
            return
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self._tok = AutoTokenizer.from_pretrained(self.cfg.reranker_model)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.cfg.reranker_model, torch_dtype=dtype
        )
        self._model.to("cuda" if torch.cuda.is_available() else "cpu").eval()

    @torch.inference_mode()
    def rerank(self, query: str, hits: list[Hit], keep: int | None = None) -> list[Hit]:
        """Reorder hits by cross-encoder relevance; keep top `keep` (None = keep all)."""
        if not hits:
            return hits
        self._load()
        enc = self._tok(
            [query] * len(hits),
            [h.text for h in hits],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self._model.device)
        logits = self._model(**enc).logits.squeeze(-1).float()
        scores = torch.sigmoid(logits).cpu().tolist()
        reranked = sorted(
            (Hit(h.index, h.text, s) for h, s in zip(hits, scores)),
            key=lambda h: -h.score,
        )
        return reranked[:keep] if keep else reranked
