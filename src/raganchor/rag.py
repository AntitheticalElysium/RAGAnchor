"""Vanilla RAG: retrieve -> grounding prompt -> generate. The Phase-0 baseline."""

from __future__ import annotations

from dataclasses import dataclass

from raganchor.data import Source
from raganchor.llm import GenerationResult, LocalLLM
from raganchor.retrieval import ContextPruner, HybridRetriever, Reranker

GROUNDING_SYSTEM = (
    "You are a careful assistant. Use only the provided context. "
    "If the context does not contain the answer, say \"I don't know.\" "
    "Do not add outside information."
)


@dataclass
class RAGOutput:
    answer: str
    contexts: list[str]  # what faithfulness is scored against
    gen: GenerationResult


def _build_user_prompt(task_type: str, context: str, question: str | None) -> str:
    if task_type == "QA":
        return f"Context:\n{context}\n\nQuestion: {question}\nAnswer using only the context above."
    if task_type == "Data2txt":
        return f"Write an objective overview of this business using only the structured data:\n\n{context}"
    return f"Summarize the following, using only its content:\n\n{context}"


def _build_nocontext_prompt(task_type: str, question: str | None) -> str:
    """The CAD 'without-context' input: same instruction, context removed."""
    if task_type == "QA":
        return f"Question: {question}\nAnswer."
    if task_type == "Data2txt":
        return "Write an objective overview of this business."
    return "Write a summary."


class VanillaRAG:
    def __init__(
        self,
        llm: LocalLLM,
        retriever: HybridRetriever | None = None,
        reranker: Reranker | None = None,
        rerank_keep: int | None = None,
        pruner: ContextPruner | None = None,
        prune_threshold: float | None = None,
        cad_alpha: float | None = None,
        cad_adaptive: bool = False,
        cad_warmup: float = 0.3,
    ):
        self.llm = llm
        self.retriever = retriever or HybridRetriever()
        self.reranker = reranker  # None => no reranking
        self.rerank_keep = rerank_keep  # passages kept after reranking (None = all)
        self.pruner = pruner  # None => no context pruning
        self.prune_threshold = prune_threshold
        # CAD: cad_adaptive => AdaCAD (per-token alpha); else static alpha; else off
        self.cad_alpha = cad_alpha
        self.cad_adaptive = cad_adaptive
        self.cad_warmup = cad_warmup  # AdaCAD warmup floor on alpha (0 = pure JSD)

    def run(self, source: Source, **gen_overrides) -> RAGOutput:
        if source.task_type == "QA" and source.question:
            self.retriever.index(source.passages)
            hits = self.retriever.retrieve(source.question)
            if self.reranker is not None:
                hits = self.reranker.rerank(source.question, hits, keep=self.rerank_keep)
            if self.pruner is not None:
                contexts = [self.pruner.prune(
                    source.question, [h.text for h in hits], threshold=self.prune_threshold
                )]
            else:
                contexts = [h.text for h in hits]
        else:
            contexts = [source.context]  # single source doc — retrieval is identity

        context_block = "\n\n".join(contexts)
        messages = [
            {"role": "system", "content": GROUNDING_SYSTEM},
            {"role": "user", "content": _build_user_prompt(source.task_type, context_block, source.question)},
        ]
        if self.cad_alpha is not None or self.cad_adaptive:
            nocontext = [
                {"role": "system", "content": GROUNDING_SYSTEM},
                {"role": "user", "content": _build_nocontext_prompt(source.task_type, source.question)},
            ]
            gen = self.llm.generate_cad(
                messages, nocontext, alpha=self.cad_alpha,
                adaptive=self.cad_adaptive, warmup_lambda=self.cad_warmup,
            )
        else:
            gen = self.llm.generate(messages, **gen_overrides)
        return RAGOutput(answer=gen.text, contexts=contexts, gen=gen)
