"""Vanilla RAG: retrieve -> grounding prompt -> generate. The Phase-0 baseline."""

from __future__ import annotations

from dataclasses import dataclass

from raganchor.data import Source
from raganchor.llm import GenerationResult, LocalLLM
from raganchor.retrieval import HybridRetriever, Reranker

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


class VanillaRAG:
    def __init__(
        self,
        llm: LocalLLM,
        retriever: HybridRetriever | None = None,
        reranker: Reranker | None = None,
        rerank_keep: int | None = None,
    ):
        self.llm = llm
        self.retriever = retriever or HybridRetriever()
        self.reranker = reranker  # None => no reranking
        self.rerank_keep = rerank_keep  # passages kept after reranking (None = all)

    def run(self, source: Source) -> RAGOutput:
        if source.task_type == "QA" and source.question:
            self.retriever.index(source.passages)
            hits = self.retriever.retrieve(source.question)
            if self.reranker is not None:
                hits = self.reranker.rerank(source.question, hits, keep=self.rerank_keep)
            contexts = [h.text for h in hits]
        else:
            contexts = [source.context]  # single source doc — retrieval is identity

        context_block = "\n\n".join(contexts)
        messages = [
            {"role": "system", "content": GROUNDING_SYSTEM},
            {"role": "user", "content": _build_user_prompt(source.task_type, context_block, source.question)},
        ]
        gen = self.llm.generate(messages)
        return RAGOutput(answer=gen.text, contexts=contexts, gen=gen)
