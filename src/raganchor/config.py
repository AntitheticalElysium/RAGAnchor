"""Central config — one place to turn the knobs for the "what sticks" study."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RUNS_DIR = REPO_ROOT / "runs"
RAGTRUTH_DIR = DATA_DIR / "ragtruth"


class ModelConfig(BaseModel):
    model_id: str = "Qwen/Qwen2.5-3B-Instruct"
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"  # bf16 matmuls on Ampere
    max_new_tokens: int = 256
    do_sample: bool = False  # greedy — grounding wants determinism
    temperature: float = 0.0


class RetrievalConfig(BaseModel):
    # bge-large-en-v1.5: 335M, 1024-dim, top MTEB. A/B later vs gte-large / mxbai-embed-large.
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    query_instruction: str = "Represent this sentence for searching relevant passages: "
    normalize_embeddings: bool = True
    top_k: int = 4  # moderate context keeps the KV cache small
    rrf_k: int = 60
    use_bm25: bool = True
    use_dense: bool = True
    reranker_model: str = "BAAI/bge-reranker-v2-m3"  # 0.6B cross-encoder, sigmoid relevance


class JudgeConfig(BaseModel):
    # Faithfulness judge: LettuceDetect (ModernBERT token-classifier trained on
    # RAGTruth). Independent of NLI, so no circularity if we later ablate an NLI gate.
    # large = 396M, 79.2 example-F1 on RAGTruth; base = 150M, 76.1.
    model_path: str = "KRLabsOrg/lettucedect-large-modernbert-en-v1"


class NLIConfig(BaseModel):
    # NOT the judge anymore — kept for the post-gen NLI-gate *method* to ablate later.
    # entailment_threshold stays PROVISIONAL until that method is read + implemented.
    model_id: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    entailment_threshold: float = 0.5  # claim supported if max entailment over context >= this


class Settings(BaseModel):
    model: ModelConfig = Field(default_factory=ModelConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    nli: NLIConfig = Field(default_factory=NLIConfig)
    seed: int = 0


SETTINGS = Settings()
