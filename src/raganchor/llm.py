"""Local 4-bit model. Returns text + token counts + latency for the cost metrics.
We stream so TTFT (time to first token) is a real measurement, not a guess.
`logits_processors` is threaded through now so Phase-2 decode methods drop in later."""

from __future__ import annotations

import time
from dataclasses import dataclass
from threading import Thread

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    LogitsProcessor,
    TextIteratorStreamer,
)

from raganchor.config import SETTINGS, ModelConfig

_DTYPE = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def _jsd_base2(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Jensen-Shannon divergence (log base 2) of two prob vectors -> scalar in [0, 1]."""
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    m = 0.5 * (p + q)
    kl = lambda a: torch.sum(a * (torch.log2(a) - torch.log2(m)))
    return (0.5 * (kl(p) + kl(q))).clamp(0.0, 1.0)


class CADLogitsProcessor(LogitsProcessor):
    """Context-Aware Decoding (Shi et al. 2023) and AdaCAD (Wang et al. 2024). Run a batch
    of 2 in lockstep — row 0 = with-context, row 1 = without-context — and combine each step:
        adjusted = (1 + alpha) * logit_with - alpha * logit_without
    Both rows get the combined distribution so greedy stays synced.

    Static CAD uses a fixed alpha (which over-corrects when context and prior don't
    conflict). AdaCAD sets alpha per token = JSD(p_with || p_without): ~0 when they agree
    (no over-correction), high when they conflict. warmup_lambda floors alpha for long-form.
    Also records the first-step time so TTFT is measurable without a streamer."""

    def __init__(self, alpha: float | None = None, adaptive: bool = False, warmup_lambda: float = 0.3):
        self.alpha = alpha
        self.adaptive = adaptive
        self.warmup_lambda = warmup_lambda
        self.t_first: float | None = None
        self.alphas: list[float] = []  # per-step alpha (for inspecting AdaCAD behaviour)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.t_first is None:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.t_first = time.perf_counter()
        with_ctx, without_ctx = scores[0], scores[1]
        if self.adaptive:
            p = torch.softmax(with_ctx, dim=-1)
            q = torch.softmax(without_ctx, dim=-1)
            alpha = float(max(_jsd_base2(p, q).item(), self.warmup_lambda))
        else:
            alpha = self.alpha
        self.alphas.append(alpha)
        combined = (1 + alpha) * with_ctx - alpha * without_ctx
        scores[0] = combined
        scores[1] = combined
        return scores


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    ttft_s: float  # time to first token
    latency_s: float  # end to end
    mean_alpha: float | None = None  # CAD/AdaCAD: avg per-token alpha actually applied

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LocalLLM:
    """A 4-bit instruction model held resident on the GPU."""

    def __init__(self, cfg: ModelConfig | None = None):
        self.cfg = cfg or SETTINGS.model
        quant = None
        if self.cfg.load_in_4bit:
            quant = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=self.cfg.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=self.cfg.bnb_4bit_use_double_quant,
                bnb_4bit_compute_dtype=_DTYPE[self.cfg.bnb_4bit_compute_dtype],
            )
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_id,
            quantization_config=quant,
            device_map="cuda" if torch.cuda.is_available() else "cpu",
            dtype=_DTYPE[self.cfg.bnb_4bit_compute_dtype],
        )
        self.model.eval()

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_new_tokens: int | None = None,
        do_sample: bool | None = None,  # override cfg (e.g. sampling on gate retry)
        temperature: float | None = None,
        logits_processors=None,
    ) -> GenerationResult:
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        prompt_tokens = int(inputs["input_ids"].shape[1])

        sample = self.cfg.do_sample if do_sample is None else do_sample
        gen_kwargs: dict = dict(
            max_new_tokens=max_new_tokens or self.cfg.max_new_tokens,
            do_sample=sample,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        if sample:
            gen_kwargs["temperature"] = self.cfg.temperature if temperature is None else temperature
        if logits_processors is not None:
            gen_kwargs["logits_processor"] = logits_processors

        # Stream on a worker thread; the main thread times the first chunk (TTFT)
        # and the last (end-to-end), and reassembles the text.
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        @torch.inference_mode()
        def _run() -> None:
            self.model.generate(**inputs, **gen_kwargs, streamer=streamer)

        thread = Thread(target=_run)
        thread.start()

        chunks: list[str] = []
        ttft = 0.0
        for chunk in streamer:
            if not chunks:
                ttft = time.perf_counter() - t0
            chunks.append(chunk)
        thread.join()
        latency = time.perf_counter() - t0

        text = "".join(chunks).strip()
        completion_tokens = len(self.tokenizer(text, add_special_tokens=False)["input_ids"])
        return GenerationResult(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            ttft_s=round(ttft, 4),
            latency_s=round(latency, 4),
        )

    @torch.inference_mode()
    def generate_cad(
        self,
        full_messages: list[dict[str, str]],
        nocontext_messages: list[dict[str, str]],
        *,
        alpha: float | None = None,
        adaptive: bool = False,
        warmup_lambda: float = 0.3,
        max_new_tokens: int | None = None,
    ) -> GenerationResult:
        """Context-Aware Decoding: contrast with-context vs without-context logits.
        Two forward passes per step (batch of 2) => ~2x the decode cost of plain greedy.
        adaptive=True => AdaCAD (per-token alpha from JSD)."""
        full = self.tokenizer.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=True
        )
        noctx = self.tokenizer.apply_chat_template(
            nocontext_messages, tokenize=False, add_generation_prompt=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"  # required for batched generation
        enc = self.tokenizer([full, noctx], return_tensors="pt", padding=True).to(self.model.device)
        prompt_len = enc["input_ids"].shape[1]
        prompt_tokens = int(enc["attention_mask"][0].sum())  # row 0 = with-context

        proc = CADLogitsProcessor(alpha=alpha, adaptive=adaptive, warmup_lambda=warmup_lambda)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = self.model.generate(
            **enc,
            max_new_tokens=max_new_tokens or self.cfg.max_new_tokens,
            do_sample=False,
            logits_processor=[proc],
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latency = time.perf_counter() - t0

        new_tokens = out[0][prompt_len:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        completion_tokens = len(self.tokenizer(text, add_special_tokens=False)["input_ids"])
        return GenerationResult(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            ttft_s=round((proc.t_first - t0) if proc.t_first else 0.0, 4),
            latency_s=round(latency, 4),
            mean_alpha=round(sum(proc.alphas) / len(proc.alphas), 3) if proc.alphas else None,
        )
