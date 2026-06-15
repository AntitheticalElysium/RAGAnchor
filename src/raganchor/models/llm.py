"""Local 4-bit model. Returns text + token counts + latency for the cost metrics.
`logits_processors` is threaded through now so Phase-2 decode methods drop in later."""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from raganchor.config import SETTINGS, ModelConfig

_DTYPE = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float

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

    @torch.inference_mode()
    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_new_tokens: int | None = None,
        logits_processors=None,
    ) -> GenerationResult:
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        prompt_tokens = int(inputs["input_ids"].shape[1])

        gen_kwargs: dict = dict(
            max_new_tokens=max_new_tokens or self.cfg.max_new_tokens,
            do_sample=self.cfg.do_sample,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        if self.cfg.do_sample:
            gen_kwargs["temperature"] = self.cfg.temperature
        if logits_processors is not None:
            gen_kwargs["logits_processor"] = logits_processors

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = self.model.generate(**inputs, **gen_kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latency = time.perf_counter() - t0

        new_tokens = out[0][prompt_tokens:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return GenerationResult(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=int(new_tokens.shape[0]),
            latency_s=latency,
        )
