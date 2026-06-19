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
    TextIteratorStreamer,
)

from raganchor.config import SETTINGS, ModelConfig

_DTYPE = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    ttft_s: float  # time to first token
    latency_s: float  # end to end

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
