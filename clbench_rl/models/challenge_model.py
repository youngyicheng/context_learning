"""Challenge model (Teacher) wrapper for generating context and questions."""

from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class ChallengeModel:
    """Challenge model that provides context and questions to the Solver."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: Optional[str] = None,
        use_fast: bool = True,
    ):
        """
        Args:
            model_name: HuggingFace model identifier.
            device: Device to load model on (None = auto).
            use_fast: Use fast tokenizer.
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, use_fast=use_fast
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
        )
        if self.device == "cpu":
            self.model = self.model.to(self.device)

    def generate(
        self,
        messages: List[dict],
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        do_sample: bool = True,
        **kwargs,
    ) -> str:
        """
        Generate context + question from input messages.

        By default, uses raw messages as-is. Can be extended to augment/rewrite.

        Args:
            messages: Input in OpenAI chat format [{"role": "...", "content": "..."}].
            max_new_tokens: Max tokens to generate.
            temperature: Sampling temperature.
            do_sample: Whether to sample (vs greedy).
            **kwargs: Additional generation kwargs.

        Returns:
            Generated text (or pass-through of user content if no generation).
        """
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                **kwargs,
            )

        input_len = inputs["input_ids"].shape[1]
        generated = self.tokenizer.decode(
            outputs[0][input_len:], skip_special_tokens=True
        )
        return generated.strip()

    def format_prompt(self, messages: List[dict]) -> List[dict]:
        """
        Format messages for Solver. By default returns as-is.
        Override to add hints, restructure, etc.
        """
        return messages
