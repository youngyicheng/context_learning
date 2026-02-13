"""Solver model (Student) wrapper for generating answers from context."""

from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class SolverModel:
    """Solver model that answers questions based on provided context."""

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
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        do_sample: bool = True,
        return_logprobs: bool = False,
        **kwargs,
    ):
        """
        Generate answer given context and question (in messages).

        Args:
            messages: OpenAI chat format [{"role": "system", "content": "..."}, ...].
            max_new_tokens: Max tokens to generate.
            temperature: Sampling temperature.
            do_sample: Whether to sample.
            return_logprobs: Whether to return token log-probs (for RL).
            **kwargs: Additional generation kwargs.

        Returns:
            If return_logprobs: (response_text, logprobs_tensor).
            Else: response_text.
        """
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
            **kwargs,
        }
        if return_logprobs:
            gen_kwargs["output_scores"] = True
            gen_kwargs["return_dict_in_generate"] = True

        with torch.set_grad_enabled(return_logprobs):
            outputs = self.model.generate(**inputs, **gen_kwargs)

        input_len = inputs["input_ids"].shape[1]
        if return_logprobs:
            gen_ids = outputs.sequences[0][input_len:]
            response = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            logprobs = self._get_sequence_logprobs(outputs, input_len)
            return response.strip(), logprobs
        else:
            response = self.tokenizer.decode(
                outputs[0][input_len:], skip_special_tokens=True
            )
            return response.strip()

    def _get_sequence_logprobs(self, outputs, input_len: int) -> torch.Tensor:
        """Compute per-token log-probs for generated sequence (for REINFORCE)."""
        scores = outputs.scores
        if not scores:
            return torch.tensor(0.0, device=self.model.device)
        gen_ids = outputs.sequences[0][input_len:]
        logprobs = []
        for t, token_id in enumerate(gen_ids):
            if t < len(scores):
                logits = scores[t][0]
                logp = torch.log_softmax(logits.float(), dim=-1)
                lp = logp[token_id].item()
                logprobs.append(lp)
        return torch.tensor(logprobs, device=self.model.device)
