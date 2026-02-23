"""Solver model (Student) wrapper for generating answers from context."""

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
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
        **kwargs,
    ) -> str:
        """
        Generate answer (inference only, no gradient).

        Args:
            messages: OpenAI chat format [{"role": "system", "content": "..."}, ...].
            max_new_tokens: Max tokens to generate.
            temperature: Sampling temperature.
            do_sample: Whether to sample.

        Returns:
            Generated response text.
        """
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
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
        response = self.tokenizer.decode(
            outputs[0][input_len:], skip_special_tokens=True
        )
        return response.strip()

    def generate_for_rl(
        self,
        messages: List[dict],
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        do_sample: bool = True,
        **kwargs,
    ) -> Tuple[str, torch.Tensor, torch.Tensor]:
        """
        Generate answer and return token IDs for subsequent differentiable
        log-prob computation (REINFORCE). Generation itself is non-differentiable.

        Args:
            messages: OpenAI chat format messages.
            max_new_tokens: Max tokens to generate.
            temperature: Sampling temperature.
            do_sample: Whether to sample.

        Returns:
            (response_text, input_ids [1, prompt_len], generated_ids [1, gen_len])
        """
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_ids = inputs["input_ids"]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                **kwargs,
            )

        input_len = input_ids.shape[1]
        generated_ids = outputs[:1, input_len:]
        response = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return response.strip(), input_ids.detach(), generated_ids.detach()

    def compute_sequence_logprob(
        self,
        input_ids: torch.Tensor,
        generated_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Differentiable forward pass to compute mean log-probability of the
        generated sequence, suitable for REINFORCE policy gradient.

        Args:
            input_ids: Prompt token IDs, shape [1, prompt_len].
            generated_ids: Generated token IDs, shape [1, gen_len].

        Returns:
            Scalar tensor: mean log P(generated | prompt), with gradient.
        """
        if generated_ids.numel() == 0:
            return torch.tensor(0.0, device=self.model.device, requires_grad=True)

        full_ids = torch.cat([input_ids, generated_ids], dim=1)
        outputs = self.model(full_ids)
        logits = outputs.logits.float()

        input_len = input_ids.shape[1]
        shift_logits = logits[:, input_len - 1:-1, :]
        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(2, generated_ids.unsqueeze(-1)).squeeze(-1)
        return token_log_probs.mean(dim=-1).squeeze(0)
