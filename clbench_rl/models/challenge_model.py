"""Challenge model (Teacher) wrapper for generating context and questions."""

from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

CHALLENGER_SYSTEM_PROMPT = (
    "You are an expert question designer. Given a context passage, your job is to "
    "craft a challenging, non-trivial question that requires deep comprehension of "
    "the context. The context may come from any domain — including but not limited "
    "to science, history, literature, law, medicine, finance, technology, and "
    "everyday reasoning.\n\n"
    "FIRST, in your private scratch-pad, analyze the context and think step-by-step "
    "about what aspects are most important, nuanced, or easily misunderstood.\n\n"
    "THEN, without revealing any of your private thoughts, output exactly the "
    "following blocks:\n"
    "<question>\n"
    "{A clear, self-contained question that can only be answered correctly by "
    "someone who truly understands the context}\n"
    "</question>\n\n"
    "<rubric>\n"
    "{Evaluation criteria: list the key points a correct answer must cover}\n"
    "</rubric>\n\n"
    "<answer>\n"
    "{The reference answer}\n"
    "</answer>\n\n"
    "Do NOT output anything else — no explanations, no extra markup."
)

CHALLENGER_USER_PROMPT = (
    "Based on the given context, generate one new, challenging question now. "
    "Remember to format the output exactly as instructed."
)


class ChallengeModel:
    """Challenge model that provides context and questions to the Solver."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: Optional[str] = None,
        use_fast: bool = True,
        system_prompt: Optional[str] = None,
    ):
        """
        Args:
            model_name: HuggingFace model identifier.
            device: Device to load model on (None = auto).
            use_fast: Use fast tokenizer.
            system_prompt: Custom system prompt (defaults to CHALLENGER_SYSTEM_PROMPT).
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.system_prompt = system_prompt or CHALLENGER_SYSTEM_PROMPT
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
        messages: Optional[List[dict]] = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        do_sample: bool = True,
        **kwargs,
    ) -> str:
        """
        Generate a challenge question.

        If messages is None, uses the default challenger prompt template (B.2).
        Otherwise uses the provided messages directly.

        Args:
            messages: Input in OpenAI chat format. If None, uses default template.
            max_new_tokens: Max tokens to generate.
            temperature: Sampling temperature.
            do_sample: Whether to sample (vs greedy).
            **kwargs: Additional generation kwargs.

        Returns:
            Generated text containing <question>, <rubric>, and <answer> blocks.
        """
        if messages is None:
            messages = self.build_challenge_messages()

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

    def build_challenge_messages(self) -> List[dict]:
        """Build the default challenger prompt messages following template B.2."""
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": CHALLENGER_USER_PROMPT},
        ]

    def format_prompt(self, messages: List[dict]) -> List[dict]:
        """
        Format messages for Solver consumption.

        Injects the challenger system prompt if no system message is present.
        """
        has_system = any(m.get("role") == "system" for m in messages)
        if has_system:
            return messages
        return [{"role": "system", "content": self.system_prompt}] + messages
