"""Rubrics-based reward using LLM-as-judge or rule-based scoring."""

import json
import re
from typing import Any, Dict, List, Optional

from .base_reward import BaseReward


def build_rubrics_text(rubrics: List) -> str:
    """Build rubrics checklist from rubrics list."""
    if not rubrics:
        return "No specific rubrics provided."
    lines = []
    for i, rubric in enumerate(rubrics, 1):
        if isinstance(rubric, dict):
            criteria = rubric.get("rubric_criteria", rubric.get("criteria", "")).strip()
        else:
            criteria = str(rubric).strip()
        if criteria:
            lines.append(f"{i}. {criteria}")
    return "\n".join(lines) if lines else "No specific rubrics provided."


class RubricsReward(BaseReward):
    """
    Reward based on CL-bench rubrics.
    Supports rule-based (keyword/heuristic) or LLM-as-judge (OpenAI-compatible API).
    """

    def __init__(
        self,
        use_llm_judge: bool = False,
        judge_model: str = "gpt-4",
        api_client=None,
        sub_category_weights: Optional[Dict[str, float]] = None,
        challenge_reward_scale: float = 1.0,
    ):
        """
        Args:
            use_llm_judge: If True, use LLM API for grading; else use rule-based heuristic.
            judge_model: Model name for LLM judge (when use_llm_judge=True).
            api_client: OpenAI-compatible client (required if use_llm_judge=True).
            sub_category_weights: Optional weights per sub_category for reward scaling.
            challenge_reward_scale: Scale factor for challenge reward (often = solver_reward).
        """
        self.use_llm_judge = use_llm_judge
        self.judge_model = judge_model
        self.api_client = api_client
        self.sub_category_weights = sub_category_weights or {}
        self.challenge_reward_scale = challenge_reward_scale

    def compute_solver_reward(
        self,
        answer: str,
        rubrics: List[str],
        metadata: Dict[str, Any],
        **kwargs,
    ) -> float:
        """Compute solver reward: 0 or 1 (binary) or continuous via heuristics."""
        if not answer or not answer.strip():
            return 0.0

        if self.use_llm_judge and self.api_client:
            score = self._llm_grade(answer, rubrics)
        else:
            score = self._heuristic_grade(answer, rubrics)

        sub = metadata.get("sub_category", "")
        weight = self.sub_category_weights.get(sub, 1.0)
        return score * weight

    def _heuristic_grade(self, answer: str, rubrics: List) -> float:
        """Simple heuristic: check if answer is non-empty and reasonably long."""
        if not answer or len(answer.strip()) < 10:
            return 0.0
        rubric_count = len([r for r in rubrics if str(r).strip()])
        if rubric_count == 0:
            return 1.0 if len(answer) > 50 else 0.5
        return 0.5  # Placeholder: real implementation would do keyword/regex checks

    def _llm_grade(self, answer: str, rubrics: List) -> float:
        """Use LLM judge to score 0 or 1 based on rubrics."""
        rubrics_text = build_rubrics_text(rubrics)
        prompt = (
            "You are a strict grading teacher. Grade the student answer based on the rubrics.\n"
            "Output ONLY a JSON object with keys: \"score\" (0 or 1) and \"rationale\" (brief string).\n"
            "Score 1 only if ALL rubric requirements are fully satisfied. Otherwise score 0.\n\n"
            f"Rubrics:\n{rubrics_text}\n\n"
            f"Student answer:\n{answer}\n\n"
            "Output (JSON only):"
        )
        try:
            response = self.api_client.chat.completions.create(
                model=self.judge_model,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.choices[0].message.content.strip()
            text = re.sub(r"^```\w*\n?", "", text)
            text = re.sub(r"\n?```$", "", text)
            data = json.loads(text)
            return float(data.get("score", 0))
        except Exception:
            return 0.0

    def compute_challenge_reward(
        self,
        solver_reward: float,
        context: str,
        metadata: Dict[str, Any],
        **kwargs,
    ) -> float:
        """
        Challenge reward: proportional to Solver's success.
        Can add penalty for trivial/too-helpful context in future.
        """
        return solver_reward * self.challenge_reward_scale
