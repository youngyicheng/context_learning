"""RL environment encapsulating sample -> Challenge -> Solver -> Scoring."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..models.challenge_model import ChallengeModel
from ..models.solver_model import SolverModel
from ..rewards.base_reward import BaseReward


@dataclass
class EnvStep:
    """Output of a single environment step."""

    solver_reward: float
    challenge_reward: float
    answer: str
    challenge_output: Optional[str]
    messages: List[dict]
    rubrics: List[str]
    metadata: Dict[str, Any]


class CLBenchEnv:
    """
    Environment that runs: sample -> (optional Challenge transform) -> Solver generate -> score.
    """

    def __init__(
        self,
        challenge_model: ChallengeModel,
        solver_model: SolverModel,
        reward_fn: BaseReward,
        challenge_pass_through: bool = True,
    ):
        """
        Args:
            challenge_model: Teacher model (can transform or pass through context).
            solver_model: Student model that generates answers.
            reward_fn: Computes solver and challenge rewards.
            challenge_pass_through: If True, skip Challenge generation and use raw messages.
        """
        self.challenge_model = challenge_model
        self.solver_model = solver_model
        self.reward_fn = reward_fn
        self.challenge_pass_through = challenge_pass_through

    def step(
        self,
        sample: Dict[str, Any],
        return_logprobs: bool = False,
    ) -> EnvStep:
        """
        Run one episode: get messages from sample -> Solver answers -> compute rewards.

        Args:
            sample: Dict with 'messages', 'rubrics', 'metadata'.
            return_logprobs: If True, Solver returns (answer, logprobs) for RL training.

        Returns:
            EnvStep with rewards, answer, and context. logprobs stored in metadata if requested.
        """
        messages = sample.get("messages", [])
        rubrics = sample.get("rubrics", [])
        metadata = sample.get("metadata", {})

        if not messages:
            return EnvStep(
                solver_reward=0.0,
                challenge_reward=0.0,
                answer="",
                challenge_output=None,
                messages=messages,
                rubrics=rubrics,
                metadata=metadata,
            )

        challenge_output = None
        if self.challenge_pass_through:
            solver_messages = messages
        else:
            solver_messages = self.challenge_model.format_prompt(messages)
            challenge_output = self.challenge_model.generate(messages)
            if challenge_output:
                solver_messages = [
                    {"role": "system", "content": messages[0].get("content", "") if messages else ""},
                    {"role": "user", "content": challenge_output},
                ]

        if return_logprobs:
            answer, logprobs = self.solver_model.generate(
                solver_messages, return_logprobs=True
            )
            metadata = {**metadata, "_logprobs": logprobs}
        else:
            answer = self.solver_model.generate(solver_messages)

        context_str = self._extract_context(solver_messages)

        solver_reward = self.reward_fn.compute_solver_reward(
            answer=answer,
            rubrics=rubrics,
            metadata=metadata,
            challenge_output=challenge_output,
            messages=messages,
        )

        challenge_reward = self.reward_fn.compute_challenge_reward(
            solver_reward=solver_reward,
            context=context_str,
            metadata=metadata,
            answer=answer,
            rubrics=rubrics,
        )

        return EnvStep(
            solver_reward=solver_reward,
            challenge_reward=challenge_reward,
            answer=answer,
            challenge_output=challenge_output,
            messages=messages,
            rubrics=rubrics,
            metadata=metadata,
        )

    def _extract_context(self, messages: List[dict]) -> str:
        """Extract context string from messages for reward computation."""
        parts = []
        for m in messages:
            if isinstance(m, dict) and m.get("content"):
                parts.append(m["content"])
        return "\n".join(parts)
