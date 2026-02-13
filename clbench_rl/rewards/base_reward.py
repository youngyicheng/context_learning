"""Base reward interface for extensible reward design."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseReward(ABC):
    """Abstract base class for dual rewards (Solver + Challenge)."""

    @abstractmethod
    def compute_solver_reward(
        self,
        answer: str,
        rubrics: List[str],
        metadata: Dict[str, Any],
        **kwargs,
    ) -> float:
        """
        Compute reward for the Solver model based on its answer.

        Args:
            answer: Solver's generated answer.
            rubrics: List of evaluation criteria from the dataset.
            metadata: Task metadata (task_id, context_id, context_category, sub_category).
            **kwargs: Additional context (e.g., challenge_output, messages).

        Returns:
            Reward value (typically 0.0 to 1.0, or binary 0/1).
        """
        pass

    @abstractmethod
    def compute_challenge_reward(
        self,
        solver_reward: float,
        context: str,
        metadata: Dict[str, Any],
        **kwargs,
    ) -> float:
        """
        Compute reward for the Challenge model.

        Typically derived from Solver's performance; can include penalties
        for context quality, difficulty calibration, etc.

        Args:
            solver_reward: Reward received by the Solver for this episode.
            context: The context/question provided by Challenge.
            metadata: Task metadata.
            **kwargs: Additional context (e.g., answer, rubrics).

        Returns:
            Reward value for the Challenge model.
        """
        pass
