"""Training logic for Challenge and Solver models."""

from .grpo_trainer import GRPOTrainer
from .reinforce_trainer import ReinforceTrainer

__all__ = ["ReinforceTrainer", "GRPOTrainer"]
