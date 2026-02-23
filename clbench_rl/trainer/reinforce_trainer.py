"""REINFORCE-style trainer for Solver (and optionally Challenge) model."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from tqdm import tqdm

from ..data.loader import CLBenchDataLoader
from ..env.clbench_env import CLBenchEnv
from ..models.challenge_model import ChallengeModel
from ..models.solver_model import SolverModel
from ..rewards.rubrics_reward import RubricsReward

logger = logging.getLogger(__name__)


class ReinforceTrainer:
    """
    Trainer that runs pipeline: load data -> env.step -> accumulate rewards -> update.
    Uses REINFORCE for Solver (and optionally Challenge) with policy gradient.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        challenge_model: Optional[ChallengeModel] = None,
        solver_model: Optional[SolverModel] = None,
        reward_fn: Optional[RubricsReward] = None,
    ):
        from ..config.default_config import merge_config

        self.config = merge_config(config)
        self.cfg = self.config

        self.challenge_model = challenge_model or self._create_challenge_model()
        self.solver_model = solver_model or self._create_solver_model()
        self.reward_fn = reward_fn or self._create_reward()

        self.env = CLBenchEnv(
            challenge_model=self.challenge_model,
            solver_model=self.solver_model,
            reward_fn=self.reward_fn,
            challenge_pass_through=True,
        )

        self.checkpoint_dir = Path(
            self.cfg.get("training", {}).get("checkpoint_dir", "checkpoints")
        )
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _create_challenge_model(self) -> ChallengeModel:
        cm = self.cfg.get("challenge_model", {})
        return ChallengeModel(
            model_name=cm.get("model_name", "Qwen/Qwen2.5-1.5B-Instruct"),
            device=cm.get("device"),
        )

    def _create_solver_model(self) -> SolverModel:
        sm = self.cfg.get("solver_model", {})
        return SolverModel(
            model_name=sm.get("model_name", "Qwen/Qwen2.5-1.5B-Instruct"),
            device=sm.get("device"),
        )

    def _create_reward(self) -> RubricsReward:
        rw = self.cfg.get("reward", {})
        return RubricsReward(
            use_llm_judge=rw.get("use_llm_judge", False),
            judge_model=rw.get("judge_model", "gpt-4"),
            api_client=None,
            challenge_correctness_weight=rw.get("challenge_correctness_weight", 1.0),
            repetition_penalty_weight=rw.get("repetition_penalty_weight", 0.3),
            format_penalty_weight=rw.get("format_penalty_weight", 0.2),
            relevance_weight=rw.get("relevance_weight", 0.3),
            rubric_quality_weight=rw.get("rubric_quality_weight", 0.2),
            solver_correctness_weight=rw.get("solver_correctness_weight", 1.0),
            context_grounding_weight=rw.get("context_grounding_weight", 0.3),
            tool_usage_weight=rw.get("tool_usage_weight", 0.2),
        )

    def _create_dataloader(self) -> CLBenchDataLoader:
        d = self.cfg.get("data", {})
        loader = CLBenchDataLoader(
            split=d.get("split", "train"),
            max_samples=d.get("max_samples"),
            subset=d.get("subset"),
            cache_dir=d.get("cache_dir"),
        )
        return loader

    def run_episode(
        self,
        sample: Dict[str, Any],
        return_logprobs: bool = False,
    ) -> Dict[str, Any]:
        """Run one episode and return step result with rewards."""
        step = self.env.step(sample, return_logprobs=return_logprobs)
        out = {
            "solver_reward": step.solver_reward,
            "challenge_reward": step.challenge_reward,
            "solver_breakdown": step.solver_reward_breakdown,
            "challenge_breakdown": step.challenge_reward_breakdown,
            "answer": step.answer,
            "metadata": step.metadata,
        }
        if "_logprobs" in step.metadata:
            out["logprobs"] = step.metadata.pop("_logprobs", None)
        return out

    def train(self) -> Dict[str, float]:
        """
        Run training loop: iterate data, run episodes, optionally update models.
        Collects multi-component rewards and logs detailed breakdowns.
        """
        loader = self._create_dataloader()
        loader.load()

        train_cfg = self.cfg.get("training", {})
        log_every = train_cfg.get("log_every", 5)
        save_every = train_cfg.get("save_every", 100)

        total_solver_r = 0.0
        total_challenge_r = 0.0
        total_solver_correctness = 0.0
        total_solver_grounding = 0.0
        total_challenge_correctness = 0.0
        total_challenge_repetition = 0.0
        n = 0
        samples = list(loader)
        iterator = tqdm(samples, desc="Training")

        for i, sample in enumerate(iterator):
            result = self.run_episode(sample, return_logprobs=False)
            r_s = result["solver_reward"]
            r_c = result["challenge_reward"]
            total_solver_r += r_s
            total_challenge_r += r_c
            n += 1

            s_bd = result.get("solver_breakdown")
            c_bd = result.get("challenge_breakdown")
            if s_bd:
                total_solver_correctness += s_bd.correctness
                total_solver_grounding += s_bd.context_grounding
            if c_bd:
                total_challenge_correctness += c_bd.correctness
                total_challenge_repetition += c_bd.repetition_penalty

            if (i + 1) % log_every == 0:
                avg_s = total_solver_r / n
                avg_c = total_challenge_r / n
                iterator.set_postfix(
                    solver_r=round(avg_s, 4),
                    challenge_r=round(avg_c, 4),
                    s_correct=round(total_solver_correctness / n, 4),
                    s_ground=round(total_solver_grounding / n, 4),
                )

            if save_every and (i + 1) % save_every == 0:
                self._save_checkpoint(i + 1)

        metrics = {
            "mean_solver_reward": total_solver_r / n if n else 0.0,
            "mean_challenge_reward": total_challenge_r / n if n else 0.0,
            "mean_solver_correctness": total_solver_correctness / n if n else 0.0,
            "mean_solver_grounding": total_solver_grounding / n if n else 0.0,
            "mean_challenge_correctness": total_challenge_correctness / n if n else 0.0,
            "mean_challenge_repetition_penalty": total_challenge_repetition / n if n else 0.0,
            "num_episodes": n,
        }
        logger.info("Training complete. Metrics: %s", metrics)
        return metrics

    def _save_checkpoint(self, step: int) -> None:
        """Save model checkpoints."""
        train_cfg = self.cfg.get("training", {})
        if train_cfg.get("train_solver"):
            path = self.checkpoint_dir / f"solver_step_{step}"
            self.solver_model.model.save_pretrained(path)
            self.solver_model.tokenizer.save_pretrained(path)
        if train_cfg.get("train_challenge"):
            path = self.checkpoint_dir / f"challenge_step_{step}"
            self.challenge_model.model.save_pretrained(path)
            self.challenge_model.tokenizer.save_pretrained(path)
