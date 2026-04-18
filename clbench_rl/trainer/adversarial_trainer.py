"""Adversarial self-play trainer aligned with the Self-Evolving ICL paper.

Both Solver (πθs) and Challenger (πθc) are trained via GRPO:

  Solver objective:     max E[J_score(A, R)]
  Challenger objective: max E[w1·R_adv - w2·R_rep - w3·R_fmt + w4·R_rel + w5·R_rubric]

Training loop (per epoch):
  For each context C in the dataset:
    1. Challenger generates G candidate (Q,E,R) from C   → compute R_C for each
    2. Pick best (Q,E,R) by R_C; Solver generates G answers A  → compute J_score
    3. GRPO update on Challenger using challenger rewards
    4. GRPO update on Solver using solver rewards
    5. Advance dynamic weight scheduler

Supports DeepSpeed ZeRO via Accelerate for 8×H100 training.
"""

import concurrent.futures
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from tqdm import tqdm

from ..config.default_config import merge_config
from ..data.loader import CLBenchDataLoader
from ..models.challenge_model import ChallengeModel, parse_challenger_output
from ..models.solver_model import SolverModel
from ..rewards.base_reward import ChallengeRewardResult
from ..rewards.rubrics_reward import DynamicWeightScheduler, RubricsReward
from ..utils.metrics_logger import MetricsLogger

logger = logging.getLogger(__name__)


@dataclass
class GroupResult:
    """Result of one GRPO group (shared by Solver and Challenger)."""
    responses: List[str] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    advantages: List[float] = field(default_factory=list)
    input_ids: Optional[torch.Tensor] = None
    generated_ids_list: List[torch.Tensor] = field(default_factory=list)


class AdversarialTrainer:
    """Dual-model GRPO trainer: trains both Solver and Challenger.

    Paper: "The framework is trained using an actor-critic or direct evolution
    approach with the following objectives" — both θs and θc are optimized.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.cfg = merge_config(config)

        grpo = self.cfg.get("grpo", {})
        self.group_size = grpo.get("group_size", 4)
        self.clip_eps = grpo.get("clip_eps", 0.2)
        self.kl_beta = grpo.get("kl_beta", 0.04)
        self.adv_eps = grpo.get("adv_eps", 1e-8)
        self.max_grad_norm = grpo.get("max_grad_norm", 1.0)
        self.mu_iterations = grpo.get("mu_iterations", 1)

        self._init_distributed()

        self.solver_device, self.challenger_device = self._assign_devices()

        self.solver = self._build_solver()
        self.challenger = self._build_challenger()

        # LoRA mode: ref policy = same base model with the adapter disabled
        # (see SolverModel.disabled_adapter). No separate frozen copy is
        # instantiated, saving ~2× model memory per GPU.
        self.use_lora = self.solver.use_lora and self.challenger.use_lora
        self.solver_ref: Optional[SolverModel] = None
        self.challenger_ref: Optional[ChallengeModel] = None
        if not self.use_lora:
            # Legacy path — full fine-tune fallback (also keeps 8-bit AdamW +
            # ref-offload optimization from before).
            self.solver_ref = self._build_reference(self.solver)
            self.challenger_ref = self._build_reference_challenger(self.challenger)
            self.solver_ref.model.to("cpu")
            self.challenger_ref.model.to("cpu")
            torch.cuda.empty_cache()
            logger.info(
                "Full fine-tune mode: reference models offloaded to CPU"
            )
        else:
            logger.info(
                "LoRA mode: reference policy uses `disable_adapter()` — "
                "no separate ref model instantiated."
            )

        self.reward_fn = self._build_reward()

        train_cfg = self.cfg.get("training", {})
        # 8-bit AdamW is only useful when optimizing the full base model.
        # In LoRA mode, trainable params are tiny (<1% of the model) so plain
        # fp32 AdamW on LoRA weights is both simpler and faster.
        use_8bit = train_cfg.get("use_8bit_optimizer", True) and not self.use_lora
        AdamWCls = self._resolve_adamw(use_8bit)

        self.solver_optimizer = AdamWCls(
            self._trainable_params(self.solver.model),
            lr=train_cfg.get("solver_lr", train_cfg.get("lr", 1e-5)),
            weight_decay=train_cfg.get("weight_decay", 0.01),
        )
        self.challenger_optimizer = AdamWCls(
            self._trainable_params(self.challenger.model),
            lr=train_cfg.get("challenger_lr", train_cfg.get("lr", 1e-5)),
            weight_decay=train_cfg.get("weight_decay", 0.01),
        )

        self.checkpoint_dir = Path(train_cfg.get("checkpoint_dir", "checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        rw_cfg = self.cfg.get("reward", {})
        self.rep_batch_size = rw_cfg.get("repetition_batch_size", 16)

        # Scale-up: parallelise judge API calls per step (I/O bound).
        jc = train_cfg.get("judge_concurrency")
        self.judge_concurrency = int(jc) if jc else max(self.group_size, 1)

        # Gradient checkpointing (training forward only). We toggle it around
        # `_do_grpo_update`, so `generate()` keeps its KV cache for speed.
        self.use_grad_ckpt = bool(train_cfg.get("gradient_checkpointing", True))

        # Fixed train/test split controls.
        self.test_size = int(train_cfg.get("test_size", 100) or 0)
        self.test_seed = int(train_cfg.get("test_seed", 42))
        self.run_eval = bool(train_cfg.get("run_eval", True))
        self.eval_max_new_tokens = int(
            train_cfg.get("eval_max_new_tokens", 0) or 0
        )

    # ------------------------------------------------------------------
    # Distributed init & device assignment
    # ------------------------------------------------------------------

    def _init_distributed(self) -> None:
        """Initialise torch.distributed if launched via torchrun.

        Detection is purely env-driven (`WORLD_SIZE`, `RANK`, `LOCAL_RANK`),
        so plain `python scripts/train_adversarial.py` keeps working
        single-process with `world_size=1`.

        We deliberately do **not** wrap models in `DistributedDataParallel`.
        In LoRA-GRPO the training-time forward paths include `generate()`,
        `disable_adapter()` and a hand-written batched logprob kernel, none
        of which play nicely with DDP's module wrapper. Instead we manually
        `all_reduce` LoRA gradients just before `optimizer.step()` — this
        is the approach used by trl / openrlhf and keeps `self.model` a
        plain `PeftModel`.
        """
        ws = int(os.environ.get("WORLD_SIZE", "1"))
        self.world_size = ws
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.is_rank0 = self.rank == 0

        if ws > 1 and not dist.is_initialized():
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            if torch.cuda.is_available():
                torch.cuda.set_device(self.local_rank)
            dist.init_process_group(backend=backend)
            logger.info(
                "torch.distributed initialised: rank=%d/%d local_rank=%d backend=%s",
                self.rank, ws, self.local_rank, backend,
            )

    def _is_distributed(self) -> bool:
        return self.world_size > 1 and dist.is_available() and dist.is_initialized()

    def _barrier(self) -> None:
        if self._is_distributed():
            dist.barrier()

    @staticmethod
    def _resolve_adamw(use_8bit: bool):
        """Return an AdamW-like optimizer class.

        Prefer bitsandbytes PagedAdamW8bit when available — cuts optimizer
        state memory from ~32 GB to ~4 GB per 4B model, which is critical
        for fitting dual-model adversarial GRPO on 2×80 GB.
        """
        if use_8bit:
            try:
                import bitsandbytes as bnb
                logger.info("Using bitsandbytes PagedAdamW8bit (saves ~28 GB/GPU)")
                return bnb.optim.PagedAdamW8bit
            except ImportError:
                logger.warning(
                    "bitsandbytes not installed — falling back to torch.optim.AdamW. "
                    "Install with: pip install bitsandbytes"
                )
        return torch.optim.AdamW

    def _assign_devices(self):
        """Place Solver and Challenger on GPU(s).

        - DDP mode (torchrun, world_size > 1): each rank pins to
          `cuda:{LOCAL_RANK}` and hosts a full Solver+Challenger pair.
        - Single-process: honour `training.colocate_models` (default True),
          else fall back to the legacy split (solver→cuda:0,
          challenger→cuda:1).
        """
        n = torch.cuda.device_count()
        train_cfg = self.cfg.get("training", {})
        colocate = train_cfg.get("colocate_models", True)

        if n == 0:
            logger.info("No GPU detected — running on CPU")
            return "cpu", "cpu"

        if self._is_distributed():
            dev = f"cuda:{self.local_rank}"
            logger.info(
                "DDP rank=%d: Solver+Challenger co-located on %s",
                self.rank, dev,
            )
            return dev, dev

        if colocate:
            idx = self.local_rank if self.local_rank < n else 0
            dev = f"cuda:{idx}"
            logger.info(
                "Co-locating Solver and Challenger on %s "
                "(LoRA leaves ample headroom on 80GB)",
                dev,
            )
            return dev, dev

        if n >= 2:
            logger.info("Split mode: solver→cuda:0, challenger→cuda:1")
            return "cuda:0", "cuda:1"

        logger.info("Single GPU — all models on cuda:0")
        return "cuda:0", "cuda:0"

    # ------------------------------------------------------------------
    # Model builders
    # ------------------------------------------------------------------

    @staticmethod
    def _trainable_params(model):
        return [p for p in model.parameters() if p.requires_grad]

    def _allreduce_grads(self, model) -> None:
        """Average LoRA grads across ranks (manual DDP).

        No-op when `world_size == 1`. Called just before `optimizer.step()`
        so every rank applies the same update on its local LoRA weights.
        """
        if not self._is_distributed():
            return
        ws = float(self.world_size)
        for p in self._trainable_params(model):
            if p.grad is None:
                continue
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad.mul_(1.0 / ws)

    def _build_solver(self) -> SolverModel:
        sm = self.cfg.get("solver_model", {})
        return SolverModel(
            model_name=sm.get("model_name", "Qwen/Qwen3-4B-Instruct-2507"),
            device=sm.get("device") or self.solver_device,
            lora_cfg=self.cfg.get("lora"),
        )

    def _build_challenger(self) -> ChallengeModel:
        cm = self.cfg.get("challenge_model", {})
        return ChallengeModel(
            model_name=cm.get("model_name", "Qwen/Qwen3-4B-Instruct-2507"),
            device=cm.get("device") or self.challenger_device,
            lora_cfg=self.cfg.get("lora"),
        )

    def _build_reference(self, solver: SolverModel) -> SolverModel:
        sm = self.cfg.get("solver_model", {})
        ref = SolverModel(
            model_name=sm.get("model_name", "Qwen/Qwen3-4B-Instruct-2507"),
            device=sm.get("device") or self.solver_device,
        )
        ref.model.load_state_dict(solver.model.state_dict())
        ref.model.eval()
        for p in ref.model.parameters():
            p.requires_grad = False
        return ref

    def _build_reference_challenger(self, challenger: ChallengeModel) -> ChallengeModel:
        cm = self.cfg.get("challenge_model", {})
        ref = ChallengeModel(
            model_name=cm.get("model_name", "Qwen/Qwen3-4B-Instruct-2507"),
            device=cm.get("device") or self.challenger_device,
        )
        ref.model.load_state_dict(challenger.model.state_dict())
        ref.model.eval()
        for p in ref.model.parameters():
            p.requires_grad = False
        return ref

    def _build_reward(self) -> RubricsReward:
        rw = self.cfg.get("reward", {})
        train_cfg = self.cfg.get("training", {})

        scheduler = None
        if rw.get("use_dynamic_weights", True):
            data_cfg = self.cfg.get("data", {})
            total_steps = max(
                train_cfg.get("epochs", 3) * (data_cfg.get("max_samples") or 5000),
                1,
            )
            scheduler = DynamicWeightScheduler(
                total_steps=total_steps,
                w1_init=rw.get("w1_init", 0.3),
                w1_final=rw.get("w1_final", 1.0),
                w2=rw.get("w2_repetition", 0.3),
                w3_init=rw.get("w3_init", 0.5),
                w3_final=rw.get("w3_final", 0.1),
                w4=rw.get("w4_relevance", 0.3),
                w5=rw.get("w5_rubric", 0.2),
            )

        from ..rewards.rubrics_reward import build_judge_api_client

        use_llm = rw.get("use_llm_judge", True)
        client = build_judge_api_client() if use_llm else None

        return RubricsReward(
            use_llm_judge=use_llm,
            judge_model=rw.get("judge_model", "gpt-4o-mini"),
            judge_temperature=rw.get("judge_temperature", 0.1),
            api_client=client,
            w1_adversarial=rw.get("w1_adversarial", 1.0),
            w2_repetition=rw.get("w2_repetition", 0.3),
            w3_format=rw.get("w3_format", 0.2),
            w4_relevance=rw.get("w4_relevance", 0.3),
            w5_rubric=rw.get("w5_rubric", 0.2),
            bleu_distance_threshold=rw.get("bleu_distance_threshold", 0.5),
            weight_scheduler=scheduler,
        )

    # ------------------------------------------------------------------
    # GRPO primitives (shared between Solver & Challenger)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_advantages(rewards: List[float], eps: float = 1e-8) -> List[float]:
        t = torch.tensor(rewards, dtype=torch.float32)
        return ((t - t.mean()) / (t.std() + eps)).tolist()

    def _do_grpo_update(self, model, ref_model, optimizer, group: GroupResult):
        """Batched GRPO update.

        Ref log-probs:
          * LoRA mode: `model.compute_per_token_logprobs_ref_batched()` —
            same base model with adapter disabled, one batched forward for
            all G trajectories.
          * Full-FT mode: separate CPU-offloaded `ref_model`, also batched.

        Policy log-probs: one batched forward over all valid trajectories.
        All per-trajectory losses are summed into a single scalar and we
        call `.backward()` **once** — gradients from every trajectory share
        the same computation graph.
        """
        metrics: Dict[str, float] = {
            "policy_loss": 0.0, "kl": 0.0, "total_loss": 0.0,
        }
        device = model.model.device
        G = len(group.responses)
        input_ids = group.input_ids.to(device)
        gen_list = [g.to(device) for g in group.generated_ids_list]

        valid_idx = [i for i in range(G) if gen_list[i].numel() > 0]
        if not valid_idx:
            return metrics
        valid_gens = [gen_list[i] for i in valid_idx]

        # ---- Phase 1: ref logprobs (batched) ----
        if self.use_lora:
            ref_lps = model.compute_per_token_logprobs_ref_batched(
                input_ids, valid_gens
            )
        else:
            assert ref_model is not None, "ref_model required in full-FT mode"
            ref_model.model.to(device)
            ref_lps = ref_model.compute_per_token_logprobs_detached_batched(
                input_ids, valid_gens
            )
            ref_model.model.to("cpu")
            torch.cuda.empty_cache()

        # ---- Phase 2: GRPO mu-iterations (batched fwd + single bwd) ----
        # Gradient checkpointing cuts activation memory ~2-3× at ~20% extra
        # compute. Switch on only around the differentiable forward, so the
        # (KV-cache-dependent) `generate()` calls outside remain fast.
        gc_active = self.use_grad_ckpt and any(
            p.requires_grad for p in model.model.parameters()
        )
        if gc_active:
            model.enable_gradient_checkpointing()
        try:
            for _ in range(self.mu_iterations):
                model.model.train()
                optimizer.zero_grad()

                cur_lps = model.compute_per_token_logprobs_batched(
                    input_ids, valid_gens
                )

                total_loss: Optional[torch.Tensor] = None
                sum_policy = 0.0
                sum_kl = 0.0
                scale = max(len(valid_idx), 1)

                for k, i in enumerate(valid_idx):
                    cur_lp = cur_lps[k]
                    ref_lp = ref_lps[k].to(device)
                    old_lp = cur_lp.detach()

                    ratio = torch.exp(cur_lp - old_lp)
                    adv = torch.tensor(
                        group.advantages[i], device=device, dtype=torch.float32
                    )
                    surr1 = ratio * adv
                    surr2 = torch.clamp(
                        ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps
                    ) * adv
                    p_loss_i = -torch.min(surr1, surr2).mean()
                    kl_i = (cur_lp - ref_lp).mean()

                    loss_i = (p_loss_i + self.kl_beta * kl_i) / scale
                    total_loss = loss_i if total_loss is None else total_loss + loss_i

                    sum_policy += p_loss_i.item()
                    sum_kl += kl_i.item()

                if total_loss is not None:
                    total_loss.backward()
                    self._allreduce_grads(model.model)
                    torch.nn.utils.clip_grad_norm_(
                        self._trainable_params(model.model), self.max_grad_norm
                    )
                    optimizer.step()

                metrics = {
                    "policy_loss": sum_policy / scale,
                    "kl": sum_kl / scale,
                    "total_loss": (sum_policy + self.kl_beta * sum_kl) / scale,
                }
        finally:
            if gc_active:
                model.disable_gradient_checkpointing()

        model.model.eval()
        return metrics

    # ------------------------------------------------------------------
    # Context extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_context(messages: List[dict]) -> str:
        parts = []
        for m in messages:
            if isinstance(m, dict) and m.get("content"):
                parts.append(m["content"])
        return "\n".join(parts)

    @staticmethod
    def _truncate_text(text: str, max_chars: int) -> str:
        if max_chars <= 0 or len(text) <= max_chars:
            return text
        return text[:max_chars] + f"\n... [truncated, {len(text)} chars total]"

    @staticmethod
    def _json_safe(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {str(k): AdversarialTrainer._json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [AdversarialTrainer._json_safe(x) for x in obj]
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        return str(obj)

    def _append_rollout_trace(
        self,
        path: Path,
        record: Dict[str, Any],
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self) -> Dict[str, float]:
        """Run adversarial self-play training.

        Per-step:
          1. Challenger samples G outputs (Q,E,R) from context C
          2. Compute challenger rewards R_C for each
          3. GRPO update Challenger on R_C
          4. Pick best (Q,E,R); Solver samples G answers
          5. Compute solver rewards J_score for each
          6. GRPO update Solver on J_score
        """
        # Load the full split once (no max_samples here): we slice
        # train/test deterministically *in Python* so the held-out test set
        # is invariant to --max-samples.
        data_cfg = self.cfg.get("data", {})
        loader = CLBenchDataLoader(**{
            k: v for k, v in data_cfg.items()
            if k in ("split", "subset", "cache_dir")
        })
        loader.load()

        train_cfg = self.cfg.get("training", {})
        epochs = train_cfg.get("epochs", 3)
        log_every = train_cfg.get("log_every", 10)
        save_every = train_cfg.get("save_every", 500)
        ref_sync_every = train_cfg.get("ref_sync_every", 200)

        save_rollouts = train_cfg.get("save_rollout_traces", True)
        is_rank0 = self.is_rank0
        ckpt_root = Path(train_cfg.get("checkpoint_dir", "checkpoints"))
        trace_dir_cfg = train_cfg.get("rollout_trace_dir")
        rollout_trace_dir = (
            Path(trace_dir_cfg)
            if trace_dir_cfg
            else ckpt_root / "rollout_traces"
        )
        max_ctx = int(train_cfg.get("rollout_log_max_context_chars", 8000))
        max_field = int(train_cfg.get("rollout_log_max_field_chars", 16000))
        rollout_jsonl = rollout_trace_dir / "training_rollouts.jsonl"
        if save_rollouts and is_rank0:
            rollout_trace_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Rollout traces → %s", rollout_jsonl)

        cm_cfg = self.cfg.get("challenge_model", {})
        sm_cfg = self.cfg.get("solver_model", {})

        # Input-length caps to bound activation memory.
        max_ctx_ch = int(train_cfg.get("max_context_chars_challenger", 6000))
        max_ctx_sv = int(train_cfg.get("max_context_chars_solver", 4000))

        all_samples = list(loader)
        train_samples, test_samples = self._split_train_test(
            all_samples,
            test_size=self.test_size,
            seed=self.test_seed,
            max_train=data_cfg.get("max_samples"),
        )
        if is_rank0:
            logger.info(
                "Split: %d total → %d train (capped by max_samples=%s) + "
                "%d test (seed=%d, fixed)",
                len(all_samples), len(train_samples),
                data_cfg.get("max_samples"),
                len(test_samples), self.test_seed,
            )

        samples = train_samples
        if self._is_distributed():
            full_n = len(samples)
            samples = samples[self.rank::self.world_size]
            logger.info(
                "DDP rank %d/%d: %d samples (of %d total)",
                self.rank, self.world_size, len(samples), full_n,
            )
        global_step = 0

        acc = {
            "solver_reward": 0.0, "challenger_reward": 0.0,
            "j_score": 0.0, "r_adv": 0.0, "r_rep": 0.0,
            "r_fmt": 0.0, "r_rel": 0.0, "r_rubric": 0.0,
            "solver_loss": 0.0, "challenger_loss": 0.0,
        }
        n = 0

        metrics_path = Path(train_cfg.get("checkpoint_dir", "checkpoints")) / "metrics.jsonl"
        ml = MetricsLogger(metrics_path, flush_every=max(1, log_every)) if is_rank0 else None

        for epoch in range(epochs):
            iterator = tqdm(
                samples,
                desc=f"Epoch {epoch+1}/{epochs}",
                disable=not is_rank0,
            )

            batch_questions: List[str] = []

            for sample in iterator:
                messages = sample.get("messages", [])
                if not messages:
                    continue

                context = self._extract_context(messages)
                ch_context = context[:max_ctx_ch] if max_ctx_ch > 0 else context
                sv_context = context[:max_ctx_sv] if max_ctx_sv > 0 else context
                challenger_messages = self.challenger.build_context_messages(ch_context)

                # === Phase 1: Challenger generates G (Q,E,R) ===
                c_group_raw = self.challenger.generate_group(
                    challenger_messages,
                    group_size=self.group_size,
                    max_new_tokens=cm_cfg.get("max_new_tokens", 1024),
                    temperature=cm_cfg.get("temperature", 0.7),
                )

                c_texts = [r[0] for r in c_group_raw]
                c_input_ids = c_group_raw[0][1]
                c_gen_ids_list = [r[2] for r in c_group_raw]

                parsed_outputs = [parse_challenger_output(t) for t in c_texts]
                questions = [p.question or p.raw for p in parsed_outputs]
                batch_questions.extend(questions)

                # Compute batch-level repetition penalties
                if len(batch_questions) >= self.rep_batch_size:
                    rep_penalties = self.reward_fn.compute_batch_repetition(
                        batch_questions[-self.rep_batch_size:]
                    )
                else:
                    rep_penalties = self.reward_fn.compute_batch_repetition(batch_questions)

                # For each challenger output, compute challenger reward.
                # Scale-up: (a) batch the G quick-answer generates into one
                # padded generate call; (b) fan out the 2×G judge API calls
                # across a thread-pool (I/O bound).
                sv_tok = self.solver.tokenizer
                quick_prompts: List[str] = []
                for gi, parsed in enumerate(parsed_outputs):
                    q_text = parsed.question or c_texts[gi]
                    msgs = self.solver._ensure_system_prompt([
                        {"role": "system", "content": sv_context[:2000]},
                        {"role": "user", "content": q_text},
                    ])
                    quick_prompts.append(
                        sv_tok.apply_chat_template(
                            msgs, tokenize=False, add_generation_prompt=True,
                        )
                    )

                quick_answers = self.solver.generate_batch(
                    quick_prompts,
                    max_new_tokens=sm_cfg.get("max_new_tokens", 2048),
                    temperature=0.3,
                )

                G_c = len(parsed_outputs)
                rubrics_list = [
                    ([p.rubric] if p.rubric else sample.get("rubrics", []))
                    for p in parsed_outputs
                ]
                meta = sample.get("metadata", {})

                def _eval_pair(gi: int):
                    parsed = parsed_outputs[gi]
                    s_res = self.reward_fn.compute_solver_reward(
                        answer=quick_answers[gi], rubrics=rubrics_list[gi],
                        context=context, metadata=meta,
                    )
                    rep_idx = (
                        min(gi, len(rep_penalties) - 1) if rep_penalties else 0
                    )
                    rep_val = rep_penalties[rep_idx] if rep_penalties else 0.0
                    c_res = self.reward_fn.compute_challenge_reward(
                        solver_correctness=s_res.correctness,
                        challenge_output=c_texts[gi],
                        context=context,
                        question=parsed.question,
                        rubrics=rubrics_list[gi],
                        metadata=meta,
                        evidence_span=parsed.evidence_span,
                        batch_repetition_penalty=rep_val,
                    )
                    return gi, s_res, c_res

                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=min(G_c, self.judge_concurrency)
                ) as ex:
                    pair_results = list(ex.map(_eval_pair, range(G_c)))
                pair_results.sort(key=lambda x: x[0])

                c_rewards: List[float] = []
                c_reward_details: List[ChallengeRewardResult] = []
                best_idx = 0
                best_c_reward = float("-inf")
                for gi, _s_res, c_res in pair_results:
                    c_rewards.append(c_res.total)
                    c_reward_details.append(c_res)
                    if c_res.total > best_c_reward:
                        best_c_reward = c_res.total
                        best_idx = gi

                # === Phase 2: GRPO update Challenger ===
                c_advantages = self._compute_advantages(c_rewards, self.adv_eps)
                c_group = GroupResult(
                    responses=c_texts,
                    rewards=c_rewards,
                    advantages=c_advantages,
                    input_ids=c_input_ids,
                    generated_ids_list=c_gen_ids_list,
                )
                c_metrics = self._do_grpo_update(
                    self.challenger, self.challenger_ref,
                    self.challenger_optimizer, c_group,
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # === Phase 3: Solver generates G answers for best (Q,E,R) ===
                best_parsed = parsed_outputs[best_idx]
                best_q = best_parsed.question or c_texts[best_idx]
                best_rubrics = [best_parsed.rubric] if best_parsed.rubric else sample.get("rubrics", [])

                solver_prompt = [
                    {"role": "system", "content": sv_context},
                    {"role": "user", "content": best_q},
                ]

                s_group_raw = self.solver.generate_group(
                    solver_prompt,
                    group_size=self.group_size,
                    max_new_tokens=sm_cfg.get("max_new_tokens", 2048),
                    temperature=sm_cfg.get("temperature", 0.7),
                )

                s_texts = [r[0] for r in s_group_raw]
                s_input_ids = s_group_raw[0][1]
                s_gen_ids_list = [r[2] for r in s_group_raw]

                def _score_solver(ans: str):
                    return self.reward_fn.compute_solver_reward(
                        answer=ans, rubrics=best_rubrics,
                        context=context, metadata=sample.get("metadata", {}),
                    )

                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=min(len(s_texts) or 1, self.judge_concurrency)
                ) as ex:
                    s_breakdowns = list(ex.map(_score_solver, s_texts))
                s_rewards = [sr.total for sr in s_breakdowns]

                # === Phase 4: GRPO update Solver ===
                s_advantages = self._compute_advantages(s_rewards, self.adv_eps)
                s_group = GroupResult(
                    responses=s_texts,
                    rewards=s_rewards,
                    advantages=s_advantages,
                    input_ids=s_input_ids,
                    generated_ids_list=s_gen_ids_list,
                )
                s_metrics = self._do_grpo_update(
                    self.solver, self.solver_ref,
                    self.solver_optimizer, s_group,
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # === Metrics ===
                self.reward_fn.step()
                global_step += 1
                n += 1

                if save_rollouts and is_rank0:
                    cand = []
                    for gi, parsed in enumerate(parsed_outputs):
                        cr = c_reward_details[gi]
                        cand.append({
                            "index": gi,
                            "question": self._truncate_text(
                                parsed.question or "", max_field
                            ),
                            "evidence_span": self._truncate_text(
                                parsed.evidence_span or "", max_field
                            ),
                            "rubric": self._truncate_text(
                                parsed.rubric or "", max_field
                            ),
                            "raw_response": self._truncate_text(
                                c_texts[gi], max_field
                            ),
                            "total_reward": cr.total,
                            "adversarial": cr.adversarial,
                            "repetition_penalty": cr.repetition_penalty,
                            "format_penalty": cr.format_penalty,
                            "relevance": cr.relevance,
                            "rubric_quality": cr.rubric_quality,
                            "details": self._json_safe(cr.details),
                        })
                    solver_rows = []
                    for j, (ans, sr) in enumerate(zip(s_texts, s_breakdowns)):
                        solver_rows.append({
                            "index": j,
                            "answer": self._truncate_text(ans, max_field),
                            "total_reward": sr.total,
                            "correctness": sr.correctness,
                            "details": self._json_safe(sr.details),
                        })
                    trace_rec = {
                        "global_step": global_step,
                        "epoch": epoch,
                        "sample_metadata": self._json_safe(
                            sample.get("metadata") or {}
                        ),
                        "context_preview": self._truncate_text(context, max_ctx),
                        "challenger_group_size": len(c_texts),
                        "best_challenger_index": best_idx,
                        "mean_challenger_reward": sum(c_rewards) / max(len(c_rewards), 1),
                        "challenger_candidates": cand,
                        "best_question": self._truncate_text(best_q, max_field),
                        "best_rubric": self._truncate_text(
                            best_parsed.rubric or "", max_field
                        ),
                        "solver_on_best": {
                            "mean_reward": sum(s_rewards) / max(len(s_rewards), 1),
                            "mean_judge_score": sum(
                                sr.correctness for sr in s_breakdowns
                            ) / max(len(s_breakdowns), 1),
                            "answers": solver_rows,
                        },
                        "challenger_loss": c_metrics.get("total_loss", 0.0),
                        "solver_loss": s_metrics.get("total_loss", 0.0),
                    }
                    self._append_rollout_trace(rollout_jsonl, trace_rec)

                mean_j = sum(sr.correctness for sr in s_breakdowns) / max(len(s_breakdowns), 1)
                acc["solver_reward"] += sum(s_rewards) / max(len(s_rewards), 1)
                acc["challenger_reward"] += sum(c_rewards) / max(len(c_rewards), 1)
                acc["j_score"] += mean_j
                acc["solver_loss"] += s_metrics.get("total_loss", 0.0)
                acc["challenger_loss"] += c_metrics.get("total_loss", 0.0)

                if global_step % log_every == 0 and n > 0 and is_rank0:
                    iterator.set_postfix(
                        j=round(acc["j_score"] / n, 4),
                        s_loss=round(acc["solver_loss"] / n, 4),
                        c_loss=round(acc["challenger_loss"] / n, 4),
                        c_r=round(acc["challenger_reward"] / n, 4),
                    )
                    if ml is not None:
                        ml.log(
                            step=global_step, epoch=epoch,
                            j_score=mean_j,
                            solver_reward=sum(s_rewards) / max(len(s_rewards), 1),
                            challenger_reward=sum(c_rewards) / max(len(c_rewards), 1),
                            solver_loss=s_metrics.get("total_loss", 0.0),
                            challenger_loss=c_metrics.get("total_loss", 0.0),
                            solver_policy_loss=s_metrics.get("policy_loss", 0.0),
                            challenger_policy_loss=c_metrics.get("policy_loss", 0.0),
                            solver_kl=s_metrics.get("kl_penalty", 0.0),
                            challenger_kl=c_metrics.get("kl_penalty", 0.0),
                            avg_j_score=acc["j_score"] / n,
                            avg_solver_reward=acc["solver_reward"] / n,
                            avg_challenger_reward=acc["challenger_reward"] / n,
                        )

                if save_every and global_step % save_every == 0 and is_rank0:
                    self._save_checkpoint(global_step)
                # Keep all ranks in lock-step after a checkpoint so that
                # rank 0 doesn't fall behind while writing.
                if save_every and global_step % save_every == 0:
                    self._barrier()

                if ref_sync_every and global_step % ref_sync_every == 0:
                    self._sync_references()

            # Trim batch_questions per epoch to avoid unbounded growth
            batch_questions = batch_questions[-self.rep_batch_size * 4:]

        if is_rank0:
            self._save_checkpoint(global_step)
            if ml is not None:
                ml.close()
            logger.info("Metrics saved to %s", metrics_path)
        self._barrier()

        out = {k: v / max(n, 1) for k, v in acc.items()} | {
            "global_steps": global_step,
            "epochs": epochs,
            "metrics_file": str(metrics_path),
            "world_size": self.world_size,
            "rank": self.rank,
        }
        if save_rollouts and is_rank0:
            out["rollout_trace_file"] = str(rollout_jsonl)

        # --------------------------- Evaluation ---------------------------
        # Only rank 0 runs it (weights are identical across ranks after
        # all_reduce); other ranks wait at the barrier so they don't exit
        # the process group prematurely.
        if self.run_eval and test_samples:
            if is_rank0:
                eval_summary = self.evaluate(
                    test_samples,
                    tag=f"step_{global_step}",
                )
                out["test_summary"] = eval_summary
            self._barrier()

        if self._is_distributed() and dist.is_initialized():
            dist.destroy_process_group()
        return out

    # ------------------------------------------------------------------
    # Checkpointing & reference sync
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Train / test split + evaluation
    # ------------------------------------------------------------------

    @staticmethod
    def _split_train_test(
        samples: List[Any],
        test_size: int,
        seed: int,
        max_train: Optional[int] = None,
    ) -> Tuple[List[Any], List[Any]]:
        """Deterministic train/test split.

        The test set is defined by a **seed-shuffled permutation of the full
        split** so that:

          * it is reproducible across runs (same `seed` → same test set);
          * it is invariant to `max_samples` (capping the training pool never
            leaks test samples into training);
          * rank 0 and non-rank-0 workers compute *the same* partition (only
            rank 0 actually evaluates, but all workers hold the same view).
        """
        import random

        if test_size <= 0 or test_size >= len(samples):
            train = samples[:]
            test: List[Any] = []
        else:
            indices = list(range(len(samples)))
            random.Random(seed).shuffle(indices)
            test_idx = sorted(indices[-test_size:])
            test_set = set(test_idx)
            test = [samples[i] for i in test_idx]
            train = [s for i, s in enumerate(samples) if i not in test_set]

        if max_train is not None and max_train > 0:
            train = train[:max_train]
        return train, test

    def evaluate(
        self,
        test_samples: List[Dict[str, Any]],
        tag: str = "final",
    ) -> Dict[str, float]:
        """Run the trained Solver+Challenger on a fixed held-out test set.

        For each context:
          1. Challenger generates one (Q, E, R) via low-temperature sampling.
          2. Solver answers that question on the same (truncated) context.
          3. Judge scores the answer against the generated rubric, yielding
             `J_score ∈ [0, 1]`.

        Per-sample records are appended to
        `<checkpoint_dir>/test_results_<tag>.jsonl`, and the aggregate goes
        to `<output_dir>/test_summary_<tag>.json` (best effort).

        Called from rank 0 only; safe to call stand-alone after loading a
        trained adapter as well.
        """
        if not self.is_rank0:
            return {}
        if not test_samples:
            logger.info("Evaluation skipped: empty test set.")
            return {}

        train_cfg = self.cfg.get("training", {})
        cm_cfg = self.cfg.get("challenge_model", {})
        sm_cfg = self.cfg.get("solver_model", {})

        max_ctx_ch = int(train_cfg.get("max_context_chars_challenger", 6000))
        max_ctx_sv = int(train_cfg.get("max_context_chars_solver", 4000))
        eval_new_tokens = (
            self.eval_max_new_tokens
            or int(sm_cfg.get("max_new_tokens", 1024))
        )
        ch_new_tokens = int(cm_cfg.get("max_new_tokens", 512))

        results_path = self.checkpoint_dir / f"test_results_{tag}.jsonl"
        if results_path.exists():
            results_path.unlink()
        results_path.parent.mkdir(parents=True, exist_ok=True)

        self.solver.model.eval()
        self.challenger.model.eval()

        totals = {
            "n": 0, "n_parsed": 0,
            "j_score_sum": 0.0,
            "challenger_fmt_ok": 0,
        }

        logger.info(
            "[Eval] Running held-out evaluation on %d samples → %s",
            len(test_samples), results_path,
        )

        iterator = tqdm(test_samples, desc=f"Eval ({tag})", disable=False)
        for idx, sample in enumerate(iterator):
            messages = sample.get("messages", [])
            if not messages:
                continue
            context = self._extract_context(messages)
            ch_context = context[:max_ctx_ch] if max_ctx_ch > 0 else context
            sv_context = context[:max_ctx_sv] if max_ctx_sv > 0 else context
            meta = sample.get("metadata", {})

            # 1) Challenger: one low-temp rollout.
            ch_msgs = self.challenger.build_context_messages(ch_context)
            try:
                ch_text = self.challenger.generate(
                    ch_msgs,
                    max_new_tokens=ch_new_tokens,
                    temperature=0.3,
                    do_sample=True,
                )
            except Exception as e:
                logger.warning("[Eval] Challenger failed on idx=%d: %s", idx, e)
                continue
            parsed = parse_challenger_output(ch_text)
            q_ok = bool(parsed.question)
            r_ok = bool(parsed.rubric)
            if q_ok and r_ok:
                totals["challenger_fmt_ok"] += 1
                totals["n_parsed"] += 1

            question = parsed.question or ch_text
            rubrics = (
                [parsed.rubric] if parsed.rubric
                else sample.get("rubrics", [])
            )

            # 2) Solver answer.
            solver_msgs = [
                {"role": "system", "content": sv_context},
                {"role": "user", "content": question},
            ]
            try:
                answer = self.solver.generate(
                    solver_msgs,
                    max_new_tokens=eval_new_tokens,
                    temperature=0.2,
                    do_sample=True,
                )
            except Exception as e:
                logger.warning("[Eval] Solver failed on idx=%d: %s", idx, e)
                continue

            # 3) Judge score.
            try:
                s_res = self.reward_fn.compute_solver_reward(
                    answer=answer,
                    rubrics=rubrics,
                    context=context,
                    metadata=meta,
                )
                j_score = float(s_res.correctness)
                s_details = self._json_safe(s_res.details)
            except Exception as e:
                logger.warning("[Eval] Judge failed on idx=%d: %s", idx, e)
                j_score = 0.0
                s_details = {"error": str(e)}

            totals["n"] += 1
            totals["j_score_sum"] += j_score

            rec = {
                "index": idx,
                "metadata": self._json_safe(meta),
                "context_preview": self._truncate_text(context, 4000),
                "challenger_raw": self._truncate_text(ch_text, 8000),
                "question": parsed.question,
                "evidence_span": parsed.evidence_span,
                "rubric": parsed.rubric,
                "answer": answer,
                "j_score": j_score,
                "solver_details": s_details,
                "challenger_format_ok": bool(q_ok and r_ok),
            }
            with open(results_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            if totals["n"] % 10 == 0:
                iterator.set_postfix(
                    j=round(totals["j_score_sum"] / max(totals["n"], 1), 4),
                    fmt=round(totals["challenger_fmt_ok"] / max(totals["n"], 1), 3),
                )

        n = max(totals["n"], 1)
        summary = {
            "tag": tag,
            "num_samples": totals["n"],
            "num_challenger_parsed": totals["n_parsed"],
            "mean_j_score": totals["j_score_sum"] / n,
            "challenger_format_ok_rate": totals["challenger_fmt_ok"] / n,
            "results_file": str(results_path),
        }

        # Best-effort summary dump next to the results file.
        summary_path = self.checkpoint_dir / f"test_summary_{tag}.json"
        try:
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            summary["summary_file"] = str(summary_path)
        except Exception as e:
            logger.warning("[Eval] Could not write summary file: %s", e)

        logger.info(
            "[Eval] Done: n=%d  mean_J=%.4f  fmt_ok=%.3f  → %s",
            summary["num_samples"],
            summary["mean_j_score"],
            summary["challenger_format_ok_rate"],
            results_path,
        )
        return summary

    def _save_checkpoint(self, step: int) -> None:
        for name, model in [("solver", self.solver), ("challenger", self.challenger)]:
            path = self.checkpoint_dir / f"{name}_step_{step}"
            # In LoRA mode this writes just the adapter (~50 MB); in full-FT
            # mode it writes the whole model.
            model.save_adapter(path)
        tag = "adapter" if self.use_lora else "full"
        logger.info("Checkpoint [%s] saved at step %d", tag, step)

    def _sync_references(self) -> None:
        """Periodic ref policy refresh.

        In LoRA mode the reference policy is *always* the frozen base model
        (adapter disabled), so there is nothing to sync — KL is computed
        against the original base, which is the standard RLHF-with-LoRA
        setup. We keep the method as a no-op for API compatibility.
        """
        if self.use_lora:
            return
        cpu_sd = {k: v.cpu() for k, v in self.solver.model.state_dict().items()}
        self.solver_ref.model.load_state_dict(cpu_sd)
        cpu_sd = {k: v.cpu() for k, v in self.challenger.model.state_dict().items()}
        self.challenger_ref.model.load_state_dict(cpu_sd)
        logger.info("Reference models synced (on CPU).")
