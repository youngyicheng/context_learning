# CL-bench RL: Technical Implementation Specification

**Version:** 0.1.0  
**Purpose:** Technical specification for implementing the Teacher-Student reinforcement learning framework for context learning, using the [tencent/CL-bench](https://huggingface.co/datasets/tencent/CL-bench) dataset.

---

## 1. Overview

This document describes the implementation of a dual-model RL system:

- **Challenge Model (Teacher):** Provides context and questions to the Solver.
- **Solver Model (Student):** Generates answers based on the provided context.
- **Reward:** Both models receive separate rewards; Solver reward is derived from dataset rubrics; Challenge reward is derived from Solver performance (and optionally context-related factors).

### 1.1 Architecture Diagram

```
┌─────────────────┐     messages,rubrics,metadata     ┌─────────────────┐
│  CL-bench       │ ────────────────────────────────►│  Challenge      │
│  Dataset        │                                    │  Model          │
└─────────────────┘                                    └────────┬────────┘
                                                                 │ context + question
                                                                 ▼
┌─────────────────┐     solver_reward                 ┌─────────────────┐
│  Evaluator      │ ◄────────────────────────────────│  Solver Model   │
│  (Rubrics-based)│                                   │  (RL)           │
└────────┬────────┘                                   └─────────────────┘
         │ challenge_reward
         │
         ▼
┌─────────────────┐
│  Challenge      │
│  Model          │
└─────────────────┘
```

---

## 2. Dataset: CL-bench

**Source:** [https://huggingface.co/datasets/tencent/CL-bench](https://huggingface.co/datasets/tencent/CL-bench)

### 2.1 Sample Structure

Each sample is a JSON object with:

| Field     | Type   | Description                                                |
|----------|--------|------------------------------------------------------------|
| `messages` | list | Multi-turn conversation in OpenAI chat format              |
| `rubrics`  | list | List of evaluation criteria (strings)                      |
| `metadata` | dict | `task_id`, `context_id`, `context_category`, `sub_category` |

Example:

```json
{
  "messages": [
    {"role": "system", "content": "You are an AI designed to ..."},
    {"role": "user", "content": "RULE BOOK ... context and task ..."}
  ],
  "rubrics": [
    "The response should define what a Sighting card is...",
    "The response should name the four different sighting card types..."
  ],
  "metadata": {
    "task_id": "2bbe2e03-972d-45d5-8e54-0654115fddd1",
    "context_id": "71a2cd92-6978-4ea8-a37f-d9972812989",
    "context_category": "Rule System Application",
    "sub_category": "Game Mechanics"
  }
}
```

### 2.2 Loading

```python
from datasets import load_dataset
ds = load_dataset("tencent/CL-bench", split="train")
```

---

## 3. Module Specifications

### 3.1 Data Loader (`clbench_rl/data/loader.py`)

**Class:** `CLBenchDataLoader`

**Responsibilities:**
- Load CL-bench from HuggingFace.
- Expose iteration over samples as `{"messages", "rubrics", "metadata"}`.
- Support `max_samples` for quick validation.
- Support subset and cache directory.

**Interface:**
- `__init__(split, max_samples, subset, cache_dir)`
- `load()` → self
- `__iter__()` → Iterator[dict]
- `get_batch(batch_size)` → Iterator[List[dict]]

---

### 3.2 Models

#### 3.2.1 Challenge Model (`clbench_rl/models/challenge_model.py`)

**Class:** `ChallengeModel`

**Responsibilities:**
- Wrap a HuggingFace causal LM.
- Optionally transform/augment input messages before passing to Solver.
- Generate context+question when not in pass-through mode.

**Interface:**
- `__init__(model_name, device, use_fast)`
- `generate(messages, max_new_tokens, temperature, do_sample, **kwargs)` → str
- `format_prompt(messages)` → List[dict]

**Default:** Pass-through (use raw `messages` as-is). Override `format_prompt` or `generate` for augmentation.

#### 3.2.2 Solver Model (`clbench_rl/models/solver_model.py`)

**Class:** `SolverModel`

**Responsibilities:**
- Wrap a HuggingFace causal LM.
- Generate answers from context+question (in `messages`).
- Optionally return token-level log-probs for RL (REINFORCE/PPO).

**Interface:**
- `__init__(model_name, device, use_fast)`
- `generate(messages, max_new_tokens, temperature, do_sample, return_logprobs, **kwargs)` → str | (str, Tensor)

---

### 3.3 Rewards (`clbench_rl/rewards/`)

#### 3.3.1 Base Interface (`base_reward.py`)

```python
class BaseReward(ABC):
    def compute_solver_reward(self, answer, rubrics, metadata, **kwargs) -> float: ...
    def compute_challenge_reward(self, solver_reward, context, metadata, **kwargs) -> float: ...
```

#### 3.3.2 Rubrics-Based Reward (`rubrics_reward.py`)

**Class:** `RubricsReward(BaseReward)`

**Modes:**
1. **Heuristic:** Placeholder scoring (non-empty, length checks).
2. **LLM-as-judge:** Call OpenAI-compatible API with rubrics + answer; parse score 0/1.

**Extensibility:**
- `sub_category_weights`: Scale rewards by `metadata.sub_category`.
- `challenge_reward_scale`: Scale Challenge reward (default: `solver_reward`).

**Challenge reward:** `solver_reward * challenge_reward_scale`. Can be extended with penalties (e.g., context too trivial, too helpful).

---

### 3.4 RL Environment (`clbench_rl/env/clbench_env.py`)

**Class:** `CLBenchEnv`

**Responsibilities:**
- Implement one episode: `sample → Challenge → Solver → score`.
- Return `EnvStep` with `solver_reward`, `challenge_reward`, `answer`, `challenge_output`, `messages`, `rubrics`, `metadata`.

**Interface:**
- `__init__(challenge_model, solver_model, reward_fn, challenge_pass_through)`
- `step(sample, return_logprobs=False)` → EnvStep

**Data flow:**
1. Extract `messages`, `rubrics`, `metadata` from sample.
2. If `challenge_pass_through`: use `messages` as-is. Else: Challenge generates/transforms → new messages.
3. Solver generates `answer` from messages.
4. Compute `solver_reward` via `reward_fn.compute_solver_reward`.
5. Compute `challenge_reward` via `reward_fn.compute_challenge_reward`.

---

### 3.5 Trainer (`clbench_rl/trainer/reinforce_trainer.py`)

**Class:** `ReinforceTrainer`

**Responsibilities:**
- Load data via `CLBenchDataLoader`.
- Create Challenge, Solver, Reward from config.
- Run `env.step` for each sample.
- Aggregate rewards and report metrics.
- Save checkpoints.

**Interface:**
- `__init__(config, challenge_model, solver_model, reward_fn)`
- `train()` → Dict with `mean_solver_reward`, `mean_challenge_reward`, `num_episodes`

**Config keys:** `data`, `challenge_model`, `solver_model`, `reward`, `training`.

**Note:** Gradient-based REINFORCE/PPO updates require integration with TRL or a custom forward-backward over generated tokens. The current implementation runs the pipeline and collects rewards; parameter updates can be added using TRL's PPOTrainer.

---

### 3.6 Config (`clbench_rl/config/default_config.py`)

**Function:** `get_default_config()` → dict

**Function:** `merge_config(user_config)` → dict

Default model: `Qwen/Qwen2.5-1.5B-Instruct` (suitable for local runs).

---

## 4. Local Model Recommendations

| Model                    | Params | Use Case                      |
|--------------------------|--------|--------------------------------|
| Qwen2.5-1.5B-Instruct    | 1.5B   | Minimal, quick pipeline check |
| Phi-3-mini               | 3.8B   | Better reasoning               |
| Qwen2.5-7B / Mistral-7B  | 7B     | When GPU memory allows         |

**Strategy:** Start with 1.5B–3.8B to verify the pipeline, then scale up.

---

## 5. Reward Extensibility

### 5.1 Context-Related Rewards

To incorporate context into rewards (e.g., per-category or difficulty-aware):

1. Extend `BaseReward` with a new subclass.
2. Use `metadata["context_category"]`, `metadata["sub_category"]` for weighting or bonuses.
3. Optionally use `context` string length or content for difficulty or relevance.

### 5.2 Example: Category-Weighted Reward

```python
class ContextAwareReward(RubricsReward):
    def compute_solver_reward(self, answer, rubrics, metadata, **kwargs):
        base = super().compute_solver_reward(answer, rubrics, metadata, **kwargs)
        sub = metadata.get("sub_category", "")
        weight = {"Game Mechanics": 1.2, "Procedure": 1.0}.get(sub, 1.0)
        return base * weight
```

---

## 6. File Structure

```
llm_research/
├── clbench_rl/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── loader.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── challenge_model.py
│   │   └── solver_model.py
│   ├── env/
│   │   ├── __init__.py
│   │   └── clbench_env.py
│   ├── rewards/
│   │   ├── __init__.py
│   │   ├── base_reward.py
│   │   └── rubrics_reward.py
│   ├── trainer/
│   │   ├── __init__.py
│   │   └── reinforce_trainer.py
│   └── config/
│       ├── __init__.py
│       └── default_config.py
├── scripts/
│   └── run_pipeline.py
├── requirements.txt
└── IMPLEMENTATION.md
```

---

## 7. Running the Pipeline

```bash
# Install dependencies
pip install -r requirements.txt

# Run pipeline with 5 samples (quick validation)
python scripts/run_pipeline.py --max-samples 5 --model Qwen/Qwen2.5-1.5B-Instruct
```

---

## 8. References

- [CL-bench Dataset](https://huggingface.co/datasets/tencent/CL-bench)
- [CL-bench GitHub](https://github.com/Tencent-Hunyuan/CL-bench)
- [CL-bench Paper (arXiv:2602.03587)](https://arxiv.org/abs/2602.03587)
- [TRL (PPO/RLHF)](https://huggingface.co/docs/trl)
