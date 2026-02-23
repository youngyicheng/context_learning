"""Rubrics-based reward with multi-component scoring for both Solver and Challenge.

Implements:
  - GPT-4o as judge (Appendix B.3)
  - BLEU-based batch repetition penalty with agglomerative clustering (Appendix B.4)
"""

import json
import logging
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base_reward import BaseReward, ChallengeRewardResult, SolverRewardResult

logger = logging.getLogger(__name__)

try:
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

    _NLTK_AVAILABLE = True
except ImportError:
    _NLTK_AVAILABLE = False
    logger.warning("nltk not available; BLEU-based repetition penalty will use n-gram fallback.")

try:
    from sklearn.cluster import AgglomerativeClustering

    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available; repetition penalty clustering will use n-gram fallback.")


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


# ---------------------------------------------------------------------------
# Repetition penalty (Appendix B.4)
#
# 1. Pairwise distance: d_ij = 1 - BLEU(x_i, x_j)
#    using nltk sentence_bleu with SmoothingFunction().method1
# 2. Agglomerative clustering: metric='precomputed', linkage='average'
# 3. Penalty: r_rep(x_i) = |C_k| / B
# ---------------------------------------------------------------------------

def _bleu_distance(x_i: str, x_j: str) -> float:
    """Compute BLEU-based distance: d_ij = 1 - BLEU(x_i, x_j)."""
    ref = x_i.split()
    hyp = x_j.split()
    if not ref or not hyp:
        return 1.0
    smoothing = SmoothingFunction().method1
    score = sentence_bleu([ref], hyp, smoothing_function=smoothing)
    return 1.0 - score


def compute_bleu_distance_matrix(questions: List[str]) -> np.ndarray:
    """
    Compute pairwise BLEU-based distance matrix for a batch of questions.

    d_ij = 1 - BLEU(x_i, x_j), tokenized by whitespace splitting,
    smoothed with SmoothingFunction().method1.
    """
    n = len(questions)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = _bleu_distance(questions[i], questions[j])
            dist[i, j] = d
            dist[j, i] = d
    return dist


def compute_batch_repetition_penalties(
    questions: List[str],
    distance_threshold: float = 0.5,
) -> List[float]:
    """
    Compute per-question repetition penalty for a batch using BLEU clustering.

    Algorithm (Appendix B.4):
        1. Compute pairwise distance: d_ij = 1 - BLEU(x_i, x_j)
        2. Agglomerative clustering (metric='precomputed', linkage='average')
        3. r_rep(x_i) = |C_k| / B, where C_k is the cluster of x_i, B = batch size

    Penalty semantics: a value near 1/B means the question is unique in the batch;
    a value near 1.0 means the entire batch is one cluster (all similar).

    Args:
        questions: List of question strings in the batch.
        distance_threshold: Clustering distance threshold for agglomerative clustering.

    Returns:
        List of penalty values, one per question. Higher = more repetition.
    """
    n = len(questions)
    if n <= 1:
        return [0.0] * n

    if not (_NLTK_AVAILABLE and _SKLEARN_AVAILABLE):
        return _ngram_fallback_batch_penalties(questions)

    dist_matrix = compute_bleu_distance_matrix(questions)

    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="precomputed",
        linkage="average",
        distance_threshold=distance_threshold,
    )
    labels = clustering.fit_predict(dist_matrix)

    cluster_sizes = Counter(labels)
    penalties = [cluster_sizes[labels[i]] / n for i in range(n)]
    return penalties


def _ngram_fallback_batch_penalties(questions: List[str]) -> List[float]:
    """Fallback when nltk/sklearn are not available: token-overlap based similarity."""
    n = len(questions)
    if n <= 1:
        return [0.0] * n

    token_sets = [set(q.lower().split()) for q in questions]
    penalties = []
    for i in range(n):
        sim_count = 0
        for j in range(n):
            if i == j:
                continue
            if not token_sets[i] or not token_sets[j]:
                continue
            overlap = len(token_sets[i] & token_sets[j])
            union = len(token_sets[i] | token_sets[j])
            if union > 0 and overlap / union > 0.5:
                sim_count += 1
        penalties.append((sim_count + 1) / n)
    return penalties


# ---------------------------------------------------------------------------
# Single-text repetition (used when batch info is unavailable)
# ---------------------------------------------------------------------------

def _extract_ngrams(text: str, n: int) -> List[str]:
    """Extract word-level n-grams from text."""
    tokens = text.split()
    if len(tokens) < n:
        return []
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def compute_single_repetition_penalty(
    text: str,
    ngram_sizes: Tuple[int, ...] = (3, 4, 5),
    max_penalty: float = 0.5,
) -> float:
    """
    Fallback single-text repetition penalty based on repeated n-gram ratio.
    Used only when batch-level BLEU penalty is not applicable.
    Returns a value in [-max_penalty, 0.0].
    """
    if not text or not text.strip():
        return 0.0

    total_ratio = 0.0
    count = 0
    for n in ngram_sizes:
        ngrams = _extract_ngrams(text, n)
        if not ngrams:
            continue
        freq = Counter(ngrams)
        repeated = sum(c - 1 for c in freq.values() if c > 1)
        ratio = repeated / len(ngrams)
        total_ratio += ratio
        count += 1

    if count == 0:
        return 0.0
    avg_ratio = total_ratio / count
    return -max_penalty * min(avg_ratio * 3.0, 1.0)


# ---------------------------------------------------------------------------
# Format check helpers
# ---------------------------------------------------------------------------

def compute_format_penalty(
    text: str,
    min_length: int = 20,
    max_penalty: float = 0.3,
) -> float:
    """
    Penalize poorly formatted challenge output.

    Checks: minimum length, presence of structured sections (<question>,
    <rubric>, <answer> tags), and absence of degenerate patterns.
    Returns a value in [-max_penalty, 0.0].
    """
    if not text or not text.strip():
        return -max_penalty

    stripped = text.strip()
    penalty = 0.0

    if len(stripped) < min_length:
        penalty -= max_penalty * 0.3

    has_question = "<question>" in stripped and "</question>" in stripped
    has_answer = "<answer>" in stripped and "</answer>" in stripped
    if not has_question:
        penalty -= max_penalty * 0.35
    if not has_answer:
        penalty -= max_penalty * 0.2

    alpha_ratio = sum(c.isalpha() or c.isspace() for c in stripped) / max(len(stripped), 1)
    if alpha_ratio < 0.3:
        penalty -= max_penalty * 0.15

    return max(penalty, -max_penalty)


# ---------------------------------------------------------------------------
# Context-question relevance
# ---------------------------------------------------------------------------

def compute_context_question_relevance_heuristic(
    context: str,
    question: str,
) -> float:
    """
    Heuristic measure of context-question relevance via token overlap.
    Returns a value in [0.0, 1.0].
    """
    if not context or not question:
        return 0.0

    ctx_tokens = set(context.lower().split())
    q_tokens = set(question.lower().split())

    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "in", "on", "at",
        "to", "for", "of", "and", "or", "but", "with", "this", "that",
        "it", "be", "as", "by", "from", "not", "do", "does", "did",
        "what", "which", "who", "how", "when", "where", "why",
    }
    ctx_tokens -= stop_words
    q_tokens -= stop_words

    if not q_tokens:
        return 0.0

    overlap = ctx_tokens & q_tokens
    return len(overlap) / len(q_tokens)


# ---------------------------------------------------------------------------
# Context grounding (Solver answer grounded in context, not pretrained knowledge)
# ---------------------------------------------------------------------------

def compute_context_grounding_heuristic(
    answer: str,
    context: str,
    threshold: float = 0.15,
) -> float:
    """
    Heuristic check whether the solver's answer is grounded in the context
    rather than hallucinated from pretrained knowledge.

    Measures overlap of answer content tokens with context tokens.
    Returns 1.0 if well-grounded, scaled down toward 0.0 otherwise.
    """
    if not answer or not context:
        return 0.0

    ans_tokens = set(answer.lower().split())
    ctx_tokens = set(context.lower().split())

    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "in", "on", "at",
        "to", "for", "of", "and", "or", "but", "with", "this", "that",
        "it", "be", "as", "by", "from", "not", "i", "my", "your",
    }
    ans_tokens -= stop_words
    ctx_tokens -= stop_words

    if not ans_tokens:
        return 0.0

    overlap = ans_tokens & ctx_tokens
    ratio = len(overlap) / len(ans_tokens)

    if ratio >= threshold:
        return min(ratio / 0.5, 1.0)
    return ratio / threshold * 0.5


# ---------------------------------------------------------------------------
# Tool usage detection (Solver referencing or using context via tools)
# ---------------------------------------------------------------------------

TOOL_USE_PATTERNS = [
    r"\b(according to|based on|from the|the (context|passage|text|document) (says|states|mentions|indicates|shows))",
    r"\b(as (stated|mentioned|described|shown) in)",
    r"\b(referring to|reference to|cited from)",
    r"\b(the (given|provided) (information|data|context|text))",
    r"<tool_call>.*?</tool_call>",
    r"\[tool\].*?\[/tool\]",
    r"```tool\b",
]


def detect_tool_usage(answer: str) -> float:
    """
    Detect if the solver's answer shows evidence of tool usage or explicit
    context referencing. Returns a score in [0.0, 1.0].
    """
    if not answer:
        return 0.0

    matches = 0
    for pattern in TOOL_USE_PATTERNS:
        if re.search(pattern, answer, re.IGNORECASE | re.DOTALL):
            matches += 1

    return min(matches / 3.0, 1.0)


# ===========================================================================
# Main reward class
# ===========================================================================

class RubricsReward(BaseReward):
    """
    Multi-component reward based on CL-bench rubrics.

    Challenge reward components:
        1. Correctness reward  = -1 * solver_correctness (adversarial)
        2. Repetition penalty  (BLEU-clustering batch penalty, Appendix B.4)
        3. Format penalty      (penalize degenerate output)
        4. Context-question relevance (reward meaningful questions)
        5. Rubric quality      (reward well-defined rubrics)

    Solver reward components:
        1. Correctness reward  = 0/1 from GPT-4o judge (Appendix B.3)
        2. Context grounding   (encourage context-derived answers)
        3. Tool usage reward   (reward explicit context referencing)
    """

    def __init__(
        self,
        use_llm_judge: bool = False,
        judge_model: str = "gpt-4o",
        judge_temperature: float = 0.1,
        api_client=None,
        sub_category_weights: Optional[Dict[str, float]] = None,
        # Challenge reward weights
        challenge_correctness_weight: float = 1.0,
        repetition_penalty_weight: float = 0.3,
        format_penalty_weight: float = 0.2,
        relevance_weight: float = 0.3,
        rubric_quality_weight: float = 0.2,
        # Solver reward weights
        solver_correctness_weight: float = 1.0,
        context_grounding_weight: float = 0.3,
        tool_usage_weight: float = 0.2,
        # Repetition penalty clustering params
        bleu_distance_threshold: float = 0.5,
    ):
        self.use_llm_judge = use_llm_judge
        self.judge_model = judge_model
        self.judge_temperature = judge_temperature
        self.api_client = api_client
        self.sub_category_weights = sub_category_weights or {}

        self.challenge_correctness_weight = challenge_correctness_weight
        self.repetition_penalty_weight = repetition_penalty_weight
        self.format_penalty_weight = format_penalty_weight
        self.relevance_weight = relevance_weight
        self.rubric_quality_weight = rubric_quality_weight

        self.solver_correctness_weight = solver_correctness_weight
        self.context_grounding_weight = context_grounding_weight
        self.tool_usage_weight = tool_usage_weight

        self.bleu_distance_threshold = bleu_distance_threshold

    # -----------------------------------------------------------------------
    # Solver Reward
    # -----------------------------------------------------------------------

    def compute_solver_reward(
        self,
        answer: str,
        rubrics: List[str],
        context: str,
        metadata: Dict[str, Any],
        **kwargs,
    ) -> SolverRewardResult:
        """
        Compute multi-component solver reward.

        Components:
            1. correctness  : 0/1 from GPT-4o judge (B.3) or heuristic fallback
            2. context_grounding : how well the answer is grounded in context
            3. tool_usage   : whether the solver references context explicitly
        """
        result = SolverRewardResult()

        if not answer or not answer.strip():
            return result

        ground_truth = kwargs.get("ground_truth", "")

        # --- 1. Correctness via GPT-4o judge (B.3) ---
        if self.use_llm_judge and self.api_client:
            if ground_truth:
                correctness = self._gpt4o_judge_correctness(answer, ground_truth)
            else:
                correctness = self._llm_judge_correctness_rubrics(answer, rubrics)
        else:
            correctness = self._heuristic_grade(answer, rubrics)
        result.correctness = correctness

        # --- 2. Context grounding reward ---
        grounding = compute_context_grounding_heuristic(answer, context)
        if self.use_llm_judge and self.api_client:
            llm_grounding = self._llm_judge_context_grounding(answer, context)
            grounding = 0.5 * grounding + 0.5 * llm_grounding
        result.context_grounding = grounding

        # --- 3. Tool usage / context referencing reward ---
        result.tool_usage = detect_tool_usage(answer)

        # --- Weighted total ---
        sub = metadata.get("sub_category", "")
        cat_weight = self.sub_category_weights.get(sub, 1.0)

        result.total = cat_weight * (
            self.solver_correctness_weight * result.correctness
            + self.context_grounding_weight * result.context_grounding
            + self.tool_usage_weight * result.tool_usage
        )

        result.details = {
            "correctness_raw": correctness,
            "grounding_raw": grounding,
            "tool_usage_raw": result.tool_usage,
            "sub_category": sub,
            "cat_weight": cat_weight,
            "judge_mode": "gpt4o" if (self.use_llm_judge and self.api_client) else "heuristic",
        }
        return result

    def _heuristic_grade(self, answer: str, rubrics: List) -> float:
        """Rule-based fallback: keyword matching against rubrics."""
        if not answer or len(answer.strip()) < 10:
            return 0.0

        valid_rubrics = [str(r).strip().lower() for r in rubrics if str(r).strip()]
        if not valid_rubrics:
            return 1.0 if len(answer) > 50 else 0.5

        answer_lower = answer.lower()
        matched = sum(
            1 for r in valid_rubrics
            if any(kw in answer_lower for kw in r.split() if len(kw) > 3)
        )
        return min(matched / max(len(valid_rubrics), 1), 1.0)

    def _gpt4o_judge_correctness(self, answer: str, ground_truth: str) -> float:
        """
        GPT-4o as judge for answer correctness (adapted from Appendix B.3).

        Generalized for any domain — compares the candidate answer against
        a reference ground truth and returns binary Yes/No.

        Returns: 1.0 if "Yes", 0.0 if "No" or error.
        """
        try:
            response = self.api_client.chat.completions.create(
                model=self.judge_model,
                temperature=self.judge_temperature,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an answer correctness checker.",
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Hi, there is an answer: {answer},\n"
                            f"and the ground truth answer is: {ground_truth},\n"
                            "please check whether the answer is correct or not, "
                            "and return the **only** Yes or No."
                        ),
                    },
                ],
            )
            text = response.choices[0].message.content.strip().lower()
            if "yes" in text:
                return 1.0
            return 0.0
        except Exception as e:
            logger.warning("GPT-4o judge correctness call failed: %s", e)
            return 0.0

    def _llm_judge_correctness_rubrics(self, answer: str, rubrics: List) -> float:
        """Fallback LLM judge when ground_truth is not available: evaluate via rubrics."""
        rubrics_text = build_rubrics_text(rubrics)
        try:
            response = self.api_client.chat.completions.create(
                model=self.judge_model,
                temperature=self.judge_temperature,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a strict grading judge.",
                    },
                    {
                        "role": "user",
                        "content": (
                            "Evaluate the student answer against the rubrics.\n"
                            "Output ONLY a JSON object: "
                            "{\"score\": 0 or 1, \"rationale\": \"brief explanation\"}\n"
                            "Score 1 ONLY if ALL rubric requirements are fully satisfied.\n\n"
                            f"Rubrics:\n{rubrics_text}\n\n"
                            f"Student answer:\n{answer}\n\n"
                            "JSON output:"
                        ),
                    },
                ],
            )
            text = response.choices[0].message.content.strip()
            text = re.sub(r"^```\w*\n?", "", text)
            text = re.sub(r"\n?```$", "", text)
            data = json.loads(text)
            return float(data.get("score", 0))
        except Exception as e:
            logger.warning("LLM judge rubrics call failed: %s", e)
            return 0.0

    def _llm_judge_context_grounding(self, answer: str, context: str) -> float:
        """
        Use LLM judge to verify the answer is derived from the given context
        rather than fabricated from pretrained knowledge.
        """
        try:
            response = self.api_client.chat.completions.create(
                model=self.judge_model,
                temperature=self.judge_temperature,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a grounding verification judge.",
                    },
                    {
                        "role": "user",
                        "content": (
                            "Determine whether the answer is derived from the given "
                            "context or appears to be hallucinated / based on external "
                            "knowledge.\n"
                            "Output ONLY a JSON object: "
                            "{\"grounded\": 0 or 1, \"rationale\": \"brief explanation\"}\n"
                            "Score 1 if the answer is clearly supported by the context.\n\n"
                            f"Context:\n{context[:3000]}\n\n"
                            f"Answer:\n{answer}\n\n"
                            "JSON output:"
                        ),
                    },
                ],
            )
            text = response.choices[0].message.content.strip()
            text = re.sub(r"^```\w*\n?", "", text)
            text = re.sub(r"\n?```$", "", text)
            data = json.loads(text)
            return float(data.get("grounded", 0))
        except Exception as e:
            logger.warning("LLM judge grounding call failed: %s", e)
            return 0.0

    # -----------------------------------------------------------------------
    # Challenge Reward
    # -----------------------------------------------------------------------

    def compute_challenge_reward(
        self,
        solver_correctness: float,
        challenge_output: str,
        context: str,
        question: str,
        rubrics: List[str],
        metadata: Dict[str, Any],
        **kwargs,
    ) -> ChallengeRewardResult:
        """
        Compute multi-component challenge reward.

        Components:
            1. correctness       : -1 * solver_correctness (adversarial incentive)
            2. repetition_penalty: BLEU-clustering batch penalty (B.4) or single-text fallback
            3. format_penalty    : penalize degenerate/malformed output
            4. relevance         : context-question relationship quality
            5. rubric_quality    : quality of rubrics the challenge model defines

        The repetition_penalty can be overridden via kwargs['batch_repetition_penalty']
        when the trainer computes it at batch level.
        """
        result = ChallengeRewardResult()
        output_text = challenge_output or ""

        # --- 1. Adversarial correctness: -1 * solver_correctness ---
        result.correctness = -1.0 * solver_correctness

        # --- 2. Repetition penalty ---
        batch_rep = kwargs.get("batch_repetition_penalty")
        if batch_rep is not None:
            result.repetition_penalty = -abs(batch_rep)
        else:
            result.repetition_penalty = compute_single_repetition_penalty(output_text)

        # --- 3. Format check penalty ---
        result.format_penalty = compute_format_penalty(output_text)

        # --- 4. Context-question relevance ---
        if self.use_llm_judge and self.api_client:
            result.relevance = self._llm_judge_relevance(context, question)
        else:
            result.relevance = compute_context_question_relevance_heuristic(
                context, question
            )

        # --- 5. Rubric quality ---
        if self.use_llm_judge and self.api_client:
            result.rubric_quality = self._llm_judge_rubric_quality(
                question, rubrics
            )
        else:
            result.rubric_quality = self._heuristic_rubric_quality(rubrics)

        # --- Weighted total ---
        result.total = (
            self.challenge_correctness_weight * result.correctness
            + self.repetition_penalty_weight * result.repetition_penalty
            + self.format_penalty_weight * result.format_penalty
            + self.relevance_weight * result.relevance
            + self.rubric_quality_weight * result.rubric_quality
        )

        result.details = {
            "solver_correctness_input": solver_correctness,
            "output_length": len(output_text),
            "used_batch_repetition": batch_rep is not None,
        }
        return result

    def compute_batch_repetition(self, questions: List[str]) -> List[float]:
        """
        Compute batch-level repetition penalties (Appendix B.4).

        Algorithm:
            1. Pairwise distance: d_ij = 1 - BLEU(x_i, x_j)
            2. Agglomerative clustering (precomputed, average linkage)
            3. r_rep(x_i) = |C_k| / B

        Args:
            questions: All questions generated in the current batch.

        Returns:
            Per-question penalty values.
        """
        return compute_batch_repetition_penalties(
            questions,
            distance_threshold=self.bleu_distance_threshold,
        )

    def _heuristic_rubric_quality(self, rubrics: List) -> float:
        """
        Heuristic evaluation of rubric quality.
        Good rubrics: multiple criteria, each with sufficient detail.
        """
        valid = [str(r).strip() for r in rubrics if str(r).strip()]
        if not valid:
            return 0.0

        count_score = min(len(valid) / 3.0, 1.0)

        avg_len = sum(len(r) for r in valid) / len(valid)
        detail_score = min(avg_len / 50.0, 1.0)

        diversity = len(set(r.lower() for r in valid)) / len(valid)

        return (count_score + detail_score + diversity) / 3.0

    def _llm_judge_relevance(self, context: str, question: str) -> float:
        """Use LLM to evaluate how well the question relates to the context."""
        try:
            response = self.api_client.chat.completions.create(
                model=self.judge_model,
                temperature=self.judge_temperature,
                messages=[
                    {
                        "role": "system",
                        "content": "You are evaluating question quality.",
                    },
                    {
                        "role": "user",
                        "content": (
                            "Rate how relevant and meaningful the question is "
                            "with respect to the given context.\n"
                            "Output ONLY a JSON object: "
                            "{\"score\": <float 0.0-1.0>, \"rationale\": \"brief explanation\"}\n"
                            "1.0 = highly relevant, well-formed question.\n"
                            "0.0 = irrelevant or nonsensical.\n\n"
                            f"Context:\n{context[:3000]}\n\n"
                            f"Question:\n{question}\n\n"
                            "JSON output:"
                        ),
                    },
                ],
            )
            text = response.choices[0].message.content.strip()
            text = re.sub(r"^```\w*\n?", "", text)
            text = re.sub(r"\n?```$", "", text)
            data = json.loads(text)
            raw = float(data.get("score", 0))
            return max(0.0, min(1.0, raw))
        except Exception as e:
            logger.warning("LLM judge relevance call failed: %s", e)
            return 0.0

    def _llm_judge_rubric_quality(self, question: str, rubrics: List) -> float:
        """
        Use LLM to evaluate the quality of rubrics generated for a question.
        Good rubrics should be specific, comprehensive, and well-matched to the question.
        """
        rubrics_text = build_rubrics_text(rubrics)
        try:
            response = self.api_client.chat.completions.create(
                model=self.judge_model,
                temperature=self.judge_temperature,
                messages=[
                    {
                        "role": "system",
                        "content": "You are evaluating rubric quality.",
                    },
                    {
                        "role": "user",
                        "content": (
                            "Rate how well the rubrics serve as evaluation criteria "
                            "for the given question.\n"
                            "Output ONLY a JSON object: "
                            "{\"score\": <float 0.0-1.0>, \"rationale\": \"brief explanation\"}\n"
                            "1.0 = comprehensive, specific, well-matched rubrics.\n"
                            "0.0 = missing, vague, or irrelevant rubrics.\n\n"
                            f"Question:\n{question}\n\n"
                            f"Rubrics:\n{rubrics_text}\n\n"
                            "JSON output:"
                        ),
                    },
                ],
            )
            text = response.choices[0].message.content.strip()
            text = re.sub(r"^```\w*\n?", "", text)
            text = re.sub(r"\n?```$", "", text)
            data = json.loads(text)
            raw = float(data.get("score", 0))
            return max(0.0, min(1.0, raw))
        except Exception as e:
            logger.warning("LLM judge rubric quality call failed: %s", e)
            return 0.0
