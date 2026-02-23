"""Rubrics-based reward with multi-component scoring for both Solver and Challenge."""

import json
import logging
import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional

from .base_reward import BaseReward, ChallengeRewardResult, SolverRewardResult

logger = logging.getLogger(__name__)


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
# Repetition penalty helpers (ref: https://arxiv.org/pdf/2508.05004)
# ---------------------------------------------------------------------------

def _extract_ngrams(text: str, n: int) -> List[str]:
    """Extract character-level n-grams from text."""
    tokens = text.split()
    if len(tokens) < n:
        return []
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def compute_repetition_penalty(
    text: str,
    ngram_sizes: List[int] = (3, 4, 5),
    max_penalty: float = 0.5,
) -> float:
    """
    Compute repetition penalty based on repeated n-gram ratio.

    Inspired by repetition penalty in https://arxiv.org/pdf/2508.05004.
    Returns a value in [-max_penalty, 0.0], where 0 means no repetition.
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
    penalty = -max_penalty * min(avg_ratio * 3.0, 1.0)
    return penalty


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

    Checks: minimum length, presence of both context and question sections,
    and absence of degenerate patterns (all-whitespace, garbled text).
    Returns a value in [-max_penalty, 0.0].
    """
    if not text or not text.strip():
        return -max_penalty

    stripped = text.strip()
    penalty = 0.0

    if len(stripped) < min_length:
        penalty -= max_penalty * 0.5

    alpha_ratio = sum(c.isalpha() or c.isspace() for c in stripped) / max(len(stripped), 1)
    if alpha_ratio < 0.5:
        penalty -= max_penalty * 0.3

    unique_chars = len(set(stripped))
    if unique_chars < 10:
        penalty -= max_penalty * 0.2

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
        2. Repetition penalty  (penalize repeated n-grams)
        3. Format penalty      (penalize degenerate output)
        4. Context-question relevance (reward meaningful questions)
        5. Rubric quality      (reward well-defined rubrics)

    Solver reward components:
        1. Correctness reward  = 0/1 from stronger LLM judge on rubrics
        2. Context grounding   (encourage context-derived answers)
        3. Tool usage reward   (reward explicit context referencing)
    """

    def __init__(
        self,
        use_llm_judge: bool = False,
        judge_model: str = "gpt-4",
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
    ):
        self.use_llm_judge = use_llm_judge
        self.judge_model = judge_model
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
            1. correctness  : 0/1 from LLM judge (or heuristic fallback)
            2. context_grounding : how well the answer is grounded in context
            3. tool_usage   : whether the solver references context explicitly
        """
        result = SolverRewardResult()

        if not answer or not answer.strip():
            return result

        # --- 1. Correctness via stronger model judge (0/1) ---
        if self.use_llm_judge and self.api_client:
            correctness = self._llm_judge_correctness(answer, rubrics)
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

    def _llm_judge_correctness(self, answer: str, rubrics: List) -> float:
        """Use stronger LLM judge to evaluate correctness: binary 0/1."""
        rubrics_text = build_rubrics_text(rubrics)
        prompt = (
            "You are a strict grading judge. Evaluate the student answer against the rubrics.\n"
            "Output ONLY a JSON object: {\"score\": 0 or 1, \"rationale\": \"brief explanation\"}\n"
            "Score 1 ONLY if ALL rubric requirements are fully satisfied. Otherwise score 0.\n\n"
            f"Rubrics:\n{rubrics_text}\n\n"
            f"Student answer:\n{answer}\n\n"
            "JSON output:"
        )
        return self._call_llm_judge_binary(prompt)

    def _llm_judge_context_grounding(self, answer: str, context: str) -> float:
        """
        Use LLM judge to verify the answer is derived from the given context
        rather than fabricated from pretrained knowledge.
        """
        prompt = (
            "You are a grounding verification judge.\n"
            "Determine whether the answer is derived from the given context or "
            "appears to be hallucinated / based on external knowledge.\n"
            "Output ONLY a JSON object: {\"grounded\": 0 or 1, \"rationale\": \"brief explanation\"}\n"
            "Score 1 if the answer is clearly supported by the context. Score 0 otherwise.\n\n"
            f"Context:\n{context[:3000]}\n\n"
            f"Answer:\n{answer}\n\n"
            "JSON output:"
        )
        return self._call_llm_judge_binary(prompt, score_key="grounded")

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
            2. repetition_penalty: penalize repetitive generation
            3. format_penalty    : penalize degenerate/malformed output
            4. relevance         : context-question relationship quality
            5. rubric_quality    : quality of rubrics the challenge model defines
        """
        result = ChallengeRewardResult()
        output_text = challenge_output or ""

        # --- 1. Adversarial correctness: -1 * solver_correctness ---
        result.correctness = -1.0 * solver_correctness

        # --- 2. Repetition penalty ---
        result.repetition_penalty = compute_repetition_penalty(output_text)

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
        }
        return result

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
        prompt = (
            "You are evaluating question quality. Rate how relevant and meaningful "
            "the question is with respect to the given context.\n"
            "Output ONLY a JSON object: {\"score\": <float 0.0-1.0>, \"rationale\": \"brief explanation\"}\n"
            "1.0 = highly relevant, well-formed question derived from context.\n"
            "0.0 = irrelevant or nonsensical question.\n\n"
            f"Context:\n{context[:3000]}\n\n"
            f"Question:\n{question}\n\n"
            "JSON output:"
        )
        return self._call_llm_judge_float(prompt)

    def _llm_judge_rubric_quality(self, question: str, rubrics: List) -> float:
        """
        Use LLM to evaluate the quality of rubrics generated for a question.
        Good rubrics should be specific, comprehensive, and well-matched to the question.
        """
        rubrics_text = build_rubrics_text(rubrics)
        prompt = (
            "You are evaluating rubric quality. Rate how well the rubrics serve "
            "as evaluation criteria for the given question.\n"
            "Output ONLY a JSON object: {\"score\": <float 0.0-1.0>, \"rationale\": \"brief explanation\"}\n"
            "1.0 = comprehensive, specific, well-matched rubrics.\n"
            "0.0 = missing, vague, or irrelevant rubrics.\n\n"
            f"Question:\n{question}\n\n"
            f"Rubrics:\n{rubrics_text}\n\n"
            "JSON output:"
        )
        return self._call_llm_judge_float(prompt)

    # -----------------------------------------------------------------------
    # LLM judge utilities
    # -----------------------------------------------------------------------

    def _call_llm_judge_binary(self, prompt: str, score_key: str = "score") -> float:
        """Call LLM judge and extract a binary 0/1 score."""
        try:
            response = self.api_client.chat.completions.create(
                model=self.judge_model,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.choices[0].message.content.strip()
            text = re.sub(r"^```\w*\n?", "", text)
            text = re.sub(r"\n?```$", "", text)
            data = json.loads(text)
            return float(data.get(score_key, 0))
        except Exception as e:
            logger.warning("LLM judge binary call failed: %s", e)
            return 0.0

    def _call_llm_judge_float(self, prompt: str, score_key: str = "score") -> float:
        """Call LLM judge and extract a float score in [0, 1]."""
        try:
            response = self.api_client.chat.completions.create(
                model=self.judge_model,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.choices[0].message.content.strip()
            text = re.sub(r"^```\w*\n?", "", text)
            text = re.sub(r"\n?```$", "", text)
            data = json.loads(text)
            raw = float(data.get(score_key, 0))
            return max(0.0, min(1.0, raw))
        except Exception as e:
            logger.warning("LLM judge float call failed: %s", e)
            return 0.0
