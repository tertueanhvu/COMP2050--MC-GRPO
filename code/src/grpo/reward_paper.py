"""

Setting: binary correctness + 5-gram repetition penalty.

"""
from __future__ import annotations
import re


def _extract_boxed(text: str) -> str:
    """Walk balanced braces in last \\boxed{...}; return inner content, or empty."""
    if not isinstance(text, str):
        return ""
    matches = list(re.finditer(r"\\boxed\{", text))
    if not matches:
        return ""
    start = matches[-1].end()
    depth, i = 1, start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    return text[start:i - 1].strip() if depth == 0 else ""


def _normalize(ans: str) -> str:
    """Light normalization matching ms-swift baseline's ModeAdaptiveReward."""
    if not ans:
        return ""
    ans = ans.lower().strip().replace(" ", "")
    for sym in ["\\%", "^\\circ", "\\$", "\\pi"]:
        ans = ans.replace(sym, "")
    return ans


def _ngram_repetition_penalty(text: str, n: int = 5, floor: float = -0.5) -> float:
    """Penalty in [floor, 0]. floor if completely repetitive, 0 if all unique n-grams.

    Per baseline: 5-gram, min value -0.5.
    """
    tokens = text.split()
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    if not ngrams:
        return 0.0
    diversity = len(set(ngrams)) / len(ngrams)
    # diversity=1 (all unique) → penalty 0
    # diversity=0 (all duplicate) → penalty = floor
    return floor * (1.0 - diversity)


def paper_reward(
    prompts,
    completions,
    completion_ids=None,
    trainer_state=None,
    log_metric=None,
    log_extra=None,
    *,
    answer,
    mode=None,
    question_id=None,
    **kwargs,
) -> list[float]:
    """TRL v1.4.0-compatible reward function.

    Args:
        prompts: list[str] (standard format) — input prompts
        completions: list[str] — student model rollouts
        completion_ids: list[list[int]] — token IDs (unused here)
        trainer_state: transformers.TrainerState (unused)
        log_metric: callable for scalar logging
        log_extra: callable for column logging
        answer: list[str] — ground-truth "\\boxed{...}" strings (forwarded from dataset)
        mode: list[str] | None — mode labels (forwarded; used only for diagnostic logging)
        question_id: list[int] | None — question IDs (forwarded; unused)
        **kwargs: absorb any future-added kwargs from TRL

    Returns:
        list[float] in [-0.5, 1.0], length == len(completions)
    """
    rewards: list[float] = []
    n_correct = 0

    for c, gt in zip(completions, answer):
        # Handle conversational fallback: completion could be list of dicts
        if isinstance(c, list):
            try:
                c = c[-1]["content"]
            except (KeyError, IndexError, TypeError):
                c = ""

        pred_norm = _normalize(_extract_boxed(c))
        gt_norm = _normalize(_extract_boxed(gt))
        r_acc = 1.0 if (pred_norm and gt_norm and pred_norm == gt_norm) else 0.0
        if r_acc > 0:
            n_correct += 1
        r_pen = _ngram_repetition_penalty(c, n=5, floor=-0.5)
        rewards.append(float(r_acc + r_pen))

    # Diagnostic logging (per-mode breakdown of accuracy)
    if log_metric is not None and mode is not None and len(mode) == len(rewards):
        for m in {"low", "medium", "high"}:
            mode_rewards = [r for r, mm in zip(rewards, mode) if mm == m]
            if mode_rewards:
                log_metric(f"reward_paper/mean_{m}", sum(mode_rewards) / len(mode_rewards))
                # Approx per-mode accuracy = fraction with reward > 0 (correct, even with rep penalty)
                # binary correctness signal: r_acc = (r - r_pen); since r_pen <= 0,
                # r >= 0.5 → definitely correct; r <= 0 → definitely wrong
                n_correct_m = sum(1 for r in mode_rewards if r > 0.5)
                log_metric(f"reward_paper/acc_{m}", n_correct_m / len(mode_rewards))

    return rewards
