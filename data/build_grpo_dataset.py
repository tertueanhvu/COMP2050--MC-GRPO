""" 

Build mode-aware GRPO dataset from raw GSM8K + MATH train splits.

Output: data/train_grpo_3mode.jsonl
Schema (one JSON object per line):
  {
    "question_id": <int>,                # unique per question, repeats across 3 modes
    "prompt":      "<mode prefix>\\n\\n<format instructions>\\n\\n<question>...",
    "mode":        "low" | "medium" | "high",
    "answer":      "\\boxed{<ground truth>}"
  }

Expected output: 14,973 unique questions × 3 modes = 44,919 rows
  (slightly fewer if MATH \\boxed{} parse fails for some samples)
"""
from __future__ import annotations
import json
import re
from pathlib import Path

from datasets import load_dataset

MODE_PREFIXES: dict[str, str] = {
    "low":    "Respond concisely with minimal reasoning.",
    "medium": "Solve step-by-step.",
    "high":   "Think deeply, verify, and self-correct.",
}

FORMAT_SUFFIX = (
    "\n\nprovide your reasoning in <think> tags, "
    "then put your final answer in \\boxed{{}}\n\n"
    "{question} Please reason step by step, and put your final answer within "
    "\\boxed{{}}. Do NOT include explanations or reasoning in the final answer "
    "- only the numeric value in \\boxed{{}}."
)


def extract_boxed(text: str) -> str:
    """Walk balanced braces in last \\boxed{...}; return inner content, or empty."""
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


def build_prompt(prefix: str, question: str) -> str:
    return prefix + FORMAT_SUFFIX.format(question=question)


def main(output: str = "data/train_grpo_3mode.jsonl") -> None:
    # GSM8K — openai/gsm8k, config "main". Fields: question, answer ("... #### N")
    print("Loading GSM8K train split (openai/gsm8k)...")
    gsm8k = load_dataset("openai/gsm8k", "main", split="train")
    print(f"  GSM8K: {len(gsm8k)} problems")

    # MATH — Maxwell-Jia/MATH (replacement for the taken-down hendrycks/competition_math).
    # Fields: problem, solution, answer, level, type
    print("Loading MATH train split (Maxwell-Jia/MATH)...")
    math = load_dataset("Maxwell-Jia/MATH", split="train")
    print(f"  MATH:  {len(math)} problems")
    print(f"  MATH columns: {math.column_names}")

    rows: list[dict] = []
    qid = 0
    skipped_math = 0

    # GSM8K — answer comes after "####"
    for ex in gsm8k:
        gt = ex["answer"].split("####")[-1].strip().replace(",", "")
        for mode, prefix in MODE_PREFIXES.items():
            rows.append({
                "question_id": qid,
                "prompt":      build_prompt(prefix, ex["question"]),
                "mode":        mode,
                "answer":      f"\\boxed{{{gt}}}",
            })
        qid += 1

    # MATH — prefer explicit 'answer' field; fall back to \boxed{} in 'solution'
    for ex in math:
        gt_raw = ""
        if ex.get("answer"):
            gt_raw = str(ex["answer"]).strip()
        if not gt_raw:
            gt_raw = extract_boxed(ex.get("solution", ""))
        if not gt_raw:
            skipped_math += 1
            continue
        for mode, prefix in MODE_PREFIXES.items():
            rows.append({
                "question_id": qid,
                "prompt":      build_prompt(prefix, ex["problem"]),
                "mode":        mode,
                "answer":      f"\\boxed{{{gt_raw}}}",
            })
        qid += 1

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print(f"\nDone. Wrote {len(rows)} rows ({qid} unique questions × 3 modes) to {output}")
    if skipped_math:
        print(f"  Skipped {skipped_math} MATH samples (no \\boxed{{}} in solution)")

    # Quick mode-distribution check
    from collections import Counter
    modes = Counter(r["mode"] for r in rows)
    print(f"  Mode distribution: {dict(modes)}")


if __name__ == "__main__":
    main()
