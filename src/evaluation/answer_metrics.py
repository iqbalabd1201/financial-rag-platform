"""Answer-correctness evaluation: numeric tolerance matching (deterministic)
plus LLM-as-judge for open-ended answers.

IMPORTANT: an earlier version of the judge-parsing function here had a bug --
`"correct" in verdict.lower()` also matches inside the string "incorrect",
silently marking every wrong answer as right. This produced an invalid 88.3%
result that was only caught by manual spot-checking. The fix below checks
"incorrect" FIRST. This is left as an explicit lesson in the docstring
because it's an easy mistake to reintroduce.
"""
import re


def parse_gold_numeric(gold_answer: str):
    """Return a float if gold_answer is a bare number/currency/percent string,
    else None (meaning: treat as a qualitative answer for LLM-judge instead).
    """
    s = str(gold_answer).strip()
    if not re.match(r"^\$?-?[\d,]+\.?\d*\s*%?$", s):
        return None
    cleaned = s.replace("$", "").replace(",", "").replace("%", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


def numeric_match(gold_answer: str, computed_value, tol: float = 0.05) -> bool | None:
    """True/False if gold_answer is numeric, else None (not applicable).

    Matches on either (a) correct rounding to the gold answer's own decimal
    precision, or (b) 5% relative tolerance -- whichever is more lenient,
    since gold answers themselves sometimes carry small rounding noise
    (a documented FinanceBench data-quality issue, not a pipeline bug).
    """
    gold = parse_gold_numeric(gold_answer)
    if gold is None or computed_value is None:
        return None
    decimals = len(gold_answer.split(".")[-1].rstrip("%")) if "." in gold_answer else 0
    rounded = round(computed_value, decimals)
    if abs(rounded - gold) < 10 ** (-decimals) / 2 + 1e-9:
        return True
    return abs(computed_value - gold) / max(abs(gold), 1e-9) <= tol


def parse_llm_judge_verdict(verdict: str) -> bool | None:
    """Check 'incorrect' BEFORE 'correct' -- see module docstring for why."""
    v = verdict.strip().lower()
    if "incorrect" in v:
        return False
    if "correct" in v:
        return True
    return None


JUDGE_SYSTEM_PROMPT = """You are grading a financial QA system. Compare the generated answer to the gold
answer. Output ONLY one word: "correct" if the generated answer's core claim matches the gold
answer's meaning (minor wording differences OK), or "incorrect" if it contradicts, hedges away
from, or misses the gold answer's key point."""


def llm_judge(client, question: str, gold_answer: str, generated_text: str,
              model: str = "gpt-4o-mini") -> bool | None:
    resp = client.chat.completions.create(
        model=model, temperature=0,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {question}\nGold: {gold_answer}\nGenerated: {generated_text}\n\nVerdict:"},
        ],
    )
    return parse_llm_judge_verdict(resp.choices[0].message.content)
