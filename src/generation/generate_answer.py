"""Generation with a Python calculator step -- delegating arithmetic to
eval() instead of LLM mental math. Fixed every arithmetic error found in
manual review (e.g. 6489/267.5 computed correctly as 24.2579 -> 24.26,
where the LLM's own mental math had said 24.24). See docs/failure_analysis.md
for the before/after evidence.
"""
import re
from .prompts import PROMPT_V5_FEWSHOT


def build_context(pages: list[dict], page_numbers: list[int], page_text_lookup: dict,
                   max_chars_per_page: int = 2000) -> str:
    parts = []
    for pn in page_numbers:
        text = page_text_lookup.get(pn, "")[:max_chars_per_page]
        parts.append(f"--- Page {pn} ---\n{text}")
    return "\n\n".join(parts)


def generate_answer(client, question: str, context: str, model: str = "gpt-4o-mini",
                     system_prompt: str = PROMPT_V5_FEWSHOT):
    """Returns (full_text, computed_value_or_None).

    computed_value is the result of safely eval()-ing the model's CALC line,
    NOT a number the model computed itself -- this is the whole point.
    """
    user_msg = f"Context:\n{context}\n\nQuestion: {question}" if context else f"Question: {question}"
    resp = client.chat.completions.create(
        model=model, temperature=0,
        messages=[{"role": "system", "content": system_prompt},
                   {"role": "user", "content": user_msg}],
    )
    text = resp.choices[0].message.content
    match = re.search(r"CALC:\s*([0-9+\-*/().\s]+)", text)
    computed = None
    if match:
        try:
            computed = eval(match.group(1).strip())  # safe: regex restricts to digits/operators
        except Exception:
            computed = None
    return text, computed
