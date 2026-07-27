# Failure analysis: the confident-wrong trade-off

The single most important finding from the generation-stage experiments (prompt v2 through v5)
is not an accuracy number -- it's a failure-mode shift that aggregate accuracy hides.

## What happened

Moving from prompt v2 (k=5) to v3 (k=10, more assertive instructions) raised manually-graded
accuracy from ~48.3% to ~58.3%. But four questions that had previously failed safely -- the
model honestly said "I don't have enough information" -- started failing **confidently and
wrong** instead:

- Asserting a cash balance *increased* when it had in fact declined ~42%
- Concluding "gross margin isn't a useful metric" when the filing's own numbers showed a clear,
  computable, improving trend
- Computing an interest coverage ratio using the wrong line item entirely (a positive EBIT
  figure when the correct one was negative)

## Why this matters more than the accuracy number

For a tool aimed at financial analysts, a confidently wrong answer is a worse failure than an
honest "I don't know" -- the former can drive a bad decision silently; the latter prompts a
human to go check the source themselves.

## What we tried, and what actually worked

- **v4** (stricter caution instructions): did NOT reduce the confident-wrong count. It fixed two
  of the four cases but introduced two new ones elsewhere, net accuracy *dropped*. Kept as a
  documented negative result.
- **v5** (few-shot examples targeting the specific errors observed): fixed two of four cleanly,
  improved a third partially. The fourth (Boeing effective tax rate) persisted -- and the reason
  is instructive: the few-shot example addressed *comparison logic* (checking direction on a
  number line), but the actual root cause was a *sign-extraction error* upstream (the model read
  FY2022's rate as -0.6% when the correct figure was +0.62%). No amount of comparison-logic
  coaching fixes a wrong input.

## Takeaway

Prompt iteration without root-cause isolation risks polishing the wrong thing. Each version here
targeted a *specific, previously diagnosed* failure rather than being a general "try to do
better" pass -- and even then, one fix didn't land because the diagnosis needed one more level
of precision (comparison error vs. extraction error) than was initially assumed.
