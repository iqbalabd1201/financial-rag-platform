"""Unit tests for the Python-eval calculator step in generate_answer().

Uses a fake OpenAI client (no real API call, no cost) to isolate the
CALC-line extraction and eval() logic -- the part that fixes LLM mental-math
errors like the 24.24 vs 24.2579->24.26 case documented in
docs/failure_analysis.md.
"""
from src.generation.generate_answer import generate_answer


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeUsage:
    def __init__(self, prompt_tokens, completion_tokens):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = prompt_tokens + completion_tokens


class _FakeResponse:
    def __init__(self, content, prompt_tokens=100, completion_tokens=20):
        self.choices = [_FakeChoice(content)]
        self.usage = _FakeUsage(prompt_tokens, completion_tokens)


class _FakeClient:
    """Stands in for an OpenAI client -- returns a fixed response regardless
    of input, so these tests never touch the network or cost anything."""
    def __init__(self, content):
        self._content = content
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        return _FakeResponse(self._content)


def test_calc_line_evaluated_correctly():
    client = _FakeClient("The answer is 24.26.\n\nCALC: 6489/267.5")
    text, computed = generate_answer(client, "some question", "some context")
    assert computed == 6489 / 267.5  # eval() returns the exact float, unrounded


def test_no_calc_line_returns_none():
    client = _FakeClient("The filing does not mention this metric.")
    text, computed = generate_answer(client, "some question", "some context")
    assert computed is None


def test_malformed_calc_expression_fails_safely():
    """eval() on a broken expression should be caught, not raise."""
    client = _FakeClient("CALC: 5 + ")
    text, computed = generate_answer(client, "some question", "some context")
    assert computed is None


def test_usage_out_populated_when_provided():
    """usage_out is optional and additive -- omitting it (as run_eval.py and
    reproduce_generation.py do) must not change behavior; passing it should
    fill in token counts without altering the (text, computed) return."""
    client = _FakeClient("CALC: 10*2")
    usage = {}
    text, computed = generate_answer(
        client, "some question", "some context", usage_out=usage
    )
    assert computed == 20
    assert usage["prompt_tokens"] == 100
    assert usage["completion_tokens"] == 20


def test_usage_out_omitted_does_not_error():
    """The exact call signature used by run_eval.py / reproduce_generation.py
    -- must keep working unchanged."""
    client = _FakeClient("CALC: 3+3")
    text, computed = generate_answer(client, "some question", "some context")
    assert computed == 6
