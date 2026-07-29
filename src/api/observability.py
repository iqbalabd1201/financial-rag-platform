"""Observability: structured request logging + in-memory metrics.

Deliberately avoids adding a Langfuse/LangSmith dependency (another signup,
another API key to manage) in favor of two things that need zero external
setup:
  1. Structured JSON logs to stdout -- Railway (and Render, and Docker
     generally) captures stdout automatically and shows it in the
     Deployments/Logs tab, so this is "tracing" with no new service.
  2. An in-memory /metrics endpoint for aggregate stats since last restart.

Trade-off, stated plainly: metrics reset on every redeploy/restart since
they're in-memory, not persisted. Acceptable for a portfolio demo; a real
production system would ship these to a time-series store instead.
"""
import json
import logging
import sys
import threading
import time

# gpt-4o-mini pricing as of mid-2026: $0.15 / 1M input tokens, $0.60 / 1M output tokens.
# Source: platform.openai.com/pricing -- re-check before treating this as current.
PRICE_PER_1M_INPUT = 0.15
PRICE_PER_1M_OUTPUT = 0.60


def setup_logging():
    """One JSON object per line on stdout -- greppable, and Railway's log
    viewer renders it as-is. Call once at startup."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    root = logging.getLogger("rag_api")
    root.setLevel(logging.INFO)
    root.handlers = [handler]
    root.propagate = False
    return root


def estimate_cost_usd(prompt_tokens: int, completion_tokens: int) -> float:
    return (prompt_tokens / 1_000_000) * PRICE_PER_1M_INPUT + \
           (completion_tokens / 1_000_000) * PRICE_PER_1M_OUTPUT


def log_query_event(logger: logging.Logger, *, question: str, doc_id: str | None,
                     doc_match_method: str, retrieved_pages: list[int],
                     retrieval_ms: float, generation_ms: float, total_ms: float,
                     prompt_tokens: int | None, completion_tokens: int | None,
                     status: str):
    event = {
        "event": "query",
        "status": status,
        "question_preview": question[:80],
        "doc_id": doc_id,
        "doc_match_method": doc_match_method,
        "n_retrieved_pages": len(retrieved_pages),
        "retrieval_ms": round(retrieval_ms, 1),
        "generation_ms": round(generation_ms, 1),
        "total_ms": round(total_ms, 1),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "estimated_cost_usd": round(estimate_cost_usd(prompt_tokens, completion_tokens), 6)
        if prompt_tokens is not None and completion_tokens is not None else None,
    }
    logger.info(json.dumps(event))


class MetricsTracker:
    """Thread-safe in-memory aggregate. Reset on process restart -- see
    module docstring for why that trade-off is acceptable here."""

    def __init__(self):
        self._lock = threading.Lock()
        self.started_at = time.time()
        self.total_requests = 0
        self.total_errors = 0
        self.total_retrieval_ms = 0.0
        self.total_generation_ms = 0.0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def record(self, *, retrieval_ms: float, generation_ms: float,
               prompt_tokens: int, completion_tokens: int, error: bool = False):
        with self._lock:
            self.total_requests += 1
            if error:
                self.total_errors += 1
            self.total_retrieval_ms += retrieval_ms
            self.total_generation_ms += generation_ms
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens

    def snapshot(self) -> dict:
        with self._lock:
            n = max(self.total_requests, 1)
            return {
                "uptime_seconds": round(time.time() - self.started_at, 1),
                "total_requests": self.total_requests,
                "total_errors": self.total_errors,
                "avg_retrieval_ms": round(self.total_retrieval_ms / n, 1),
                "avg_generation_ms": round(self.total_generation_ms / n, 1),
                "total_prompt_tokens": self.total_prompt_tokens,
                "total_completion_tokens": self.total_completion_tokens,
                "estimated_total_cost_usd": round(
                    estimate_cost_usd(self.total_prompt_tokens, self.total_completion_tokens), 4
                ),
            }
