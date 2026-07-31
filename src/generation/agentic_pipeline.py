"""Agentic retry loop built with LangGraph, on top of the existing
retrieve/generate functions -- nothing in src/retrieval or
src/generation/generate_answer.py is modified.

Graph:
    retrieve -> generate -> assess --sufficient--> END
                               \\--insufficient--> retrieve (with a
                                                     follow-up query, up
                                                     to MAX_RETRIES)

The "assess" step is a second LLM call that judges whether the answer
just generated is sufficient, and if not, proposes a specific follow-up
search query -- this is what makes the retry targeted rather than a
blind re-fetch of the same pages. Retrieved pages accumulate across
retries (the context grows), rather than replacing each other, since a
multi-hop question typically needs evidence from more than one page/area
combined, not a single better-matched page.

Capped at MAX_RETRIES to bound cost -- an ungated retry loop on a flaky
"sufficient" judgment could loop indefinitely.
"""
import json
from typing import TypedDict, Optional

from langgraph.graph import StateGraph, END

from src.indexing.embedder import embed_query
from src.retrieval.retriever import retrieve_pages
from src.generation.generate_answer import build_context, generate_answer

MAX_RETRIES = 2
RETRIEVAL_K = 10

ASSESS_PROMPT = """Question: {question}

Answer given: {answer}

Does this answer fully and confidently address the question using only the
information actually present in the context, without guessing? Reply with
JSON only, no other text:
{{"sufficient": true or false, "follow_up_query": "<a specific, different search query that would help find the missing information, or null if sufficient>"}}"""


class AgentState(TypedDict):
    question: str
    doc_id: str
    retrieved_pages: list
    context: str
    answer: str
    computed_value: Optional[float]
    sufficient: bool
    follow_up_query: Optional[str]
    retry_count: int
    total_prompt_tokens: int
    total_completion_tokens: int


def _make_retrieve_node(embed_model, index, pages, page_text_lookup_by_doc):
    def retrieve_node(state: AgentState) -> AgentState:
        query = state.get("follow_up_query") or state["question"]
        q_emb = embed_query(embed_model, query)
        new_pages = retrieve_pages(q_emb, index, pages, state["doc_id"], k=RETRIEVAL_K)
        all_pages = sorted(set(state.get("retrieved_pages", [])) | set(new_pages))
        # BUG FIX: page_text_lookup_by_doc is {doc_id: {page_num: text}} --
        # must be narrowed to this question's doc_id before passing to
        # build_context, which expects a flat {page_num: text} dict.
        # Previously this was passed through un-narrowed, so every lookup
        # silently returned "" and every generation ran on blank context.
        page_texts = page_text_lookup_by_doc.get(state["doc_id"], {})
        context = build_context(pages, all_pages, page_texts)
        return {**state, "retrieved_pages": all_pages, "context": context}
    return retrieve_node


def _make_generate_node(client):
    def generate_node(state: AgentState) -> AgentState:
        usage = {}
        answer_text, computed = generate_answer(
            client, state["question"], state["context"], usage_out=usage
        )
        return {
            **state, "answer": answer_text, "computed_value": computed,
            "total_prompt_tokens": state.get("total_prompt_tokens", 0) + usage.get("prompt_tokens", 0),
            "total_completion_tokens": state.get("total_completion_tokens", 0) + usage.get("completion_tokens", 0),
        }
    return generate_node


def _make_assess_node(client):
    def assess_node(state: AgentState) -> AgentState:
        if state["retry_count"] >= MAX_RETRIES:
            # Retry budget exhausted -- accept the current answer as final
            # rather than judge it, to guarantee the loop terminates.
            return {**state, "sufficient": True, "follow_up_query": None}

        prompt = ASSESS_PROMPT.format(question=state["question"], answer=state["answer"])
        resp = client.chat.completions.create(
            model="gpt-4o-mini", temperature=0,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        try:
            result = json.loads(resp.choices[0].message.content)
        except (json.JSONDecodeError, KeyError):
            result = {"sufficient": True, "follow_up_query": None}

        usage = resp.usage
        return {
            **state,
            "sufficient": bool(result.get("sufficient", True)),
            "follow_up_query": result.get("follow_up_query"),
            "retry_count": state["retry_count"] + 1,
            "total_prompt_tokens": state.get("total_prompt_tokens", 0) + (usage.prompt_tokens if usage else 0),
            "total_completion_tokens": state.get("total_completion_tokens", 0) + (usage.completion_tokens if usage else 0),
        }
    return assess_node


def _should_retry(state: AgentState) -> str:
    if state["sufficient"] or not state.get("follow_up_query"):
        return "end"
    return "retrieve"


def build_agentic_graph(client, embed_model, index, pages, page_text_lookup_by_doc):
    graph = StateGraph(AgentState)
    graph.add_node("retrieve", _make_retrieve_node(embed_model, index, pages, page_text_lookup_by_doc))
    graph.add_node("generate", _make_generate_node(client))
    graph.add_node("assess", _make_assess_node(client))

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "assess")
    graph.add_conditional_edges("assess", _should_retry, {"retrieve": "retrieve", "end": END})

    return graph.compile()


def run_agentic_query(compiled_graph, question: str, doc_id: str) -> dict:
    initial_state: AgentState = {
        "question": question, "doc_id": doc_id, "retrieved_pages": [], "context": "",
        "answer": "", "computed_value": None, "sufficient": False, "follow_up_query": None,
        "retry_count": 0, "total_prompt_tokens": 0, "total_completion_tokens": 0,
    }
    return compiled_graph.invoke(initial_state)