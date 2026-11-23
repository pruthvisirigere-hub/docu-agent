"""LangGraph graph: Planner -> Retriever -> Answerer -> Verifier.

State flows through the graph in a TypedDict. Each node reads what it
needs from the state and writes its own output back. The verifier has a
conditional edge: a grounded answer ends the run; an ungrounded answer
goes back to the answerer once for a retry, and after that the run ends
with the best-effort answer plus a warning flag.

Graph (text diagram):

    question
        |
    [Planner] -- splits the question into sub-questions
        |
    [Retriever] -- BM25 search per sub-question over docs/
        |
    [Answerer] -- extractive answer (or LLM with --llm)
        |
    [Verifier] -- checks grounding
        |              \
     grounded        not grounded
        |              |
       END      retries < 1 ? --yes--> [Answerer] (retry, looser)
                         |
                         no --> END (with warning)
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, StateGraph

from .answerer import NO_ANSWER_TEXT, extractive_answer
from .llm import llm_answer
from .planner import split_question
from .retriever import DocumentStore
from .verifier import check_grounding

MAX_RETRIES = 1


class AgentState(TypedDict, total=False):
    """Everything the agents share as the question flows through the graph."""

    question: str
    sub_questions: list[str]
    retrieved: list[dict]
    draft_answer: str
    citations: list[str]
    verifier_verdict: dict
    retries: int
    answer: str
    warning: str | None
    trace: list[str]
    use_llm: bool


def build_graph(docs_dir: str) -> object:
    """Build the agent graph bound to one docs folder."""
    store = DocumentStore(docs_dir)

    def _trace(state: AgentState, message: str) -> list[str]:
        return [*state.get("trace", []), message]

    def planner_node(state: AgentState) -> dict:
        sub_questions = split_question(state["question"])
        return {
            "sub_questions": sub_questions,
            "trace": _trace(
                state, f"Planner: split into {len(sub_questions)} sub-question(s): {sub_questions}"
            ),
        }

    def retriever_node(state: AgentState) -> dict:
        retrieved: list[dict] = []
        seen: set[tuple[str, str]] = set()
        for sub in state.get("sub_questions", [state["question"]]):
            for chunk in store.search(sub, top_k=5):
                key = (chunk["source"], chunk["section"])
                if key not in seen:
                    seen.add(key)
                    retrieved.append(chunk)
        retrieved.sort(key=lambda c: c.get("score", 0), reverse=True)
        return {
            "retrieved": retrieved[:8],
            "trace": _trace(
                state,
                f"Retriever: found {len(retrieved)} chunk(s) from "
                + (", ".join(sorted({c['source'] for c in retrieved})) or "no documents"),
            ),
        }

    def answerer_node(state: AgentState) -> dict:
        retries = state.get("retries", 0)
        # On retry, be looser: more sentences, lower score threshold.
        max_sentences = 4 if retries else 3
        min_score = 0.25 if retries else 0.4
        answer, citations = extractive_answer(
            state.get("sub_questions", [state["question"]]),
            state.get("retrieved", []),
            max_sentences=max_sentences,
            min_score=min_score,
        )
        mode = "extractive"
        if state.get("use_llm") and state.get("retrieved"):
            llm_text = llm_answer(state["question"], state["retrieved"])
            if llm_text:
                answer, mode = llm_text, "llm"
                citations = sorted(
                    {f"{c['source']} — {c['section']}" for c in state["retrieved"]}
                )
        return {
            "draft_answer": answer,
            "answer": answer,
            "citations": citations,
            "trace": _trace(state, f"Answerer ({mode}): produced an answer with {len(citations)} citation(s)"),
        }

    def verifier_node(state: AgentState) -> dict:
        draft = state.get("draft_answer", "")
        if draft == NO_ANSWER_TEXT:
            # "I couldn't find it" is itself a true statement about the
            # documents, not an ungrounded claim — no retry, no warning.
            verdict = {
                "grounded": True,
                "coverage": 0.0,
                "unsupported": [],
                "reason": "No answer exists in the documents — correctly reported as not found.",
            }
            return {
                "verifier_verdict": verdict,
                "warning": None,
                "trace": _trace(state, "Verifier: no answer in the documents — correctly reported"),
            }
        verdict = check_grounding(draft, state.get("retrieved", []))
        warning = None
        if not verdict["grounded"] and state.get("retries", 0) >= MAX_RETRIES:
            warning = (
                "Best-effort answer: the verifier could not fully confirm "
                "grounding after a retry. Treat the answer with caution."
            )
        return {
            "verifier_verdict": verdict,
            "warning": warning,
            "trace": _trace(
                state,
                f"Verifier: grounded={verdict['grounded']} "
                f"(coverage {verdict['coverage']}) — {verdict['reason']}",
            ),
        }

    def route_after_verifier(state: AgentState) -> str:
        verdict = state.get("verifier_verdict", {})
        if verdict.get("grounded"):
            return END
        if state.get("retries", 0) < MAX_RETRIES:
            return "answerer"
        return END

    def bump_retry(state: AgentState) -> dict:
        return {
            "retries": state.get("retries", 0) + 1,
            "trace": _trace(state, "Verifier: answer not grounded — sending back to Answerer for one retry"),
        }

    graph = StateGraph(AgentState)
    graph.add_node("planner", planner_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("answerer", answerer_node)
    graph.add_node("verifier", verifier_node)
    # Wrapper node so the retry counter bumps before re-entering answerer.
    graph.add_node("retry", bump_retry)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "retriever")
    graph.add_edge("retriever", "answerer")
    graph.add_edge("answerer", "verifier")
    graph.add_conditional_edges(
        "verifier",
        route_after_verifier,
        {END: END, "answerer": "retry"},
    )
    graph.add_edge("retry", "answerer")
    return graph.compile()


def ask(question: str, docs_dir: str, use_llm: bool = False) -> dict:
    """Run one question through the graph and return the final state."""
    graph = build_graph(docs_dir)
    result = graph.invoke(
        {
            "question": question,
            "retries": 0,
            "use_llm": use_llm,
            "trace": [],
        }
    )
    return result
