"""Retrieval tests: the right document must win the search."""

from pathlib import Path

from docu_agent.retriever import DocumentStore

DOCS = Path(__file__).resolve().parents[1] / "docs"


def test_retrieval_finds_the_right_doc():
    store = DocumentStore(DOCS)
    results = store.search("How many days of paid annual leave do I get?")
    assert results, "expected at least one retrieved chunk"
    assert results[0]["source"] == "leave-policy.md"
    assert "20 days" in results[0]["text"]


def test_retrieval_finds_api_guide_for_auth_question():
    store = DocumentStore(DOCS)
    results = store.search("How do I authenticate with the API?")
    assert results, "expected at least one retrieved chunk"
    assert results[0]["source"] == "api-guide.md"
    assert "X-API-Key" in results[0]["text"]


def test_retrieval_returns_empty_for_unrelated_query():
    store = DocumentStore(DOCS)
    results = store.search("zebra quantum knitting")
    assert results == []


def test_empty_docs_folder_handled_gracefully(tmp_path):
    store = DocumentStore(tmp_path)  # no .md files at all
    assert store.is_empty
    assert store.search("anything") == []
