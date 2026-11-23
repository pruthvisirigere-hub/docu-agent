"""End-to-end tests: the graph and the CLI, start to finish."""

import subprocess
import sys
from pathlib import Path

from docu_agent.answerer import extractive_answer
from docu_agent.graph import ask
from docu_agent.retriever import DocumentStore

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


def test_answerer_cites_its_sources():
    store = DocumentStore(DOCS)
    chunks = store.search("How many days of paid annual leave do I get?")
    answer, citations = extractive_answer(
        ["How many days of paid annual leave do I get?"], chunks
    )
    assert "20 days" in answer
    assert citations == ["leave-policy.md — Annual Leave"]


def test_graph_answers_a_sample_question_end_to_end():
    result = ask("How many days of paid annual leave do I get?", str(DOCS))
    assert "20 days" in result["answer"]
    assert result["verifier_verdict"]["grounded"] is True
    assert result["citations"]


def test_graph_handles_a_question_with_no_answer():
    result = ask("What is the company policy on office dogs?", str(DOCS))
    assert "couldn't find" in result["answer"].lower()
    # Saying "I couldn't find it" is honest, not a verification failure.
    assert result["verifier_verdict"]["grounded"] is True
    assert result["warning"] is None


def test_cli_end_to_end_on_a_sample_question():
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "main.py"),
            "ask",
            "How do I authenticate with the API?",
        ],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert proc.returncode == 0, proc.stderr
    assert "X-API-Key" in proc.stdout
    assert "api-guide.md" in proc.stdout
    assert "Grounded: True" in proc.stdout


def test_cli_verbose_shows_agent_steps():
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "main.py"),
            "ask",
            "How many days of paid leave do I get?",
            "--verbose",
        ],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert proc.returncode == 0, proc.stderr
    for step in ("Planner:", "Retriever:", "Answerer", "Verifier:"):
        assert step in proc.stdout


def test_cli_empty_docs_folder_is_graceful(tmp_path):
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "main.py"),
            "ask",
            "Anything?",
            "--docs",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert proc.returncode == 0
    assert "No .md documents found" in proc.stdout
