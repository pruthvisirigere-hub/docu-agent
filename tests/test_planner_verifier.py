"""Planner + verifier tests."""

from docu_agent.planner import split_question
from docu_agent.verifier import check_grounding


def test_planner_splits_a_compound_question():
    subs = split_question(
        "How many days of paid leave do I get and how do I request it?"
    )
    assert len(subs) == 2
    assert "paid leave" in subs[0]
    assert "request" in subs[1]


def test_planner_leaves_a_simple_question_alone():
    subs = split_question("How many days of paid leave do I get?")
    assert subs == ["How many days of paid leave do I get?"]


def test_planner_handles_empty_question():
    assert split_question("   ") == []


def test_verifier_rejects_an_ungrounded_answer():
    chunks = [
        {
            "text": "Every full-time employee gets 20 days of paid annual leave per year.",
            "source": "leave-policy.md",
            "section": "Annual Leave",
        }
    ]
    verdict = check_grounding(
        "Employees get 45 days of leave and a free company yacht.",
        chunks,
    )
    assert verdict["grounded"] is False
    assert verdict["unsupported"], "the invented sentence should be flagged"


def test_verifier_accepts_a_grounded_answer():
    chunks = [
        {
            "text": "Every full-time employee gets 20 days of paid annual leave per year.",
            "source": "leave-policy.md",
            "section": "Annual Leave",
        }
    ]
    verdict = check_grounding(
        "Every full-time employee gets 20 days of paid annual leave per year.",
        chunks,
    )
    assert verdict["grounded"] is True
    assert verdict["coverage"] >= 0.9


def test_verifier_fails_with_no_chunks():
    verdict = check_grounding("Anything at all.", [])
    assert verdict["grounded"] is False
