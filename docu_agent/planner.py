"""Planner agent: breaks a question into smaller sub-questions.

Why split? One long question often hides two or three different things
("How many leave days do I get AND how do I request them?"). Each
sub-question gets its own retrieval pass, so the answer covers all parts
instead of only the first one the retriever happens to match.
"""

from __future__ import annotations

import re


def _is_substantial(part: str) -> bool:
    """A usable sub-question has at least 3 words."""
    return len(part.split()) >= 3


def split_question(question: str) -> list[str]:
    """Split a question into sub-questions.

    Splits on " and " and on commas, but only when both sides are
    substantial enough to stand alone. A question that doesn't split
    comes back unchanged as a single-item list.
    """
    question = " ".join(question.strip().split())
    if not question:
        return []

    # Split on " and " first (most common compound-question pattern).
    parts = re.split(r"\s+and\s+", question, flags=re.IGNORECASE)
    if len(parts) > 1 and all(_is_substantial(p) for p in parts):
        return [p.strip(" ,?") + "?" for p in parts]

    # Then try commas: "What is X, how do I Y?"
    parts = [p.strip() for p in question.split(",")]
    if len(parts) > 1 and all(_is_substantial(p) for p in parts):
        return [p.strip(" ,?") + "?" for p in parts]

    return [question if question.endswith("?") else question + "?"]
