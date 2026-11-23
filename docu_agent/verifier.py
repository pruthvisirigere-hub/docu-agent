"""Verifier agent: checks the answer is grounded in the retrieved text.

The verifier is the quality gate. It takes the answer's sentences and
checks, word by word, that each one is supported by the chunks the
retriever returned. An answer that invents facts scores low and gets sent
back for one retry; an answer that sticks to the documents passes.
"""

from __future__ import annotations

import re

from .retriever import tokenize


def _content_words(text: str) -> set[str]:
    """Content words of a text (stopwords removed)."""
    return set(tokenize(text))


def check_grounding(
    answer: str, chunks: list[dict], min_coverage: float = 0.5
) -> dict:
    """Verify every sentence of the answer against the retrieved chunks.

    Returns a dict with:
      grounded: True when every sentence is supported by the chunks.
      coverage: overall fraction of the answer's content words found in
          the retrieved text (0.0 - 1.0).
      unsupported: sentences whose coverage is below the threshold.
      reason: short human-readable explanation.
    """
    if not chunks:
        return {
            "grounded": False,
            "coverage": 0.0,
            "unsupported": [answer],
            "reason": "No documents were retrieved, so nothing supports the answer.",
        }

    retrieved_words: set[str] = set()
    for chunk in chunks:
        retrieved_words |= _content_words(chunk["text"])

    # Split the answer into sentences, stripping the [citation] markers
    # the answerer appends.
    clean = re.sub(r"\[[^\]]*\]", "", answer)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", clean) if s.strip()]

    if not sentences:
        return {
            "grounded": False,
            "coverage": 0.0,
            "unsupported": [],
            "reason": "The answer is empty.",
        }

    unsupported: list[str] = []
    total_words = 0
    covered_words = 0
    for sentence in sentences:
        words = _content_words(sentence)
        if not words:
            continue
        total_words += len(words)
        covered = len(words & retrieved_words)
        covered_words += covered
        if covered / len(words) < min_coverage:
            unsupported.append(sentence)

    coverage = covered_words / total_words if total_words else 0.0
    grounded = not unsupported
    return {
        "grounded": grounded,
        "coverage": round(coverage, 2),
        "unsupported": unsupported,
        "reason": (
            "Every sentence is supported by the retrieved documents."
            if grounded
            else f"{len(unsupported)} sentence(s) not supported by the retrieved documents."
        ),
    }
