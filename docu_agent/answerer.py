"""Answerer agent: builds an answer from retrieved chunks.

Default mode is *extractive*: it picks the sentences from the retrieved
chunks that best match the question, so every sentence in the answer
already exists in your documents. That is what makes the answer checkable
and keeps the whole pipeline working with no API key.

How sentences are picked (three rules, in order):
1. A sentence must share at least 2 content words with the question
   (1 word is enough for a one-word question) — one shared word like
   "office" is not evidence the sentence answers the question.
2. Words are weighted by IDF: rare, distinctive words ("annual") count
   more than words that appear everywhere ("leave").
3. Each extra sentence must add at least one new question word the
   earlier sentences didn't cover — so the answer doesn't repeat the
   same fact three ways.

Optional LLM mode (see llm.py) rewrites the same retrieved context with a
model, but the verifier still checks the result afterwards either way.
"""

from __future__ import annotations

import math
import re

from .retriever import STOPWORDS, tokenize

SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

NO_ANSWER_TEXT = "I couldn't find an answer in the documents."


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_RE.split(text.strip()) if s.strip()]


def _idf_weights(chunks: list[dict]) -> dict[str, float]:
    """IDF weight per word across the retrieved chunks.

    A word in few chunks is distinctive (high weight); a word in every
    chunk is generic (low weight). ``+ 1.0`` keeps every weight positive.
    """
    n = len(chunks)
    doc_freq: dict[str, int] = {}
    for chunk in chunks:
        words = set(tokenize(chunk["section"] + " " + chunk["text"]))
        for word in words:
            doc_freq[word] = doc_freq.get(word, 0) + 1
    return {
        word: math.log((n + 1) / (freq + 0.5)) + 1.0
        for word, freq in doc_freq.items()
    }


def extractive_answer(
    sub_questions: list[str],
    chunks: list[dict],
    max_sentences: int = 3,
    min_score: float = 0.4,
) -> tuple[str, list[str]]:
    """Pick the best sentences from the chunks for the given sub-questions.

    Returns (answer_text, citations). Citations are "file.md — Section"
    strings, one per sentence used. If nothing scores high enough, returns
    an honest "couldn't find it" answer with no citations.
    """
    if not chunks:
        return (NO_ANSWER_TEXT, [])

    question_tokens: set[str] = set()
    for q in sub_questions:
        question_tokens |= set(tokenize(q))
    if not question_tokens:
        return (NO_ANSWER_TEXT, [])

    # Rule 1: a sentence must match at least 2 question words (or 1, for a
    # one-word question) to be eligible at all.
    required_matches = 2 if len(question_tokens) >= 2 else 1
    weights = _idf_weights(chunks)

    scored: list[tuple[float, str, str, set[str]]] = []
    for chunk in chunks:
        boost = 1.0 + 0.25 * chunk.get("score", 0.0)
        citation = f"{chunk['source']} — {chunk['section']}"
        # The section heading counts as context for every sentence in it.
        section_tokens = set(tokenize(chunk["section"]))
        for sentence in _sentences(chunk["text"]):
            sentence_tokens = tokenize(sentence)
            token_set = set(sentence_tokens)
            matched = (token_set | section_tokens) & question_tokens
            if len(matched) < required_matches:
                continue
            # Rule 2: IDF-weighted overlap. No length penalty on purpose:
            # a longer sentence that covers more of the question IS a
            # better answer, and penalising length let short near-miss
            # sentences (e.g. sick leave for an annual-leave question)
            # outrank the sentence with the actual fact. Ties keep
            # document order, so the sentence the doc states first wins.
            weight = sum(weights.get(t, 1.0) for t in matched)
            score = weight * boost
            if score >= min_score:
                scored.append((score, sentence, citation, matched))

    scored.sort(key=lambda item: item[0], reverse=True)
    seen: set[str] = set()
    covered: set[str] = set()
    picked: list[tuple[str, str]] = []
    for _score, sentence, citation, matched in scored:
        key = sentence.lower()
        if key in seen:
            continue
        # Rule 3: after the first sentence, each new sentence must cover
        # at least one question word the earlier ones didn't.
        if picked and not (matched - covered):
            continue
        seen.add(key)
        covered |= matched
        picked.append((sentence, citation))
        if len(picked) >= max_sentences:
            break

    if not picked:
        return (NO_ANSWER_TEXT, [])

    answer_lines = []
    citations: list[str] = []
    for sentence, citation in picked:
        answer_lines.append(f"{sentence} [{citation}]")
        if citation not in citations:
            citations.append(citation)
    return ("\n".join(answer_lines), citations)
