"""Document loading, chunking and BM25 retrieval.

The retriever is the agent's "memory". It reads every .md file in a docs
folder, splits each file into chunks (one chunk per section, split on
Markdown headings), builds a BM25 index over those chunks, and returns the
most relevant chunks for a question.
"""

from __future__ import annotations

import re
from pathlib import Path

from rank_bm25 import BM25Okapi

# Small stopword list so common words ("the", "how", "do") don't dominate
# matching. This is a retrieval heuristic, not a language model.
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "can", "could", "should", "would", "will",
    "how", "what", "when", "where", "who", "why", "which", "whom",
    "i", "you", "we", "they", "he", "she", "it", "my", "your", "our",
    "me", "us", "them", "this", "that", "these", "those", "there",
    "here", "of", "in", "on", "at", "to", "for", "with", "by", "from",
    "and", "or", "but", "not", "no", "if", "then", "than", "so", "as",
    "get", "gets", "got", "give", "please", "tell",
}


def _stem(word: str) -> str:
    """Tiny stemmer so 'authenticate' matches 'authentication', 'days' matches 'day'.

    Deliberately minimal: only strips common suffixes, applied identically
    to documents and questions so both sides always agree.
    """
    if len(word) > 6 and word.endswith("ation"):
        return word[:-5] + "ate"  # authentication -> authenticate
    if len(word) > 4 and word.endswith("ing"):
        return word[:-3]
    if len(word) > 3 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("es"):
        return word[:-2]
    if len(word) > 3 and word.endswith("ed"):
        return word[:-2]
    if len(word) > 3 and word.endswith("s"):
        return word[:-1]
    return word


def tokenize(text: str) -> list[str]:
    """Lowercase, split on non-letters/digits, drop stopwords, stem."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [_stem(t) for t in tokens if t not in STOPWORDS]


def split_into_sections(path: Path) -> list[dict]:
    """Split one Markdown file into chunks, one per heading section.

    Each chunk keeps its source file and section heading so answers can
    cite exactly where they came from.
    """
    text = path.read_text(encoding="utf-8")
    chunks: list[dict] = []
    current_section = path.stem.replace("-", " ").title()
    current_lines: list[str] = []

    def flush() -> None:
        body = "\n".join(current_lines).strip()
        if body:
            chunks.append(
                {
                    "text": body,
                    "source": path.name,
                    "section": current_section,
                }
            )

    for line in text.splitlines():
        heading = re.match(r"^#{1,3}\s+(.*)", line.strip())
        if heading:
            flush()
            current_section = heading.group(1).strip()
            current_lines = []
        else:
            current_lines.append(line)
    flush()
    return chunks


class DocumentStore:
    """Loads .md files, chunks them, and answers BM25 searches."""

    def __init__(self, docs_dir: str | Path):
        self.docs_dir = Path(docs_dir)
        self.chunks: list[dict] = []
        self._bm25: BM25Okapi | None = None
        self._load()

    def _load(self) -> None:
        if not self.docs_dir.is_dir():
            return
        for path in sorted(self.docs_dir.glob("*.md")):
            self.chunks.extend(split_into_sections(path))
        if self.chunks:
            # Index the section heading together with the body: a section
            # titled "Authentication" should match "how do I authenticate".
            corpus = [
                tokenize(c["section"] + " " + c["text"]) for c in self.chunks
            ]
            # b=0 turns off BM25's length normalisation. It exists to stop
            # long documents dominating, but our chunks are pre-split into
            # similar-sized sections, so length mostly adds noise: without
            # this, a shorter near-miss section (Sick Leave) outranked the
            # right one (Annual Leave) for a leave-days question.
            self._bm25 = BM25Okapi(corpus, b=0)

    @property
    def is_empty(self) -> bool:
        return not self.chunks

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Return the top_k chunks most relevant to the query.

        Each result is a chunk dict plus a ``score`` (higher = more relevant).
        Returns an empty list when there are no documents or no match.
        """
        if self._bm25 is None or not query.strip():
            return []
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        scores = self._bm25.get_scores(query_tokens)
        ranked = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )
        results = []
        for i in ranked[:top_k]:
            if scores[i] <= 0:
                continue
            chunk = dict(self.chunks[i])
            chunk["score"] = round(float(scores[i]), 3)
            results.append(chunk)
        return results
