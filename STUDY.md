# STUDY.md — docu-agent, explained picture-first

Read this before the code. If you can explain every section below out loud, you can defend this project in an interview.

## The picture in one paragraph

You have a folder of documents — a leave policy, an API guide, a product FAQ, an onboarding guide. You ask a question in plain English. Four small agents work on it in a chain: the **Planner** splits your question into smaller questions, the **Retriever** finds the relevant parts of the documents, the **Answerer** picks the sentences that answer you, and the **Verifier** checks the answer is actually in the documents before you see it. If the answer isn't supported, it goes back for one retry. Everything is cited, so you can check every claim.

## What each file does

| File | Job | In one line |
| --- | --- | --- |
| `main.py` | The front door | Parses `ask "..."` and prints the answer, sources, and verifier verdict |
| `docu_agent/graph.py` | The boss | Defines the LangGraph graph: the 4 nodes, the order they run in, and the retry rule |
| `docu_agent/planner.py` | The splitter | Turns one compound question into sub-questions |
| `docu_agent/retriever.py` | The memory | Loads `.md` files, chunks them by heading, builds a BM25 index, searches it |
| `docu_agent/answerer.py` | The writer | Picks the best sentences from retrieved chunks, with citations |
| `docu_agent/verifier.py` | The fact-checker | Checks every answer sentence's words appear in the retrieved text |
| `docu_agent/llm.py` | The optional upgrade | Same job as the Answerer, but via OpenRouter when a key is set |
| `docs/*.md` | The knowledge | Sample documents the agent answers from |
| `tests/` | The safety net | 16 pytest tests proving each piece works |

## LangGraph in plain words

**State** is a shared dictionary that flows through the graph. It starts as just `{"question": "..."}` and each node adds its output: the Planner adds `sub_questions`, the Retriever adds `retrieved` chunks, the Answerer adds `answer` and `citations`, the Verifier adds `verifier_verdict`. Every node can read everything written before it.

**Nodes** are just Python functions. Each one takes the state and returns a small dict of updates. That's it — a node is a function with a name.

**Edges** say what runs next. Most edges are simple arrows: Planner → Retriever → Answerer → Verifier. The interesting one is the **conditional edge** out of the Verifier: it looks at the state and decides — grounded answer goes to END, ungrounded answer with retries left goes back to the Answerer (through a tiny `retry` node that bumps the counter), ungrounded with no retries left goes to END with a warning.

**Why a graph instead of four function calls in a row?** Because the retry is a loop, and loops with state are exactly what a graph expresses cleanly. If you later add a fifth agent (say, a web-search fallback when the docs don't contain the answer), you add one node and one edge instead of rewriting control flow.

## BM25 in plain words

BM25 is a search ranking formula from the 1990s, and it is still the default keyword search in most real systems (Elasticsearch uses it). The idea:

1. **Tokenise** — split text into words, lowercase them, drop boring words ("the", "how", "is"), and stem them ("days" → "day", "authentication" → "authenticate") so different forms of a word match.
2. **Score each chunk** against the question words. A chunk scores higher when it contains the question's words, especially words that are rare across all chunks. A word like "leave" appears in many chunks, so it counts less; a word like "annual" appears in few, so it counts more. That rarity weighting is called **IDF** (inverse document frequency).
3. **Term frequency saturates** — a word appearing 10 times isn't 10× better than appearing once. BM25 flattens that out.
4. **Length normalisation (turned off here)** — BM25 normally penalises long documents. Our chunks are pre-split into similar-sized sections, so length mostly adds noise: a shorter near-miss section (Sick Leave) outranked the right one (Annual Leave) until we set `b=0`. That is a real, deliberate IR decision, not a hack — and a good thing to mention in an interview.

The Answerer reuses the same idea at sentence level: it scores each sentence by IDF-weighted overlap with the question, requires at least 2 matching words (one shared word is not evidence), and each extra sentence must cover a question word the earlier ones didn't — so the answer doesn't repeat the same fact three ways.

## The verifier, and why it matters

Most beginner RAG demos skip verification, and that is exactly how chatbots invent facts. Our verifier is deliberately simple: for each sentence of the answer, what fraction of its content words appear in the retrieved chunks? Below 50% coverage, the sentence is unsupported. Simple beats clever here — it's explainable, it has no model to hallucinate, and it catches the failure mode that matters (an answer that drifts away from the documents). One retry, then best effort with a warning flag: the system never silently pretends an unverified answer is verified.

## 5 interview questions you should be able to answer

**1. "Walk me through the architecture."**
Four LangGraph nodes sharing one state dict: Planner splits the question, Retriever does BM25 search over heading-chunked Markdown, Answerer extracts cited sentences, Verifier checks grounding with a one-retry loop. Extractive by default, so it runs with no API key; OpenRouter is an optional flag.

**2. "Why BM25 instead of embeddings?"**
BM25 needs no model download, no GPU, no API key, and it is exact and explainable — you can point at why a chunk ranked first. The honest trade-off: it is keyword matching, so it misses paraphrases ("time off" won't match "leave"). The next step is hybrid BM25 + embeddings.

**3. "How do you know the answer is correct?"**
Two layers: citations on every sentence (you can check the source yourself), and the verifier, which measures word coverage of each answer sentence against the retrieved text and retries once if it fails. Correctness here means *grounded in the documents*, not *true in the world* — say that distinction out loud, interviewers like it.

**4. "What happens when the answer isn't in the documents?"**
The Answerer returns "I couldn't find an answer in the documents" instead of inventing one, and the Verifier treats that honest answer as a pass, not a failure. Refusing to answer is a feature, not a bug.

**5. "What would you improve next?"**
Hybrid retrieval (BM25 + embeddings) for paraphrases; overlapping chunks for very long sections; a fixed eval set of 15–20 questions with expected answers so quality is measured, not eyeballed; and a FastAPI front end so non-technical users can ask questions.

## Try it yourself (5 minutes)

```bash
python main.py ask "How many days of paid annual leave do I get?" --verbose
python main.py ask "What is the warranty on the Northwind Desk?"
python main.py ask "What is the company policy on office dogs?"
```

Then open `docu_agent/graph.py` and trace the state: what does each node add? Then break something on purpose — delete a docs file, ask a nonsense question — and watch how the system responds. That is how you learn it well enough to explain it.
