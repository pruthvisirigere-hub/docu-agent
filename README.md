# docu-agent

A **LangGraph multi-agent document QA agent**. Ask a question in plain English, get an answer extracted from your own Markdown documents — with citations and a verifier that checks the answer is actually grounded in what was retrieved.

No API key needed. No model download. The default answerer is extractive: every sentence in the answer already exists in your documents.

## How it works

Four agents, one LangGraph graph:

```
question
    |
[Planner]    -- splits the question into sub-questions
    |
[Retriever]  -- BM25 search per sub-question over docs/
    |
[Answerer]   -- extractive answer from the retrieved chunks
    |
[Verifier]   -- checks the answer is grounded in the retrieved text
    |              \
 grounded        not grounded
    |              |
   END      retries < 1 ? --yes--> [Answerer] (retry, looser)
                         |
                         no --> END (best effort + warning)
```

- **Planner** — "How many leave days do I get and how do I request them?" becomes two sub-questions, so the answer covers both parts.
- **Retriever** — loads every `.md` file in `docs/`, splits each into one chunk per heading section, and ranks chunks with BM25 (`rank-bm25`). Each chunk remembers its file and section for citations.
- **Answerer** — scores sentences from the retrieved chunks (IDF-weighted question overlap, section headings count as context) and picks the best ones. Every sentence is cited as `file.md — Section`.
- **Verifier** — checks each answer sentence's content words against the retrieved text. If a sentence isn't supported, the answer goes back to the Answerer once for a retry; if it still fails, you get the best-effort answer with a warning flag.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Requires Python 3.10+.

## Usage

```bash
# Ask a question (uses the sample docs in docs/)
python main.py ask "How many days of paid annual leave do I get?"

# See each agent's step
python main.py ask "How many days of paid leave do I get and how do I request it?" --verbose

# Use your own documents instead of the samples
python main.py ask "What is the refund policy?" --docs ./my-docs

# Answer with an LLM via OpenRouter instead of extraction (optional)
export OPENROUTER_API_KEY="sk-or-..."
python main.py ask "How many days of paid leave do I get?" --llm
```

## Example output

```
$ python main.py ask "How many days of paid leave do I get and how do I request it?" --verbose

== Agent trace ==
  • Planner: split into 2 sub-question(s): ['How many days of paid leave do I get?', 'how do I request it?']
  • Retriever: found 8 chunk(s) from api-guide.md, leave-policy.md
  • Answerer (extractive): produced an answer with 2 citation(s)
  • Verifier: grounded=True (coverage 1.0) — Every sentence is supported by the retrieved documents.

== Answer ==
Every full-time employee gets 20 days of paid annual leave per calendar year. [leave-policy.md — Annual Leave]
Your manager approves requests within 3 working days. [leave-policy.md — How to Request Leave]

== Sources ==
  • leave-policy.md — Annual Leave
  • leave-policy.md — How to Request Leave

== Verifier ==
  Grounded: True (coverage 1.0) — Every sentence is supported by the retrieved documents.
```

A question with no answer in the docs is reported honestly instead of invented:

```
$ python main.py ask "What is the company policy on office dogs?"

== Answer ==
I couldn't find an answer in the documents.

== Verifier ==
  Grounded: True (coverage 0.0) — No answer exists in the documents — correctly reported as not found.
```

## Sample documents

`docs/` ships with 4 short, realistic files so it runs out of the box:

| File | About |
| --- | --- |
| `leave-policy.md` | Annual leave (20 days), sick leave, how to request, carryover |
| `api-guide.md` | API key auth, rate limits, endpoints, error codes |
| `product-faq.md` | Pricing, 2-year warranty, shipping, 30-day returns |
| `onboarding.md` | First day, accounts, buddy system, 3-month probation |

Replace them with your own `.md` files — no code changes needed.

## Tests

```bash
pytest -v
```

16 tests: retrieval finds the right doc, the verifier rejects invented answers and accepts grounded ones, the planner splits compound questions, the full graph and the CLI end to end, and empty docs folders handled gracefully.

## Project structure

```
docu-agent/
├── main.py                  # CLI: `python main.py ask "..."` [--verbose] [--llm] [--docs DIR]
├── docu_agent/
│   ├── __init__.py
│   ├── graph.py             # LangGraph StateGraph: 4 nodes + verifier retry edge
│   ├── planner.py           # splits a question into sub-questions
│   ├── retriever.py         # .md loading, heading chunking, BM25 index
│   ├── answerer.py          # extractive answering with citations
│   ├── verifier.py          # grounding check (word coverage vs retrieved text)
│   └── llm.py               # optional OpenRouter answers (stdlib only)
├── docs/                    # sample Markdown documents
├── tests/                   # pytest suite (16 tests)
├── requirements.txt
└── STUDY.md                 # picture-first explainer + interview questions
```

## What I'd improve next

- **Embeddings retrieval** — BM25 is keyword matching; adding a small embedding model would catch paraphrases ("time off" → leave policy). Hybrid BM25 + vectors is the standard next step.
- **Chunking** — heading-based chunks are simple; overlapping sliding windows would help very long sections.
- **Eval set** — a fixed set of 15–20 questions with expected answers, so regressions are measurable instead of eyeballed.
- **Streaming + a web UI** — FastAPI + a small frontend so non-technical users can ask questions too.
