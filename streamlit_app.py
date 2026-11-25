"""docu-agent — live demo (Streamlit).

Runs the real LangGraph pipeline from this repo:
Planner -> Retriever (BM25) -> Answerer -> Verifier (with one retry).

Deploy on Streamlit Community Cloud with this file as the entrypoint.
No API key needed. If an OPENROUTER_API_KEY secret is set, visitors can
optionally switch the Answerer to an LLM; the Verifier still checks it.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import streamlit as st

from docu_agent.graph import ask
from docu_agent.retriever import DocumentStore

REPO_URL = "https://github.com/pruthvisirigere-hub/docu-agent"
PORTFOLIO_URL = "https://pruthvisirigere-hub.github.io"
SAMPLE_DOCS = Path(__file__).parent / "docs"
MAX_FILES = 20
MAX_FILE_BYTES = 1_000_000

EXAMPLES = [
    "How many days of paid annual leave do I get?",
    "How many days of paid leave do I get and how do I request it?",
    "How do I authenticate with the API?",
    "What is the API rate limit?",
    "Is there a parking allowance?",  # not in the docs: shows honest abstention
]

st.set_page_config(page_title="docu-agent · live demo", page_icon="🕵️‍♀️", layout="wide")


# ---------- secrets (optional LLM mode) ----------
def _secret(name: str) -> str | None:
    try:
        value = st.secrets.get(name)  # raises if no secrets file exists locally
    except Exception:
        value = None
    return value or os.environ.get(name)


_key = _secret("OPENROUTER_API_KEY")
if _key:
    os.environ["OPENROUTER_API_KEY"] = _key
    _model = _secret("OPENROUTER_MODEL")
    if _model:
        os.environ["OPENROUTER_MODEL"] = _model
LLM_AVAILABLE = bool(_key)


# ---------- header ----------
st.title("🕵️‍♀️ docu-agent")
st.markdown(
    "**Multi-agent document Q&A built with LangGraph.** Ask a question in plain English and four agents "
    "work together: a **Planner** splits it into sub-questions, a **Retriever** searches the documents with "
    "BM25, an **Answerer** writes a cited answer, and a **Verifier** checks every sentence is grounded in "
    "the retrieved text, retrying once or honestly abstaining if it isn't."
)
st.caption(f"[Source code on GitHub]({REPO_URL}) · built by [Pruthvi Sirigere]({PORTFOLIO_URL})")

with st.expander("How the agent graph works"):
    st.code(
        """question
   │
[Planner]    splits the question into sub-questions
   │
[Retriever]  BM25 search per sub-question (one chunk per Markdown section)
   │
[Answerer]   extractive answer with per-sentence citations
   │
[Verifier]   checks grounding coverage (0.5 threshold)
   │            └─ not grounded → retry Answerer once → else best effort + warning
  END""",
        language="text",
    )


# ---------- document source ----------
with st.sidebar:
    st.header("📚 Documents")
    source = st.radio(
        "Which documents should the agent read?",
        ["Sample docs (Northwind Labs)", "Upload my own Markdown files"],
    )

    docs_dir: Path | None = None
    if source.startswith("Sample"):
        docs_dir = SAMPLE_DOCS
        st.caption("A fictional company's leave policy, onboarding guide, API guide and product FAQ.")
        for path in sorted(SAMPLE_DOCS.glob("*.md")):
            with st.expander(path.name):
                st.markdown(path.read_text(encoding="utf-8"))
    else:
        uploads = st.file_uploader(
            "Upload .md files (up to 20, 1 MB each)",
            type=["md", "markdown"],
            accept_multiple_files=True,
        )
        if uploads:
            if "upload_dir" not in st.session_state:
                st.session_state.upload_dir = tempfile.mkdtemp(prefix="docu-agent-")
            upload_dir = Path(st.session_state.upload_dir)
            shutil.rmtree(upload_dir, ignore_errors=True)
            upload_dir.mkdir(parents=True, exist_ok=True)
            saved = 0
            for up in uploads[:MAX_FILES]:
                data = up.getvalue()
                if len(data) > MAX_FILE_BYTES:
                    st.warning(f"Skipped {up.name}: larger than 1 MB.")
                    continue
                name = Path(up.name).name
                if not name.lower().endswith(".md"):
                    name = Path(name).stem + ".md"
                (upload_dir / name).write_bytes(data)
                saved += 1
            if len(uploads) > MAX_FILES:
                st.warning(f"Only the first {MAX_FILES} files are used.")
            if saved:
                docs_dir = upload_dir
                st.success(f"{saved} file(s) ready.")
        else:
            st.info("Upload one or more Markdown files to ask questions about them.")

    st.divider()
    use_llm = False
    if LLM_AVAILABLE:
        use_llm = st.toggle(
            "Write the answer with an LLM (OpenRouter)",
            value=False,
            help="Default is extractive: every sentence comes straight from your documents. "
            "The Verifier checks LLM answers too.",
        )
    else:
        st.caption("Extractive mode: every answer sentence comes straight from the documents. No API key needed.")


# ---------- question ----------
if "question" not in st.session_state:
    st.session_state.question = EXAMPLES[0]

if docs_dir is SAMPLE_DOCS:
    st.write("**Try an example:**")
    cols = st.columns(len(EXAMPLES))
    for col, example in zip(cols, EXAMPLES):
        if col.button(example):
            st.session_state.question = example

question = st.text_input("Your question", key="question", max_chars=300)
st.caption(
    "Tip: the default answerer is extractive and strict. It only uses sentences that share at least two "
    "key words with your question, so phrase questions with words from the documents. If nothing matches, "
    "it says so instead of guessing."
)
run = st.button("Ask the agents ✨", type="primary", disabled=docs_dir is None)

if docs_dir is None:
    st.stop()

store = DocumentStore(docs_dir)
if store.is_empty:
    st.warning("No readable Markdown sections were found in these files.")
    st.stop()

if run and question.strip():
    with st.spinner("Planner → Retriever → Answerer → Verifier…"):
        result = ask(question.strip(), str(docs_dir), use_llm=use_llm)
    st.session_state.result = result
    st.session_state.result_q = question.strip()

result = st.session_state.get("result")
if not result:
    st.stop()

st.divider()
st.subheader("Answer")
st.caption(f"Question: {st.session_state.get('result_q', '')}")
st.markdown(result.get("answer", "No answer produced.").replace("\n", "  \n"))

if result.get("warning"):
    st.warning(result["warning"])

verdict = result.get("verifier_verdict", {}) or {}
c1, c2, c3 = st.columns(3)
grounded = verdict.get("grounded")
c1.metric("Verifier", "Grounded ✅" if grounded else "Not grounded ⚠️")
coverage = verdict.get("coverage")
c2.metric("Grounding coverage", f"{coverage:.0%}" if isinstance(coverage, (int, float)) else "—")
c3.metric("Retries", result.get("retries", 0))
if verdict.get("reason"):
    st.caption(f"Verifier: {verdict['reason']}")

citations = result.get("citations", [])
if citations:
    st.markdown("**Sources**")
    for citation in citations:
        st.markdown(f"- `{citation}`")

with st.expander("🧭 Agent trace (what each agent did)", expanded=True):
    for step in result.get("trace", []):
        st.markdown(f"- {step}")

retrieved = result.get("retrieved", [])
if retrieved:
    with st.expander(f"🔎 Retrieved chunks ({len(retrieved)})"):
        for chunk in retrieved:
            st.markdown(f"**{chunk['source']} — {chunk['section']}** · BM25 score {chunk.get('score', 0)}")
            st.text(chunk["text"])
