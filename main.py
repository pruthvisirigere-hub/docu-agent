#!/usr/bin/env python3
"""CLI for docu-agent.

Usage:
    python main.py ask "How many days of paid leave do I get?"
    python main.py ask "How do I reset my password?" --verbose
    python main.py ask "..." --llm            # OpenRouter answer (needs OPENROUTER_API_KEY)
    python main.py ask "..." --docs ./my-docs  # use a different docs folder
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from docu_agent.graph import ask

DEFAULT_DOCS = Path(__file__).parent / "docs"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="docu-agent",
        description="Multi-agent document QA: ask questions about your Markdown docs.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ask_parser = sub.add_parser("ask", help="Ask a question about the documents.")
    ask_parser.add_argument("question", help="The question to answer.")
    ask_parser.add_argument(
        "--docs", default=str(DEFAULT_DOCS), help="Folder of .md files (default: ./docs)."
    )
    ask_parser.add_argument(
        "--verbose", action="store_true", help="Show each agent's step."
    )
    ask_parser.add_argument(
        "--llm",
        action="store_true",
        help="Answer with OpenRouter instead of extraction (needs OPENROUTER_API_KEY).",
    )

    args = parser.parse_args(argv)

    if args.command == "ask":
        docs_path = Path(args.docs)
        if not docs_path.is_dir() or not list(docs_path.glob("*.md")):
            print(
                f"No .md documents found in '{args.docs}'. "
                "Add Markdown files to the docs folder and try again."
            )
            return 0

        result = ask(args.question, str(docs_path), use_llm=args.llm)

        if args.verbose:
            print("== Agent trace ==")
            for step in result.get("trace", []):
                print(f"  • {step}")
            print()

        print("== Answer ==")
        print(result.get("answer", "No answer produced."))
        print()

        citations = result.get("citations", [])
        if citations:
            print("== Sources ==")
            for citation in citations:
                print(f"  • {citation}")
            print()

        verdict = result.get("verifier_verdict", {})
        print("== Verifier ==")
        print(
            f"  Grounded: {verdict.get('grounded')} "
            f"(coverage {verdict.get('coverage')}) — {verdict.get('reason')}"
        )
        if result.get("warning"):
            print(f"  WARNING: {result['warning']}")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
