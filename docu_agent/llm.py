"""Optional LLM answers via OpenRouter (no API key needed by default).

If the env var OPENROUTER_API_KEY is set and the user passes --llm, the
answerer asks a model to write the answer from the retrieved context
instead of extracting sentences. The verifier still checks the result.

This module uses only the standard library (urllib), so no extra
dependency is needed for the optional path.
"""

from __future__ import annotations

import json
import os
import urllib.request

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = os.environ.get("OPENROUTER_MODEL", "openrouter/free")


def llm_answer(question: str, chunks: list[dict]) -> str | None:
    """Ask OpenRouter for an answer grounded in the retrieved chunks.

    Returns the model's answer text, or None when no key is configured
    or the request fails (the caller falls back to extractive answering).
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return None

    context = "\n\n".join(
        f"[{c['source']} — {c['section']}]\n{c['text']}" for c in chunks
    )
    prompt = (
        "Answer the question using ONLY the context below. "
        "If the context does not contain the answer, say so. "
        "Keep each fact on its own line and cite the source in brackets.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    payload = json.dumps(
        {
            "model": DEFAULT_MODEL,
            "messages": [{"role": "user", "content": prompt}],
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        OPENROUTER_URL,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = json.loads(response.read().decode("utf-8"))
        return body["choices"][0]["message"]["content"].strip()
    except Exception:
        return None
