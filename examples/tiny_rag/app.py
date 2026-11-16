"""Tiny RAG: the smallest app that shows how to plug a target into evalX.

evalX calls one function, ``answer(question) -> str``; everything else in
this file is the app's own business and invisible to evalX. Retrieval is a
keyword lookup over three manual excerpts, and ``generate`` stands in for an
LLM call so the example runs offline with no API key.

Switch the "model" with the TINY_RAG_MODEL environment variable to watch
evalX react to a change:

- ``v1`` (default): quotes the whole retrieved passage.
- ``v2``: keeps only the first sentence. Shorter, and it no longer repeats
  the manual's warning about ft-lb, so one case improves.
- ``v3``: like v2, but answers off-topic questions instead of refusing.
  One case regresses and the baseline gate fails.
"""

import os

DOCS = {
    "torque": "AN3 bolts are torqued to 25 in-lb. Never use ft-lb values from automotive charts.",
    "tire": "Main wheel tires are inflated with dry nitrogen to 42 psi.",
    "cable": "Replace a control cable if 3 or more wires are broken in any 1-inch length.",
}

REFUSAL = "I can't answer that from the manual."


def retrieve(question: str) -> str | None:
    """The passage whose topic word appears in the question, if any."""
    q = question.lower()
    for topic, text in DOCS.items():
        if topic in q:
            return text
    return None


def generate(question: str, context: str | None, model: str) -> str:
    """Stand-in for an LLM call: write an answer from the retrieved context."""
    if context is None:
        if model == "v3":
            return "Paris is the capital of France."  # v3 answers off-topic
        return REFUSAL
    if model == "v1":
        return f"According to the manual: {context}"
    return context.split(". ")[0].rstrip(".") + "."


def answer(question: str) -> str:
    """The one function evalX calls: --target examples.tiny_rag.app:answer"""
    model = os.environ.get("TINY_RAG_MODEL", "v1")
    return generate(question, retrieve(question), model)
