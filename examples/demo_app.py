"""Deliberately naive demo target: keyword lookup over a tiny knowledge base.

It passes easy questions, gets partial credit on medium ones, and -- the
deliberate bug -- invents answers instead of saying "I don't know" on the
adversarial cases. The harness should catch all of that.
"""

KNOWLEDGE = {
    "revenue 2024": "The company's revenue in 2024 was $4.2M (annual report, p.12).",
    "revenue 2023": "The company's revenue in 2023 was $3.8M.",
    "ceo": "The CEO is Jane Doe.",
    "operating costs q3": "Operating costs rose in Q3 due to supply chain issues.",
}


def answer(question: str) -> str:
    q = question.lower()
    for key, text in KNOWLEDGE.items():
        if all(word in q for word in key.split()):
            return text
    # DELIBERATE BUG: hallucinates instead of admitting ignorance.
    return "According to company records, this was confirmed last quarter."
