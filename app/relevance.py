"""Relevance check: can this question be answered from the database?"""

from app.sql.prompt import SCHEMA


def build_relevance_prompt(question: str) -> str:
    return f"""You decide whether a user's question can be answered with a read-only SQL SELECT query on this PostgreSQL database.

{SCHEMA}
Answer YES only if the question asks to look up, list, count, aggregate or compare data stored in these tables (products, categories, prices, stock, sales).
Answer NO for everything else: general knowledge, chit-chat, weather, news, and any request to insert, change or delete data.
Reply with exactly one word: YES or NO.

Question: {question}
Answer:"""


def parse_relevance(output: str) -> bool:
    words = output.strip().upper().split()
    return bool(words) and words[0].strip(".,!:*\"'`").startswith("YES")
