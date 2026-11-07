"""Schema-aware prompt in the format SQLCoder was trained on."""

import re
from pathlib import Path

SCHEMA_FILE = Path(__file__).resolve().parents[2] / "ingestion" / "schema.sql"


def load_schema(path: Path = SCHEMA_FILE) -> str:
    """Prompt schema derived from ingestion/schema.sql, the single source of truth.

    Keeps the tables and their comments (the comments carry hints for the model);
    drops the file header, `IF NOT EXISTS` and the index statements."""
    text = path.read_text()
    text = text[text.index("CREATE TABLE"):]  # drop the header comment
    text = re.sub(r"^CREATE INDEX .*\n?", "", text, flags=re.MULTILINE)
    text = text.replace("IF NOT EXISTS ", "")
    return text.strip() + "\n"


SCHEMA = load_schema()

STOP = ["[/SQL]", "```"]

# Business rules the schema alone does not make obvious; SQLCoder-7B tends to miss them.
HINTS = """- Revenue, total sales or income = SUM(order_items.quantity * order_items.unit_price). Never SUM(unit_price) alone: always multiply by quantity.
- Units sold or number of items sold = SUM(order_items.quantity).
- A product was "sold" only if it has rows in order_items, so JOIN order_items for any question about sold, sales, revenue or orders. Products without sales have no order_items rows.
- Category names are in categories.name; join through products.category_id.
- "How many", "number of" and "count" questions need COUNT(...), returning a single number, never a list of rows.
- Group by the category or product name you select, and use ORDER BY with LIMIT for "top", "most" and "least" questions."""


def build_prompt(question: str, previous_sql: str | None = None, error: str | None = None) -> str:
    task = f"Generate a PostgreSQL query to answer [QUESTION]{question}[/QUESTION]"
    if previous_sql and error:
        task += (
            f"\n\nA previous attempt failed.\nPrevious query:\n{previous_sql}\n"
            f"Error:\n{error}\nWrite a corrected query."
        )
    return f"""### Task
{task}

### Database Schema
The query will run on a database with the following schema:
{SCHEMA}
### Hints
{HINTS}

### Answer
Given the database schema, here is the SQL query that answers [QUESTION]{question}[/QUESTION]
[SQL]
"""


def extract_sql(text: str) -> str:
    """Strip markdown fences / stop markers the model may emit around the query."""
    text = text.strip()
    if text.lower().startswith("```sql"):
        text = text[6:]
    for marker in STOP:
        text = text.split(marker, 1)[0]
    return text.strip()
