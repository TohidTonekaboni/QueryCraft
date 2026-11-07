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
