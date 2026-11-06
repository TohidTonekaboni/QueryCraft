"""Schema-aware prompt in the format SQLCoder was trained on."""

# Keep in sync with ingestion/schema.sql.
SCHEMA = """\
CREATE TABLE categories (
  id   SERIAL PRIMARY KEY,
  name TEXT NOT NULL UNIQUE -- one of: Electronics, Books, Clothing, Home & Kitchen, Sports
);

CREATE TABLE products (
  id          SERIAL PRIMARY KEY,
  sku         TEXT NOT NULL UNIQUE,
  name        TEXT NOT NULL,
  category_id INTEGER NOT NULL REFERENCES categories(id),
  price       NUMERIC(10, 2) NOT NULL, -- current list price
  stock       INTEGER NOT NULL,        -- units in stock
  created_at  TIMESTAMPTZ NOT NULL
);

-- One row per product sold in a sale
CREATE TABLE order_items (
  id         SERIAL PRIMARY KEY,
  product_id INTEGER NOT NULL REFERENCES products(id),
  quantity   INTEGER NOT NULL,
  unit_price NUMERIC(10, 2) NOT NULL, -- price at time of sale; revenue = quantity * unit_price
  sold_at    TIMESTAMPTZ NOT NULL
);
"""

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
