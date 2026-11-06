"""Seed the QueryCraft database with products.

Waits for the Postgres container, creates the schema if missing, and inserts
products until the table holds TARGET_PRODUCTS rows. If that many already
exist, ingestion is skipped. Safe to re-run or resume after an interruption:
each product has a deterministic SKU and inserts use ON CONFLICT DO NOTHING.

Usage: pip install "psycopg[binary]" python-dotenv && python ingestion.py
"""

import os
import random
import sys
import time

import psycopg
from dotenv import load_dotenv

load_dotenv()

DB = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": int(os.getenv("POSTGRES_PORT", "5432")),
    "user": os.environ["POSTGRES_USER"],
    "password": os.environ["POSTGRES_PASSWORD"],
    "dbname": os.environ["POSTGRES_DB"],
}
TARGET_PRODUCTS = int(os.getenv("TARGET_PRODUCTS", "1000"))
WAIT_TIMEOUT = int(os.getenv("DB_WAIT_TIMEOUT", "60"))
BATCH_SIZE = 100

# Three related tables: categories <- products <- order_items.
SCHEMA = """
CREATE TABLE IF NOT EXISTS categories (
    id   SERIAL PRIMARY KEY,
    name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS products (
    id          SERIAL PRIMARY KEY,
    sku         TEXT NOT NULL UNIQUE,
    name        TEXT NOT NULL,
    category_id INTEGER NOT NULL REFERENCES categories(id),
    price       NUMERIC(10, 2) NOT NULL CHECK (price >= 0),
    stock       INTEGER NOT NULL DEFAULT 0 CHECK (stock >= 0),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS order_items (
    id         SERIAL PRIMARY KEY,
    product_id INTEGER NOT NULL REFERENCES products(id),
    quantity   INTEGER NOT NULL CHECK (quantity > 0),
    unit_price NUMERIC(10, 2) NOT NULL CHECK (unit_price >= 0),
    sold_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_products_category ON products(category_id);
CREATE INDEX IF NOT EXISTS idx_order_items_product ON order_items(product_id);
"""

# category -> (price range, product nouns)
CATALOG = {
    "Electronics": ((20, 1500), ["Laptop", "Phone", "Headphones", "Monitor", "Keyboard", "Camera", "Speaker", "Tablet"]),
    "Books": ((5, 60), ["Novel", "Cookbook", "Biography", "Textbook", "Comic", "Atlas", "Dictionary"]),
    "Clothing": ((10, 200), ["T-Shirt", "Jacket", "Jeans", "Sneakers", "Scarf", "Hat", "Sweater"]),
    "Home & Kitchen": ((8, 400), ["Blender", "Lamp", "Kettle", "Pan", "Chair", "Vacuum", "Toaster"]),
    "Sports": ((10, 500), ["Yoga Mat", "Dumbbell", "Football", "Tent", "Bicycle", "Racket", "Helmet"]),
}
ADJECTIVES = ["Classic", "Pro", "Ultra", "Mini", "Eco", "Smart", "Deluxe", "Basic", "Premium", "Compact"]


def connect() -> psycopg.Connection:
    """Retry until Postgres accepts connections (container may still be starting)."""
    deadline = time.monotonic() + WAIT_TIMEOUT
    while True:
        try:
            return psycopg.connect(**DB, connect_timeout=3)
        except psycopg.OperationalError as exc:
            if time.monotonic() > deadline:
                sys.exit(f"Postgres not reachable after {WAIT_TIMEOUT}s: {exc}")
            print("Waiting for Postgres...", flush=True)
            time.sleep(2)


def make_product(index: int, category_ids: dict[str, int], names: list[str]) -> tuple:
    """Deterministic product for a given index (same index -> same row)."""
    rng = random.Random(index)
    category = names[index % len(names)]
    (low, high), nouns = CATALOG[category]
    name = f"{rng.choice(ADJECTIVES)} {rng.choice(nouns)} {index:04d}"
    price = round(rng.uniform(low, high), 2)
    return (f"SKU-{index:05d}", name, category_ids[category], price, rng.randint(0, 500))


def main() -> None:
    with connect() as conn:
        conn.execute(SCHEMA)
        for name in CATALOG:
            conn.execute(
                "INSERT INTO categories (name) VALUES (%s) ON CONFLICT (name) DO NOTHING", (name,)
            )
        category_ids = dict(conn.execute("SELECT name, id FROM categories").fetchall())
        names = list(CATALOG)

        existing = conn.execute("SELECT count(*) FROM products").fetchone()[0]
        if existing >= TARGET_PRODUCTS:
            print(f"products already has {existing} rows (target {TARGET_PRODUCTS}); skipping ingestion.")
            return

        print(f"products has {existing}/{TARGET_PRODUCTS} rows; ingesting...")
        with conn.cursor() as cur:
            for start in range(1, TARGET_PRODUCTS + 1, BATCH_SIZE):
                stop = min(start + BATCH_SIZE, TARGET_PRODUCTS + 1)
                rows = [make_product(i, category_ids, names) for i in range(start, stop)]
                cur.executemany(
                    "INSERT INTO products (sku, name, category_id, price, stock) "
                    "VALUES (%s, %s, %s, %s, %s) ON CONFLICT (sku) DO NOTHING",
                    rows,
                )
                conn.commit()  # commit per batch so an interrupted run can resume

        total = conn.execute("SELECT count(*) FROM products").fetchone()[0]
        print(f"Done. products now has {total} rows.")


if __name__ == "__main__":
    main()
