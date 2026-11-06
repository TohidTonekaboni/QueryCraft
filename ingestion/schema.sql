-- QueryCraft schema: exactly three related tables.
--   categories 1 --- * products 1 --- * order_items

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

-- One row per product sold in a sale; unit_price is the price at time of sale.
CREATE TABLE IF NOT EXISTS order_items (
    id         SERIAL PRIMARY KEY,
    product_id INTEGER NOT NULL REFERENCES products(id),
    quantity   INTEGER NOT NULL CHECK (quantity > 0),
    unit_price NUMERIC(10, 2) NOT NULL CHECK (unit_price >= 0),
    sold_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_products_category ON products(category_id);
CREATE INDEX IF NOT EXISTS idx_order_items_product ON order_items(product_id);
