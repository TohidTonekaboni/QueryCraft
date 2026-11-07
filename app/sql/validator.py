"""SQL safety validation using sqlglot (AST based, not regex)."""

import re

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

from app.sql.prompt import SCHEMA

# Only the application's own tables may be read (derived from ingestion/schema.sql, like the prompt).
# This keeps catalog and metadata tables (pg_user, information_schema.*, ...) out of reach.
ALLOWED_TABLES = frozenset(re.findall(r"CREATE TABLE (\w+)", SCHEMA))

# Statement/clause types that must never appear anywhere in the tree.
_FORBIDDEN_NODES = (
    exp.Insert, exp.Update, exp.Delete, exp.Merge, exp.Create, exp.Drop, exp.Alter,
    exp.TruncateTable, exp.Command, exp.Set, exp.Grant, exp.Copy, exp.Into, exp.Lock,
)
# Functions with side effects or file/network/server access.
_FORBIDDEN_FUNC_PREFIXES = ("pg_", "lo_", "dblink", "set_config", "current_setting", "txid_")
# Server-info, sequence and row-explosion functions that no business question needs.
_FORBIDDEN_FUNCS = frozenset({
    "version", "nextval", "currval", "setval", "lastval", "generate_series",
    "current_user", "session_user", "current_database", "current_schema", "inet_server_addr",
})


# Fixed set of rule names, used as a metric label.
RULES = ("syntax", "multi_statement", "not_select", "forbidden_table", "forbidden_node", "forbidden_function")


def validate_sql(sql: str) -> str | None:
    """Return None if `sql` is one read-only SELECT, else a message for the correction prompt."""
    rejection = check_sql(sql)
    return rejection[1] if rejection else None


def check_sql(sql: str) -> tuple[str, str] | None:
    """Return None if `sql` is one read-only SELECT, else (rule, message); `rule` is one of RULES."""
    if not sql.strip():
        return "syntax", "Empty query."
    try:
        statements = [s for s in sqlglot.parse(sql, dialect="postgres") if s is not None]
    except SqlglotError as exc:
        return "syntax", f"SQL syntax error: {str(exc).splitlines()[0]}"

    if len(statements) != 1:
        return "multi_statement", "Exactly one SQL statement is allowed."
    stmt = statements[0]
    if not isinstance(stmt, (exp.Select, exp.Union, exp.Intersect, exp.Except)):
        return "not_select", "Only SELECT statements are allowed."

    ctes = {cte.alias_or_name.lower() for cte in stmt.find_all(exp.CTE)}
    for table in stmt.find_all(exp.Table):
        if not isinstance(table.this, exp.Identifier):
            return "forbidden_table", "Table functions are not allowed; query the tables directly."
        name = table.name.lower()
        if name in ctes and not table.db:
            continue
        if table.db.lower() not in ("", "public") or name not in ALLOWED_TABLES:
            return "forbidden_table", (
                f"Table {table.sql(dialect='postgres')} is not allowed; use only: {', '.join(sorted(ALLOWED_TABLES))}.")

    for node in stmt.walk():
        if isinstance(node, _FORBIDDEN_NODES):
            return "forbidden_node", f"{type(node).__name__} is not allowed; only read-only SELECT queries are allowed."
        if isinstance(node, exp.Func):
            # The name Postgres sees (sqlglot's internal names differ, e.g. EXPLODING_GENERATE_SERIES).
            name = node.name.lower() if isinstance(node, exp.Anonymous) else node.sql("postgres").split("(")[0].strip().lower()
            if name in _FORBIDDEN_FUNCS or name.startswith(_FORBIDDEN_FUNC_PREFIXES):
                return "forbidden_function", f"Function {name}() is not allowed."
    return None
