"""SQL safety validation using sqlglot (AST based, not regex)."""

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

# Statement/clause types that must never appear anywhere in the tree.
_FORBIDDEN_NODES = (
    exp.Insert, exp.Update, exp.Delete, exp.Merge, exp.Create, exp.Drop, exp.Alter,
    exp.TruncateTable, exp.Command, exp.Set, exp.Grant, exp.Copy, exp.Into, exp.Lock,
)
# Functions with side effects or file/network/server access.
_FORBIDDEN_FUNC_PREFIXES = ("pg_", "lo_", "dblink", "set_config", "current_setting", "txid_")


def validate_sql(sql: str) -> str | None:
    """Return None if `sql` is one read-only SELECT, else a message for the correction prompt."""
    if not sql.strip():
        return "Empty query."
    try:
        statements = [s for s in sqlglot.parse(sql, dialect="postgres") if s is not None]
    except SqlglotError as exc:
        return f"SQL syntax error: {str(exc).splitlines()[0]}"

    if len(statements) != 1:
        return "Exactly one SQL statement is allowed."
    stmt = statements[0]
    if not isinstance(stmt, (exp.Select, exp.Union, exp.Intersect, exp.Except)):
        return "Only SELECT statements are allowed."

    for node in stmt.walk():
        if isinstance(node, _FORBIDDEN_NODES):
            return f"{type(node).__name__} is not allowed; only read-only SELECT queries are allowed."
        if isinstance(node, exp.Func):
            name = (node.name if isinstance(node, exp.Anonymous) else node.sql_name()).lower()
            if name.startswith(_FORBIDDEN_FUNC_PREFIXES):
                return f"Function {name}() is not allowed."
    return None
