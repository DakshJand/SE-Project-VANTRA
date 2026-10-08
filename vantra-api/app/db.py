"""Postgres (PostGIS) access layer. psqlpy (libpq) based async pool."""
from __future__ import annotations

import json
from typing import Any, Sequence

import psqlpy
from psqlpy import ConnectionPool

from app.config import settings

_pool: ConnectionPool | None = None


def pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        _pool = ConnectionPool(
            dsn=settings.database_url,
            max_db_pool_size=10,
        )
    return _pool

def _ps(sql: str) -> str:
    """Convert psycopg-style %s placeholders to postgres $n."""
    if "%s" not in sql:
        return sql
    out = []
    n = 0
    for part in sql.split("%s"):
        out.append(part)
        n += 1
        out.append(f"${n}")
    # last part appended extra
    out.pop()
    return "".join(out)

def _split_sql(sql: str) -> list[str]:
    """Split a SQL script into single statements, ignoring semicolons inside -- comments."""
    stmts: list[str] = []
    cur: list[str] = []
    for line in sql.splitlines():
        code, sep, _comment = line.partition("--")
        cur.append(code)
        if ";" in code:
            *head, tail = cur
            stmt = "\n".join(head + [tail.split(";", 1)[0]])
            if stmt.strip():
                stmts.append(stmt)
            cur = [tail.split(";", 1)[1]]
    rest = "\n".join(cur).strip()
    if rest:
        stmts.append(rest)
    return stmts


async def execute(sql: str, params: Sequence[Any] | None = None) -> None:
    if params is None and len(_split_sql(sql)) > 1:
        for stmt in _split_sql(sql):
            await execute(stmt, [])
        return
    conn = await pool().connection()
    await conn.execute(_ps(sql), params if params else [])


async def fetch(sql: str, params: Sequence[Any] | None = None) -> list[dict[str, Any]]:
    conn = await pool().connection()
    res = await conn.execute(_ps(sql), params if params else [])
    return res.result()


async def fetch_one(sql: str, params: Sequence[Any] | None = None) -> dict[str, Any] | None:
    rows = await fetch(sql, params)
    return rows[0] if rows else None


async def fetch_val(sql: str, params: Sequence[Any] | None = None) -> Any:
    row = await fetch_one(sql, params)
    return next(iter(row.values())) if row else None


async def executemany(sql: str, rows: Sequence[Sequence[Any]]) -> None:
    if not rows:
        return
    conn = await pool().connection()
    await conn.execute_many(_ps(sql), [list(r) for r in rows])


def j(obj: Any) -> str:
    """JSON-encode a parameter for a jsonb column."""
    return json.dumps(obj, default=float)
