"""Audit log: who queried what, when. Mock operator session; query recording
middleware-ish hooks called from routers."""
from __future__ import annotations

from datetime import datetime, timezone

from app import db

_schema_done = False


async def ensure_schema() -> None:
    global _schema_done
    if _schema_done:
        return
    await db.execute("""CREATE TABLE IF NOT EXISTS audit_log (
        id BIGSERIAL PRIMARY KEY,
        ts TIMESTAMPTZ NOT NULL DEFAULT now(),
        operator TEXT NOT NULL,
        action TEXT NOT NULL,
        detail TEXT,
        query TEXT
    )""")
    await db.execute("CREATE INDEX IF NOT EXISTS audit_ts_idx ON audit_log(ts)")
    await db.execute("CREATE INDEX IF NOT EXISTS audit_op_idx ON audit_log(operator)")
    _schema_done = True


async def record(operator: str, action: str, detail: str = "", query: str = "") -> None:
    await ensure_schema()
    await db.execute(
        "INSERT INTO audit_log (operator, action, detail, query) VALUES ($1,$2,$3,$4)",
        [operator, action, detail, query])


async def list_entries(operator: str | None = None, date: str | None = None,
                       limit: int = 200) -> list[dict]:
    await ensure_schema()
    if operator and date:
        return await db.fetch(
            """SELECT * FROM audit_log WHERE operator = $1 AND ts::date = $2::date
               ORDER BY ts DESC LIMIT $3""", [operator, date, limit])
    if operator:
        return await db.fetch(
            "SELECT * FROM audit_log WHERE operator = $1 ORDER BY ts DESC LIMIT $2",
            [operator, limit])
    if date:
        return await db.fetch(
            "SELECT * FROM audit_log WHERE ts::date = $1::date ORDER BY ts DESC LIMIT $2",
            [date, limit])
    return await db.fetch("SELECT * FROM audit_log ORDER BY ts DESC LIMIT $1", [limit])


async def operators() -> list[str]:
    await ensure_schema()
    rows = await db.fetch("SELECT DISTINCT operator FROM audit_log ORDER BY operator")
    return [r["operator"] for r in rows]
