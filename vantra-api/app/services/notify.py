"""Notification hooks: stubbed webhook integration point.

On blacklist hits and hard anomalies, fire_notify() logs "would notify: <endpoint>"
and (if a webhook URL is configured) performs a best-effort POST. This is the
visible integration point; delivery is not required for the demo.
"""
from __future__ import annotations

import httpx

from app import db

_schema_done = False


async def ensure_schema() -> None:
    global _schema_done
    if _schema_done:
        return
    await db.execute("""CREATE TABLE IF NOT EXISTS notify_config (
        id INT PRIMARY KEY DEFAULT 1,
        webhook_url TEXT,
        email TEXT,
        enabled BOOLEAN DEFAULT TRUE
    )""")
    await db.execute(
        """INSERT INTO notify_config (id, webhook_url, email, enabled)
           VALUES (1, NULL, NULL, TRUE) ON CONFLICT (id) DO NOTHING""")
    _schema_done = True


async def get_config() -> dict:
    await ensure_schema()
    row = await db.fetch_one("SELECT * FROM notify_config WHERE id = 1")
    return dict(row) if row else {"webhook_url": None, "email": None, "enabled": True}


async def set_config(webhook_url: str | None, email: str | None,
                     enabled: bool = True) -> dict:
    await ensure_schema()
    await db.execute(
        """UPDATE notify_config SET webhook_url = $1, email = $2, enabled = $3
           WHERE id = 1""", [webhook_url or None, email or None, enabled])
    return await get_config()


async def fire_notify(kind: str, title: str, evidence: dict) -> None:
    """The integration point. Called on blacklist hits and hard anomalies."""
    cfg = await get_config()
    if not cfg.get("enabled"):
        print(f"[notify] disabled — would have sent: {kind}: {title}")
        return
    endpoint = cfg.get("webhook_url")
    if endpoint:
        try:
            async with httpx.AsyncClient(timeout=3) as client:
                await client.post(endpoint, json={
                    "kind": kind, "title": title, "evidence": evidence,
                    "source": "vantra"})
            print(f"[notify] POSTed webhook: {kind}: {title}")
        except Exception as e:
            print(f"[notify] webhook failed ({e}) — would notify: {endpoint}")
    else:
        print(f"[notify] would notify: {endpoint or '(no webhook configured)'} "
              f"— {kind}: {title}")
