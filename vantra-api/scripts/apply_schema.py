"""Apply schema.sql to the configured database."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import db

SCHEMA = Path(__file__).parent / "schema.sql"


async def main() -> None:
    await db.execute(SCHEMA.read_text())
    print("schema applied")


if __name__ == "__main__":
    asyncio.run(main())
