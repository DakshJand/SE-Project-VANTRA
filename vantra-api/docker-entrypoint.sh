#!/bin/sh
# VANTRA API container entrypoint: wait for DB, init schema, seed on first boot, serve.

echo "waiting for database..."
python - <<'EOF'
import asyncio
import sys
import time

sys.path.insert(0, ".")
from app import db


async def wait():
    for _ in range(90):
        try:
            await db.execute("SELECT 1")
            return
        except Exception:
            time.sleep(2)
    sys.exit("database never became reachable")


asyncio.run(wait())
EOF

python scripts/apply_schema.py

# extensions (may already exist)
python - <<'EOF'
import asyncio
import sys

sys.path.insert(0, ".")
from app import db


async def ext():
    for s in ("CREATE EXTENSION IF NOT EXISTS fuzzystrmatch",
              "CREATE EXTENSION IF NOT EXISTS btree_gist"):
        try:
            await db.execute(s)
        except Exception as e:
            print("ext skip:", e)


asyncio.run(ext())
EOF

# seed only if empty (guard against set -e: the check intentionally exits 1 when empty)
set +e
python - <<'EOF'
import asyncio
import sys

sys.path.insert(0, ".")
from app import db


async def count():
    return await db.fetch_val("SELECT count(*) FROM detections")


n = asyncio.run(count())
print(f"detections: {n}")
sys.exit(0 if n and n > 1000 else 1)
EOF
NEEDS_SEED=$?
set -e
if [ "$NEEDS_SEED" -ne 0 ]; then
    echo "seeding demo dataset (~90s)..."
    python scripts/seed_history.py --fast --days 21
    python scripts/seed_demo.py
fi

exec uvicorn app.main:app --host 0.0.0.0 --port 8712
