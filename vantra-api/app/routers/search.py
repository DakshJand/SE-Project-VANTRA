"""Plate search endpoints (exact, fuzzy, wildcard)."""
from fastapi import APIRouter, HTTPException, Query

from app import db
from app.services.matcher import edit_distance

router = APIRouter(tags=["search"])


def _like(pattern: str) -> str:
    """Translate user wildcard (* any string, ? any char) to SQL LIKE.
    Escapes SQL wildcards AND the escape character itself."""
    sql = pattern.replace("\\", "\\\\")
    sql = sql.replace("%", r"\%").replace("_", r"\_")
    sql = sql.replace("*", "%").replace("?", "_")
    return sql


@router.get("/search")
async def search_plates(q: str = Query(..., min_length=2, max_length=16,
                                       pattern=r"^[A-Za-z0-9*?\-]+$"),
                        fuzzy: bool = True,
                        limit: int = Query(50, ge=1, le=200)):
    """Multi-plate / partial-plate wildcard search. Supports * and ? wildcards.
    With fuzzy=true, also returns near matches (edit distance <= 2)."""
    q = q.upper()
    if "*" in q or "?" in q:
        rows = await db.fetch(
            """SELECT DISTINCT plate_raw AS plate, count(*) AS n_detections,
                      min(ts) AS first_seen, max(ts) AS last_seen
               FROM detections WHERE plate_raw LIKE $1 ESCAPE '\\'
               GROUP BY plate_raw ORDER BY plate_raw LIMIT $2""",
            [_like(q), limit])
        return {"query": q, "matches": rows}

    # exact
    exact = await db.fetch(
        """SELECT plate_raw AS plate, count(*) AS n_detections,
                  min(ts) AS first_seen, max(ts) AS last_seen
           FROM detections WHERE plate_raw = $1 GROUP BY plate_raw""", [q])
    if exact and not fuzzy:
        return {"query": q, "matches": exact}

    # no exact hit and no explicit wildcard -> treat as prefix search, so a
    # partial plate like "KA01" still finds every plate starting with it
    if not exact and "*" not in q and "?" not in q:
        prefix = await db.fetch(
            """SELECT plate_raw AS plate, count(*) AS n_detections,
                      min(ts) AS first_seen, max(ts) AS last_seen
               FROM detections WHERE plate_raw LIKE $1 ESCAPE '\\'
               GROUP BY plate_raw ORDER BY plate_raw LIMIT $2""",
            [_like(q + "*"), limit])
        if prefix:
            return {"query": q, "matches": prefix, "prefix_match": True}

    # fuzzy: fetch distinct plates, compute edit distance server-side
    plates = await db.fetch(
        """SELECT plate_raw AS plate, count(*) AS n_detections,
                  min(ts) AS first_seen, max(ts) AS last_seen
           FROM detections WHERE plate_raw IS NOT NULL
           GROUP BY plate_raw LIMIT 5000""")
    matches = list(exact)
    seen = {m["plate"] for m in matches}
    for p in plates:
        if p["plate"] in seen:
            continue
        if abs(len(p["plate"]) - len(q)) <= 2 and edit_distance(p["plate"], q) <= 2:
            matches.append(p)
            seen.add(p["plate"])
        if len(matches) >= limit:
            break
    matches.sort(key=lambda m: 0 if m["plate"] == q else edit_distance(m["plate"], q))
    return {"query": q, "matches": matches[:limit]}
