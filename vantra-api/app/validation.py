"""Input validation helpers shared by routers."""
from __future__ import annotations

import re

# plates: uppercase alphanumerics; queries may include * and ? wildcards
PLATE_QUERY_RE = re.compile(r"^[A-Z0-9*?\-]{2,16}$")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def validate_plate_query(q: str) -> str | None:
    """Return an error message if a plate query string is invalid."""
    if not q or len(q) > 16:
        return "query must be 1-16 characters"
    if not PLATE_QUERY_RE.match(q):
        return "query may contain only A-Z, 0-9, * ? and -"
    return None


def validate_date(d: str) -> str | None:
    if d and not DATE_RE.match(d):
        return "date must be YYYY-MM-DD"
    return None


def validate_latlng(lat: float, lng: float) -> str | None:
    if not (-90 <= lat <= 90 and -180 <= lng <= 180):
        return "lat/lng out of range"
    return None
