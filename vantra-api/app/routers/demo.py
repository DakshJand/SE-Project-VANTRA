"""Demo scenario endpoints: seeded scenario metadata + pointers."""
from fastapi import APIRouter

from app import db

router = APIRouter(tags=["demo"])


@router.get("/demo/scenarios")
async def list_scenarios():
    return await db.fetch(
        "SELECT key, title, description, plate, payload FROM demo_scenarios ORDER BY key")
