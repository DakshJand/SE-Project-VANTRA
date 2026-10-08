"""Feature-expansion endpoints: compare, what-if, tuning, audit, notify."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

from app import db
from app.services import compare, whatif, tuning, audit, notify

router = APIRouter(tags=["features"])

MOCK_OPERATOR_HEADER = "x-vantra-operator"
DEFAULT_OPERATOR = "op-demo"


def operator(request: Request) -> str:
    return request.headers.get(MOCK_OPERATOR_HEADER) or DEFAULT_OPERATOR


# ------------------------------------------------------------------ compare

class CompareReq(BaseModel):
    plates: list[str]


@router.post("/compare")
async def compare_plates(body: CompareReq, request: Request):
    if len(body.plates) < 2 or len(body.plates) > 6:
        raise HTTPException(400, "provide 2-6 plates")
    await audit.record(operator(request), "compare", f"plates={body.plates}")
    return await compare.compare_plates(body.plates)


# ------------------------------------------------------------------ what-if

@router.get("/whatif/candidates")
async def placement_candidates():
    return whatif._candidate_junctions()


@router.post("/whatif/evaluate")
async def evaluate_placement(request: Request,
                             lat: float = Query(..., ge=-90, le=90),
                             lng: float = Query(..., ge=-180, le=180)):
    await audit.record(operator(request), "whatif", f"lat={lat} lng={lng}")
    return await whatif.evaluate_placement(lat, lng)


# ------------------------------------------------------------------ tuning

class PreviewReq(BaseModel):
    fuzzy_ed_high_conf: float | None = None
    fuzzy_ed_mid_conf: float | None = None
    fuzzy_ed_low_conf: float | None = None
    appearance_floor: float | None = None
    appearance_margin: float | None = None


@router.get("/tuning")
async def get_tuning():
    return {"effective": await tuning.get_effective(),
            "defaults": tuning.DEFAULTS,
            "overrides": await tuning.get_overrides()}


@router.post("/tuning/preview")
async def preview_tuning(body: PreviewReq, request: Request):
    values = {k: v for k, v in body.model_dump().items() if v is not None}
    await audit.record(operator(request), "tuning_preview", f"{values}")
    return await tuning.preview(values)


class SaveReq(BaseModel):
    values: dict


@router.post("/tuning/save")
async def save_tuning(body: SaveReq, request: Request):
    await audit.record(operator(request), "tuning_save", f"{body.values}")
    try:
        await tuning.save_overrides(body.values)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"saved": body.values}


@router.post("/tuning/reset")
async def reset_tuning(request: Request):
    await audit.record(operator(request), "tuning_reset")
    await tuning.reset_overrides()
    return {"reset": True}


# ------------------------------------------------------------------ audit

@router.get("/audit")
async def audit_log(request: Request,
                    op: str | None = Query(None, max_length=64,
                                           pattern=r"^[A-Za-z0-9_\-]+$"),
                    date: str | None = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$")):
    await audit.record(operator(request), "audit_view",
                       f"filter op={op} date={date}")
    return {"entries": await audit.list_entries(op, date),
            "operators": await audit.operators()}


# ------------------------------------------------------------------ notify

class NotifyConfig(BaseModel):
    webhook_url: str | None = None
    email: str | None = None
    enabled: bool = True


@router.get("/notify/config")
async def get_notify_config():
    return await notify.get_config()


@router.post("/notify/config")
async def set_notify_config(body: NotifyConfig, request: Request):
    await audit.record(operator(request), "notify_config", f"webhook={body.webhook_url}")
    return await notify.set_config(body.webhook_url, body.email, body.enabled)


@router.post("/notify/test")
async def test_notify(request: Request):
    await audit.record(operator(request), "notify_test")
    await notify.fire_notify("test", "VANTRA notification test",
                             {"rule": "manual_test"})
    return {"sent": True, "config": await notify.get_config()}
