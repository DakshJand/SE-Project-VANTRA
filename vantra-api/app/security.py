"""Security middleware: rate limiting + API-key gate for admin endpoints.

- Rate limiting: simple in-memory token bucket per client IP. Demo-grade (single
  process); production would use Redis. Default 120 req/min burst 30.
- Admin gate: /api/tuning*, /api/audit, /api/notify/config POST require the
  X-VANTRA-ADMIN-KEY header to match settings.admin_api_key (default "vantra-admin"
  for the demo; override with the VANTRA_ADMIN_KEY env var).
"""
from __future__ import annotations

import time
from collections import defaultdict

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import settings

import os

RATE_LIMIT_PER_MIN = int(os.environ.get("VANTRA_RATE_LIMIT", "120"))
RATE_BURST = min(int(os.environ.get("VANTRA_RATE_BURST", "30")), RATE_LIMIT_PER_MIN)

# endpoints (prefix-matched) requiring the admin key
ADMIN_PATHS = ("/api/tuning", "/api/audit", "/api/notify/config")


class SecurityMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        self.buckets: dict[str, tuple[float, float]] = defaultdict(lambda: (time.monotonic(), RATE_BURST))

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        client = request.client.host if request.client else "unknown"

        # ---- rate limiting (skip the admin check itself to avoid lockout loops)
        if not path.startswith("/api/tuning"):
            now = time.monotonic()
            last, tokens = self.buckets[client]
            tokens = min(RATE_BURST, tokens + (now - last) * (RATE_LIMIT_PER_MIN / 60.0))
            if tokens < 1.0:
                return JSONResponse(
                    {"detail": "rate limit exceeded — slow down"},
                    status_code=429,
                    headers={"Retry-After": "10"})
            self.buckets[client] = (now, tokens - 1.0)

        # ---- admin gate: mutating requests to admin paths need the API key
        # (GETs stay open so operators can view; saves/mutations are gated)
        is_admin_path = any(path.startswith(p) for p in ADMIN_PATHS)
        write_method = request.method in ("POST", "PUT", "DELETE")
        if is_admin_path and write_method:
            key = request.headers.get("x-vantra-admin-key")
            if key != settings.admin_api_key:
                return JSONResponse(
                    {"detail": "admin API key required (X-VANTRA-ADMIN-KEY header)"},
                    status_code=403)

        return await call_next(request)
