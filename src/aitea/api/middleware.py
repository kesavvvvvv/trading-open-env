"""API middleware for AITEA."""

from __future__ import annotations

import logging
import time
from typing import Callable

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("aitea.api")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log basic request timing and response status."""

    async def dispatch(self, request: Request, call_next: Callable):
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception as exc:
            logger.exception("Unhandled API exception: %s", exc)
            return JSONResponse(
                status_code=500,
                content={"status": "error", "message": "Internal server error"},
            )
        duration_ms = (time.perf_counter() - start) * 1000.0
        logger.info(
            "%s %s -> %s in %.2fms",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )
        return response


def register_middleware(app) -> None:
    """Attach middleware to a FastAPI app."""
    app.add_middleware(RequestLoggingMiddleware)
