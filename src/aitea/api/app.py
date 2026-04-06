"""FastAPI application for AITEA."""

from __future__ import annotations

from fastapi import FastAPI

from ..env.aitea_env import AITEAEnv
from ..utils.constants import STATUS_OK
from .middleware import register_middleware
from .routes import router


def create_app() -> FastAPI:
    app = FastAPI(title="AITEA API", version="0.1.0")

    register_middleware(app)
    app.include_router(router)

    @app.on_event("startup")
    async def _startup() -> None:
        app.state.env = AITEAEnv(auto_reset=True)

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        env = getattr(app.state, "env", None)
        if env is not None:
            env.close()

    @app.get("/")
    async def root():
        return {"status": STATUS_OK, "service": "aitea"}

    return app


app = create_app()
