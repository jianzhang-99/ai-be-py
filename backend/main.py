"""FastAPI application entrypoint for AI-BE-PY."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import Response

from backend.auth.middleware import AuthMiddleware
from backend.config import get_settings
from backend.api.routers import auth, chat, health, input, oss, session
from backend.api.routers import ship as ship_router
from backend.api.routers import order as order_router
from backend.api.routers import ai_be_compat
from backend.infra.observability import get_metrics, get_metrics_content_type, setup_logging, setup_tracer

settings = get_settings()

# 初始化可观测性组件
setup_logging()
setup_tracer()

app = FastAPI(
    title=settings.app_name,
    description="航运领域 AI Agent 服务",
    version=settings.app_version,
    debug=settings.debug,
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(AuthMiddleware)

# 注册路由
app.include_router(auth.router)
app.include_router(chat.router)
app.include_router(health.router)
app.include_router(session.router)
app.include_router(input.router)
app.include_router(oss.router)
app.include_router(ship_router.router)
app.include_router(order_router.router)
app.include_router(ai_be_compat.router)


@app.get("/")
async def root():
    """返回简单的服务标识信息。"""

    return {"message": settings.app_name, "version": settings.app_version}


@app.get("/metrics")
async def metrics():
    """暴露 Prometheus 指标。"""
    return Response(
        content=get_metrics(),
        media_type=get_metrics_content_type(),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
