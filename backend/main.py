from __future__ import annotations

"""FastAPI application entrypoint for AI-BE-PY."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.auth.middleware import AuthMiddleware
from backend.config import get_settings
from backend.api.routers import auth, chat, health, session

settings = get_settings()

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


@app.get("/")
async def root():
    """返回简单的服务标识信息。"""

    return {"message": settings.app_name, "version": settings.app_version}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
