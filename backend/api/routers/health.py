from __future__ import annotations

"""健康检查路由。"""

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check():
    """返回轻量级的存活响应。"""

    return {"status": "ok"}
