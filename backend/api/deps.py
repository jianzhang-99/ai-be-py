from __future__ import annotations

"""路由器之间共享的 FastAPI 依赖。"""

from functools import lru_cache

from backend.services.chat_service import ChatService


@lru_cache(maxsize=1)
def get_chat_service() -> ChatService:
    """在请求之间重用单个服务实例。"""

    return ChatService()
