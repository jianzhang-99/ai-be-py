"""Redis 会话缓存层。

提供 token -> user_id 的缓存加速，以及会话状态的短期存储。
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

import redis.asyncio as redis

from backend.config import get_settings

logger = logging.getLogger(__name__)

# Session Token 前缀
TOKEN_PREFIX = "session:token:"
# 默认 token 有效期（天）
TOKEN_TTL_DAYS = 7


class SessionStore:
    """Redis 会话存储，支持 token 缓存和会话状态管理。"""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._client: Optional[redis.Redis] = None

    async def _get_client(self) -> redis.Redis:
        """获取或创建 Redis 客户端。"""

        if self._client is None:
            self._client = redis.Redis(
                host=self.settings.redis_host,
                port=self.settings.redis_port,
                db=self.settings.redis_db,
                password=self.settings.redis_password or None,
                decode_responses=True,
                encoding="utf-8",
            )
        return self._client

    async def close(self) -> None:
        """关闭 Redis 连接。"""

        if self._client is not None:
            await self._client.close()
            self._client = None

    async def set_token(
        self,
        token: str,
        user_id: int,
        ttl_days: int = TOKEN_TTL_DAYS,
    ) -> None:
        """缓存 token -> user_id 映射。

        Args:
            token: 认证 token
            user_id: 用户 ID
            ttl_days: 过期天数
        """

        client = await self._get_client()
        key = f"{TOKEN_PREFIX}{token}"
        ttl_seconds = ttl_days * 24 * 3600

        try:
            await client.setex(key, ttl_seconds, str(user_id))
            logger.debug(
                "token cached",
                extra={"user_id": user_id, "ttl_days": ttl_days},
            )
        except redis.RedisError as e:
            logger.error(
                "redis set_token failed",
                extra={"token": "***", "error": str(e)},
            )
            raise

    async def get_user_id_by_token(self, token: str) -> Optional[int]:
        """根据 token 查找用户 ID。

        Args:
            token: 认证 token

        Returns:
            用户 ID，不存在或过期返回 None
        """

        client = await self._get_client()
        key = f"{TOKEN_PREFIX}{token}"

        try:
            user_id_str = await client.get(key)
            if user_id_str is None:
                return None
            return int(user_id_str)
        except redis.RedisError as e:
            logger.error(
                "redis get_user_id_by_token failed",
                extra={"token": "***", "error": str(e)},
            )
            return None

    async def delete_token(self, token: str) -> bool:
        """删除 token 缓存（登出时调用）。

        Args:
            token: 认证 token

        Returns:
            是否删除成功
        """

        client = await self._get_client()
        key = f"{TOKEN_PREFIX}{token}"

        try:
            result = await client.delete(key)
            logger.info("token deleted from cache", extra={"key": key})
            return result > 0
        except redis.RedisError as e:
            logger.error(
                "redis delete_token failed",
                extra={"token": "***", "error": str(e)},
            )
            return False

    async def exists_token(self, token: str) -> bool:
        """检查 token 是否存在且有效。

        Args:
            token: 认证 token

        Returns:
            token 是否有效
        """

        client = await self._get_client()
        key = f"{TOKEN_PREFIX}{token}"

        try:
            return await client.exists(key) > 0
        except redis.RedisError as e:
            logger.error(
                "redis exists_token failed",
                extra={"token": "***", "error": str(e)},
            )
            return False

    async def set_session_data(
        self,
        session_id: str,
        data: dict[str, Any],
        ttl_hours: int = 24,
    ) -> None:
        """存储会话相关数据。

        Args:
            session_id: 会话 ID
            data: 要存储的数据
            ttl_hours: 过期小时数
        """

        client = await self._get_client()
        key = f"session:data:{session_id}"

        try:
            await client.setex(key, ttl_hours * 3600, json.dumps(data, ensure_ascii=False))
            logger.debug(
                "session data cached",
                extra={"session_id": session_id, "ttl_hours": ttl_hours},
            )
        except redis.RedisError as e:
            logger.error(
                "redis set_session_data failed",
                extra={"session_id": session_id, "error": str(e)},
            )
            raise

    async def get_session_data(self, session_id: str) -> Optional[dict[str, Any]]:
        """获取会话数据。

        Args:
            session_id: 会话 ID

        Returns:
            存储的数据，不存在返回 None
        """

        client = await self._get_client()
        key = f"session:data:{session_id}"

        try:
            data_str = await client.get(key)
            if data_str is None:
                return None
            return json.loads(data_str)
        except redis.RedisError as e:
            logger.error(
                "redis get_session_data failed",
                extra={"session_id": session_id, "error": str(e)},
            )
            return None

    async def delete_session_data(self, session_id: str) -> bool:
        """删除会话数据。

        Args:
            session_id: 会话 ID

        Returns:
            是否删除成功
        """

        client = await self._get_client()
        key = f"session:data:{session_id}"

        try:
            result = await client.delete(key)
            return result > 0
        except redis.RedisError as e:
            logger.error(
                "redis delete_session_data failed",
                extra={"session_id": session_id, "error": str(e)},
            )
            return False

    async def health_check(self) -> bool:
        """检查 Redis 连接是否正常。"""

        try:
            client = await self._get_client()
            await client.ping()
            return True
        except redis.RedisError as e:
            logger.error("redis health_check failed", extra={"error": str(e)})
            return False


# 单例模式
_session_store: Optional[SessionStore] = None


def get_session_store() -> SessionStore:
    """获取全局 SessionStore 单例。"""

    global _session_store
    if _session_store is None:
        _session_store = SessionStore()
    return _session_store
