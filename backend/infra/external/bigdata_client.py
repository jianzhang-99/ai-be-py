"""大数据 API 客户端适配器。

提供船舶分页查询等业务的异步 HTTP 调用封装。
基础地址：${bigdata.api.url}
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import httpx

from backend.config import get_settings

logger = logging.getLogger(__name__)


class BigDataApiError(Exception):
    """大数据 API 调用异常，携带错误码和消息。"""

    def __init__(self, code: int, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"[{code}] {message}")


class BigDataClient:
    """大数据 API 异步客户端。"""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.base_url = self.settings.bigdata_api_url.rstrip("/")
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """获取或创建 HTTP 客户端。"""

        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(15.0),  # 大数据 API 超时 15s
                headers={
                    "Content-Type": "application/json",
                },
            )
        return self._client

    async def close(self) -> None:
        """关闭 HTTP 客户端。"""

        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def search_ship_page_data(
        self,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """按条件搜索船舶分页数据。

        POST /ai/searchShipPageData

        Args:
            params: 查询参数，包含筛选条件（如船型、航线、载重等）

        Returns:
            船舶分页数据结果

        Raises:
            BigDataApiError: 5xxx 外部系统错误
        """

        client = await self._get_client()

        try:
            response = await client.post(
                "/ai/searchShipPageData",
                json=params,
            )
            response.raise_for_status()
            data = response.json()

            if data.get("code") != 0:
                logger.error(
                    "bigdata search_ship_page_data failed",
                    extra={"params": params, "response": data},
                )
                raise BigDataApiError(
                    data.get("code", 5000),
                    data.get("msg", "船舶分页查询失败"),
                )

            logger.info(
                "bigdata search_ship_page_data success",
                extra={
                    "total": data.get("data", {}).get("total"),
                    "page_size": len(data.get("data", {}).get("records", [])),
                },
            )
            return data

        except httpx.HTTPStatusError as e:
            logger.error(
                "bigdata search_ship_page_data http error",
                extra={"status": e.response.status_code},
            )
            raise BigDataApiError(5002, f"HTTP错误: {e.response.status_code}") from e
        except Exception as e:
            logger.error(
                "bigdata search_ship_page_data unexpected error",
                extra={"error": str(e)},
            )
            raise BigDataApiError(9000, f"未预期异常: {str(e)}") from e


# 单例模式，方便全局复用
_bigdata_client: Optional[BigDataClient] = None


def get_bigdata_client() -> BigDataClient:
    """获取全局 BigDataClient 单例。"""

    global _bigdata_client
    if _bigdata_client is None:
        _bigdata_client = BigDataClient()
    return _bigdata_client
