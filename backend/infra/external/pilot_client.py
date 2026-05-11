"""运吨吨 Pilot API 客户端适配器。

提供船舶查询、运单创建等核心业务的异步 HTTP 调用封装。
基础地址：${pilot.api.url}
认证头：phone
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import httpx

from backend.config import get_settings

logger = logging.getLogger(__name__)


class PilotApiError(Exception):
    """Pilot API 调用异常，携带错误码和消息。"""

    def __init__(self, code: int, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"[{code}] {message}")


class PilotClient:
    """运吨吨 Pilot API 异步客户端。"""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.base_url = self.settings.pilot_api_url.rstrip("/")
        self.default_phone = self.settings.pilot_phone
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """获取或创建 HTTP 客户端。"""

        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(30.0),
                headers={
                    "User-Agent": "AI-Service/1.0",
                    "Content-Type": "application/json",
                },
            )
        return self._client

    async def close(self) -> None:
        """关闭 HTTP 客户端。"""

        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _build_headers(self, phone: Optional[str] = None) -> dict[str, str]:
        """构建请求头，包含认证手机号。"""

        return {
            "phone": phone or self.default_phone,
        }

    async def search_ship_by_name(
        self,
        ship_name: str,
        phone: Optional[str] = None,
    ) -> dict[str, Any]:
        """船舶模糊查询。

        GET /pilot-api/ai/searchShipByName?shipName=

        Args:
            ship_name: 船舶名称（支持模糊匹配）
            phone: 认证手机号（可选，默认使用配置中的手机号）

        Returns:
            船舶查询结果列表

        Raises:
            PilotApiError: 5xxx 外部系统错误
        """

        client = await self._get_client()
        params = {"shipName": ship_name}

        try:
            response = await client.get(
                "/pilot-api/ai/searchShipByName",
                params=params,
                headers=self._build_headers(phone),
            )
            response.raise_for_status()
            data = response.json()

            if data.get("code") != 0:
                logger.error(
                    "pilot search_ship_by_name failed",
                    extra={"ship_name": ship_name, "response": data},
                )
                raise PilotApiError(
                    data.get("code", 5000),
                    data.get("msg", "船舶查询失败"),
                )

            logger.info(
                "pilot search_ship_by_name success",
                extra={"ship_name": ship_name, "count": len(data.get("data", []))},
            )
            return data

        except httpx.HTTPStatusError as e:
            logger.error(
                "pilot search_ship_by_name http error",
                extra={"ship_name": ship_name, "status": e.response.status_code},
            )
            raise PilotApiError(5001, f"HTTP错误: {e.response.status_code}") from e
        except Exception as e:
            logger.error(
                "pilot search_ship_by_name unexpected error",
                extra={"ship_name": ship_name, "error": str(e)},
            )
            raise PilotApiError(9000, f"未预期异常: {str(e)}") from e

    async def get_bigdata_ship_by_name(
        self,
        ship_name: str,
        phone: Optional[str] = None,
    ) -> dict[str, Any]:
        """获取大数据船舶信息。

        GET /pilot-api/ai/bigDataShip/getByName?shipName=

        Args:
            ship_name: 船舶名称
            phone: 认证手机号（可选）

        Returns:
            大数据船舶详细信息

        Raises:
            PilotApiError: 5xxx 外部系统错误
        """

        client = await self._get_client()
        params = {"shipName": ship_name}

        try:
            response = await client.get(
                "/pilot-api/ai/bigDataShip/getByName",
                params=params,
                headers=self._build_headers(phone),
            )
            response.raise_for_status()
            data = response.json()

            if data.get("code") != 0:
                logger.error(
                    "pilot get_bigdata_ship_by_name failed",
                    extra={"ship_name": ship_name, "response": data},
                )
                raise PilotApiError(
                    data.get("code", 5000),
                    data.get("msg", "大数据船舶查询失败"),
                )

            logger.info(
                "pilot get_bigdata_ship_by_name success",
                extra={"ship_name": ship_name},
            )
            return data

        except httpx.HTTPStatusError as e:
            logger.error(
                "pilot get_bigdata_ship_by_name http error",
                extra={"ship_name": ship_name, "status": e.response.status_code},
            )
            raise PilotApiError(5001, f"HTTP错误: {e.response.status_code}") from e
        except Exception as e:
            logger.error(
                "pilot get_bigdata_ship_by_name unexpected error",
                extra={"ship_name": ship_name, "error": str(e)},
            )
            raise PilotApiError(9000, f"未预期异常: {str(e)}") from e

    async def create_order(
        self,
        order_data: dict[str, Any],
        phone: Optional[str] = None,
    ) -> dict[str, Any]:
        """创建运单。

        POST /pilot-api/ai/order

        Args:
            order_data: 运单数据，包含装卸货地点、日期、货物信息等
            phone: 认证手机号（可选）

        Returns:
            创建结果，包含运单ID

        Raises:
            PilotApiError: 5xxx 外部系统错误
        """

        client = await self._get_client()

        try:
            response = await client.post(
                "/pilot-api/ai/order",
                json=order_data,
                headers=self._build_headers(phone),
            )
            response.raise_for_status()
            data = response.json()

            if data.get("code") != 0:
                logger.error(
                    "pilot create_order failed",
                    extra={"order_data": order_data, "response": data},
                )
                raise PilotApiError(
                    data.get("code", 5000),
                    data.get("msg", "创建运单失败"),
                )

            logger.info(
                "pilot create_order success",
                extra={"order_id": data.get("data", {}).get("id")},
            )
            return data

        except httpx.HTTPStatusError as e:
            logger.error(
                "pilot create_order http error",
                extra={"status": e.response.status_code},
            )
            raise PilotApiError(5001, f"HTTP错误: {e.response.status_code}") from e
        except Exception as e:
            logger.error(
                "pilot create_order unexpected error",
                extra={"error": str(e)},
            )
            raise PilotApiError(9000, f"未预期异常: {str(e)}") from e

    async def get_order(
        self,
        order_id: str,
        phone: Optional[str] = None,
    ) -> dict[str, Any]:
        """查询运单。

        GET /pilot-api/ai/order/{orderId

        Args:
            order_id: 运单ID
            phone: 认证手机号（可选）

        Returns:
            运单详情

        Raises:
            PilotApiError: 5xxx 外部系统错误
        """

        client = await self._get_client()

        try:
            response = await client.get(
                f"/pilot-api/ai/order/{order_id}",
                headers=self._build_headers(phone),
            )
            response.raise_for_status()
            data = response.json()

            if data.get("code") != 0:
                logger.error(
                    "pilot get_order failed",
                    extra={"order_id": order_id, "response": data},
                )
                raise PilotApiError(
                    data.get("code", 5000),
                    data.get("msg", "查询运单失败"),
                )

            logger.info(
                "pilot get_order success",
                extra={"order_id": order_id},
            )
            return data

        except httpx.HTTPStatusError as e:
            logger.error(
                "pilot get_order http error",
                extra={"order_id": order_id, "status": e.response.status_code},
            )
            raise PilotApiError(5001, f"HTTP错误: {e.response.status_code}") from e
        except Exception as e:
            logger.error(
                "pilot get_order unexpected error",
                extra={"order_id": order_id, "error": str(e)},
            )
            raise PilotApiError(9000, f"未预期异常: {str(e)}") from e


# 单例模式，方便全局复用
_pilot_client: Optional[PilotClient] = None


def get_pilot_client() -> PilotClient:
    """获取全局 PilotClient 单例。"""

    global _pilot_client
    if _pilot_client is None:
        _pilot_client = PilotClient()
    return _pilot_client
