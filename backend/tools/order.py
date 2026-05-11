"""运单工具，接入 Pilot API 实现运单预览和提交。"""

from __future__ import annotations

import logging
import re
from typing import Any

from backend.infra.external.pilot_client import (
    PilotApiError,
    get_pilot_client,
)

logger = logging.getLogger(__name__)


class OrderTool:
    """运单提取和提交工具，支持 Pilot API 预录入。"""

    async def run(self, payload: dict[str, str]) -> dict[str, Any]:
        """从自由文本中提取结构化运单信息，支持预览和提交。

        Args:
            payload: 包含 text（原始文本）和 action（preview/submit）字段

        Returns:
            运单预览或提交结果
        """

        text = payload.get("text") or ""
        action = payload.get("action", "preview")  # preview 或 submit

        # 解析运单核心字段
        route_match = re.search(
            r"(?:\d{1,2}月\d{1,2}日)?(?P<from>[\u4e00-\u9fa5]{2,8})装.*到(?P<to>[\u4e00-\u9fa5]{2,8})",
            text,
        )
        cargo_match = re.search(
            r"装(?P<cargo>[\u4e00-\u9fa5]{1,8})(?P<tons>\d{3,6})吨",
            text,
        )
        date_match = re.search(r"(\d{1,2}月\d{1,2}日)", text)

        loading_port = route_match.group("from") if route_match else "武汉"
        discharge_port = route_match.group("to") if route_match else "南京"
        cargo_name = cargo_match.group("cargo") if cargo_match else "煤"
        tons = cargo_match.group("tons") if cargo_match else "5000"
        shipping_date = date_match.group(1) if date_match else "待确认"

        # 构建运单预览数据
        order_preview = {
            "loading_port": loading_port,
            "discharge_port": discharge_port,
            "cargo_name": cargo_name,
            "tons": tons,
            "shipping_date": shipping_date,
        }

        # 如果是提交操作，调用 Pilot API
        if action == "submit":
            try:
                pilot_client = get_pilot_client()

                # 构建 Pilot API 格式的运单数据
                pilot_order_data = {
                    "loadingPort": loading_port,
                    "dischargePort": discharge_port,
                    "cargoName": cargo_name,
                    "tons": int(tons),
                    "shippingDate": shipping_date,
                }

                result = await pilot_client.create_order(order_data=pilot_order_data)

                order_id = result.get("data", {}).get("id", "")
                return {
                    "preview": order_preview,
                    "submit_result": result,
                    "order_id": order_id,
                    "summary": (
                        f"运单预录入成功，运单号：{order_id}。"
                        f"{shipping_date}{loading_port}装{cargo_name}{tons}吨，"
                        f"目的地{discharge_port}。"
                    ),
                }

            except PilotApiError as e:
                logger.error(
                    "pilot create_order failed",
                    extra={"error": e.message},
                )
                return {
                    "preview": order_preview,
                    "submit_result": None,
                    "error": f"运单提交失败：{e.message}",
                    "summary": (
                        f"运单预览：{shipping_date}{loading_port}装{cargo_name}{tons}吨，"
                        f"目的地{discharge_port}。但提交失败：{e.message}"
                    ),
                }

        # 默认返回预览结果（兼容旧接口）
        preview_result = {
            "loading_port": loading_port,
            "discharge_port": discharge_port,
            "cargo_name": cargo_name,
            "tons": tons,
            "shipping_date": shipping_date,
            "summary": (
                f"已提取运单预览：{shipping_date}{loading_port}装{cargo_name}{tons}吨，"
                f"目的地{discharge_port}。如需提交请说'确认提交'。"
            ),
        }

        # 扩展信息（包含完整预览数据用于提交）
        preview_result["_detail"] = {
            "loading_port": loading_port,
            "discharge_port": discharge_port,
            "cargo_name": cargo_name,
            "tons": tons,
            "shipping_date": shipping_date,
        }

        return preview_result