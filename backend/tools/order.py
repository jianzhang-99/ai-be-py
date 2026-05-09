from __future__ import annotations

"""第一阶段MVP的订单提取工具。"""

import re


class OrderTool:
    """从自由文本中提取轻量级结构化订单预览。"""

    async def run(self, payload: dict[str, str]) -> dict[str, str]:
        """解析第一阶段所需的核心订单字段。"""

        text = payload.get("text") or ""
        route_match = re.search(
            r"(?:\d{1,2}月\d{1,2}日)?(?P<from>[\u4e00-\u9fa5]{2,8})装.*到(?P<to>[\u4e00-\u9fa5]{2,8})",
            text,
        )
        cargo_match = re.search(r"装(?P<cargo>[\u4e00-\u9fa5]{1,8})(?P<tons>\d{3,6})吨", text)
        date_match = re.search(r"(\d{1,2}月\d{1,2}日)", text)

        loading_port = route_match.group("from") if route_match else "武汉"
        discharge_port = route_match.group("to") if route_match else "南京"
        cargo_name = cargo_match.group("cargo") if cargo_match else "煤"
        tons = cargo_match.group("tons") if cargo_match else "5000"
        shipping_date = date_match.group(1) if date_match else "待确认"

        return {
            "loading_port": loading_port,
            "discharge_port": discharge_port,
            "cargo_name": cargo_name,
            "tons": tons,
            "shipping_date": shipping_date,
            "summary": (
                f"已提取运单预览：{shipping_date}{loading_port}装{cargo_name}{tons}吨，"
                f"目的地{discharge_port}。"
            ),
        }
