from __future__ import annotations

"""船舶查询工具，带有可预测的模拟响应。"""

from typing import Union


class ShipTool:
    """返回第一阶段演示的单个船舶摘要。"""

    async def run(self, payload: dict[str, str]) -> dict[str, Union[str, int]]:
        """从查询文本生成标准化的船舶数据。"""

        query = (payload.get("query") or "").replace("帮我查一下", "").strip(" ，。")
        ship_name = query or "长江1号"
        return {
            "ship_name": ship_name,
            "mmsi": "413000001",
            "ship_type": "散货船",
            "load_tons": 5200,
            "summary": f"{ship_name}当前可查询到基础档案，船型为散货船，参考载重5200吨。",
        }
