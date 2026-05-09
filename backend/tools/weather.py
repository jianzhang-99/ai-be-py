from __future__ import annotations

"""第一阶段用于演示和测试的天气工具，带有确定性模拟数据。"""

from datetime import datetime
from typing import Union


class WeatherTool:
    """返回用于演示和测试流程的稳定天气数据。"""

    async def run(self, payload: dict[str, str]) -> dict[str, Union[str, int]]:
        """根据提取的城市信息构建天气响应。"""

        city = (payload.get("city") or "武汉").replace("帮我查一下", "").strip(" ，。")
        city = city or "武汉"
        return {
            "city": city,
            "weather": "多云",
            "temperature": 28,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "summary": f"{city}今天天气多云，最高温度28度，适合安排港航出行。",
        }
