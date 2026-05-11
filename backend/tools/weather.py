"""天气查询工具，支持多城市天气预报。"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Union

logger = logging.getLogger(__name__)


class WeatherTool:
    """天气查询工具，支持多城市预报。"""

    async def run(self, payload: dict[str, str]) -> dict[str, Any]:
        """根据城市信息构建天气响应，支持多城市查询。

        Args:
            payload: 包含 city 或 cities 字段

        Returns:
            天气信息（单城市或多城市列表）
        """

        # 支持多城市查询
        cities_param = payload.get("cities")
        if cities_param:
            # 多城市查询
            if isinstance(cities_param, str):
                city_list = [c.strip() for c in cities_param.split(",")]
            elif isinstance(cities_param, list):
                city_list = cities_param
            else:
                city_list = ["武汉"]

            results = []
            for city in city_list:
                city = city.replace("帮我查一下", "").strip(" ，。")
                city = city or "武汉"
                results.append(self._get_weather_data(city))

            return {
                "cities": results,
                "count": len(results),
                "summary": f"已查询{len(results)}个城市天气："
                + "、".join([r["city"] for r in results]),
            }

        # 单城市查询（兼容旧接口）
        city = (payload.get("city") or "武汉").replace("帮我查一下", "").strip(" ，。")
        city = city or "武汉"
        weather_data = self._get_weather_data(city)

        return weather_data

    def _get_weather_data(self, city: str) -> dict[str, Union[str, int]]:
        """获取指定城市的天气数据。

        Args:
            city: 城市名称

        Returns:
            天气数据字典
        """

        # 当前演示使用固定数据，后续可接入真实天气 API
        return {
            "city": city,
            "weather": "多云",
            "temperature": 28,
            "humidity": 65,
            "wind": "东南风3级",
            "date": datetime.now().strftime("%Y-%m-%d"),
            "summary": f"{city}今天天气多云，最高温度28度，适合安排港航出行。",
        }
