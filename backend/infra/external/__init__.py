"""外部系统适配器模块。

包含运吨吨 Pilot API 和大数据 API 的客户端封装。
"""

from __future__ import annotations

from backend.infra.external.bigdata_client import (
    BigDataApiError,
    BigDataClient,
    get_bigdata_client,
)
from backend.infra.external.pilot_client import (
    PilotApiError,
    PilotClient,
    get_pilot_client,
)

__all__ = [
    "PilotClient",
    "get_pilot_client",
    "PilotApiError",
    "BigDataClient",
    "get_bigdata_client",
    "BigDataApiError",
]
