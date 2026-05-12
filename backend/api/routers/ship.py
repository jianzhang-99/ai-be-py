"""船舶查询 API 路由。

提供船舶搜索、船舶详情等能力，接入大数据 API。
"""

from __future__ import annotations

from typing import Annotated, Optional

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from backend.auth.schemas import ResultResponse
from backend.infra.external.bigdata_client import BigDataApiError, BigDataClient, get_bigdata_client
from backend.infra.external.pilot_client import PilotApiError, PilotClient, get_pilot_client

router = APIRouter(prefix="/api/ship", tags=["ship"])


def get_pilot() -> PilotClient:
    """返回 PilotClient 实例"""
    return get_pilot_client()


def get_bigdata() -> BigDataClient:
    """返回 BigDataClient 实例"""
    return get_bigdata_client()


class ShipSearchRequest(BaseModel):
    """船舶搜索请求参数"""

    ship_name: Optional[str] = Field(default=None, description="船舶名称（支持模糊匹配）")
    page: int = Field(default=1, ge=1, description="页码")
    page_size: int = Field(default=20, ge=1, le=100, description="每页条数")


@router.post("/search", response_model=ResultResponse)
async def search_ship(
    request: Request,
    bigdata: Annotated[BigDataClient, Depends(get_bigdata)],
) -> ResultResponse:
    """按条件搜索船舶（航线+船型+载重筛选）。

    POST /api/ship/search
    """

    body = await request.json()
    ship_name = body.get("ship_name")
    page = body.get("page", 1)
    page_size = body.get("page_size", 20)

    if page < 1:
        return ResultResponse.failure(1001, "page 参数无效")
    if page_size < 1 or page_size > 100:
        return ResultResponse.failure(1001, "page_size 参数需在 1-100 之间")

    try:
        result = await bigdata.search_ship_page_data({
            "shipName": ship_name or "",
            "pageNum": page,
            "pageSize": page_size,
        })
        return ResultResponse.success(data=result.get("data"))
    except BigDataApiError as e:
        return ResultResponse.failure(5002, f"船舶查询失败: {e.message}")


@router.get("/detail", response_model=ResultResponse)
async def get_ship_detail(
    request: Request,
    ship_name: str,
    pilot: Annotated[PilotClient, Depends(get_pilot)],
) -> ResultResponse:
    """根据船名获取船舶详细信息。

    GET /api/ship/detail?ship_name=xxx
    """

    if not ship_name:
        return ResultResponse.failure(1001, "ship_name 参数不能为空")

    try:
        result = await pilot.get_bigdata_ship_by_name(ship_name)
        return ResultResponse.success(data=result.get("data"))
    except PilotApiError as e:
        return ResultResponse.failure(5001, f"船舶详情查询失败: {e.message}")


@router.get("/fuzzy-search", response_model=ResultResponse)
async def fuzzy_search_ship(
    request: Request,
    ship_name: str,
    pilot: Annotated[PilotClient, Depends(get_pilot)],
) -> ResultResponse:
    """船舶名称模糊搜索（用于输入提示）。

    GET /api/ship/fuzzy-search?ship_name=xxx
    """

    if not ship_name or len(ship_name) < 2:
        return ResultResponse.failure(1001, "ship_name 至少需要2个字符")

    try:
        result = await pilot.search_ship_by_name(ship_name)
        return ResultResponse.success(data=result.get("data"))
    except PilotApiError as e:
        return ResultResponse.failure(5001, f"船舶搜索失败: {e.message}")