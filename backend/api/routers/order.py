"""运单 API 路由。

提供运单信息抽取预览、运单预录入提交等能力，接入运吨吨 Pilot API。
"""

from __future__ import annotations

from typing import Annotated, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from backend.auth.schemas import ResultResponse
from backend.infra.external.pilot_client import PilotApiError, PilotClient, get_pilot_client
from backend.tools.order import OrderTool

router = APIRouter(prefix="/api/order", tags=["order"])


def get_pilot() -> PilotClient:
    """返回 PilotClient 实例"""
    return get_pilot_client()


class OrderPreviewRequest(BaseModel):
    """运单预览请求参数"""

    message: str = Field(..., description="自然语言运单描述，如：5月10日武汉装煤5000吨到南京")


class OrderSubmitRequest(BaseModel):
    """运单提交请求参数"""

    loading_port: str = Field(..., min_length=1, description="装货港")
    unloading_port: str = Field(..., min_length=1, description="卸货港")
    loading_date: str = Field(..., description="装货日期 YYYY-MM-DD")
    cargo_name: str = Field(..., min_length=1, description="货物名称")
    cargo_weight: float = Field(..., gt=0, description="货物重量（吨）")
    ship_name: Optional[str] = Field(default=None, description="船舶名称（可选）")
    remark: Optional[str] = Field(default=None, description="备注")


def get_order_tool() -> OrderTool:
    """返回 OrderTool 实例"""
    return OrderTool()


@router.post("/preview", response_model=ResultResponse)
async def preview_order(
    body: OrderPreviewRequest,
    order_tool: Annotated[OrderTool, Depends(get_order_tool)],
) -> ResultResponse:
    """从自然语言抽取运单信息并预览。

    POST /api/order/preview
    """

    message = body.message

    if not message:
        return ResultResponse.failure(1001, "message 参数不能为空")

    try:
        result = await order_tool.run({"message": message})
        return ResultResponse.success(data=result)
    except Exception as e:
        return ResultResponse.failure(2001, f"运单信息抽取失败: {str(e)}")


@router.post("/submit", response_model=ResultResponse)
async def submit_order(
    body: OrderSubmitRequest,
    pilot: Annotated[PilotClient, Depends(get_pilot)],
) -> ResultResponse:
    """将运单数据提交到 Pilot API 进行预录入。

    POST /api/order/submit
    """

    # 构建运单数据
    order_data = {
        "loadingPort": body.loading_port,
        "unloadingPort": body.unloading_port,
        "loadingDate": body.loading_date,
        "cargoName": body.cargo_name,
        "cargoWeight": body.cargo_weight,
        "shipName": body.ship_name,
        "remark": body.remark,
    }

    try:
        result = await pilot.create_order(order_data)
        return ResultResponse.success(data=result.get("data"))
    except PilotApiError as e:
        return ResultResponse.failure(5001, f"运单提交失败: {e.message}")


@router.get("/query", response_model=ResultResponse)
async def query_order(
    order_id: str,
    pilot: Annotated[PilotClient, Depends(get_pilot)],
) -> ResultResponse:
    """查询运单状态。

    GET /api/order/query?order_id=xxx
    """

    if not order_id:
        return ResultResponse.failure(1001, "order_id 参数不能为空")

    try:
        result = await pilot.get_order(order_id)
        return ResultResponse.success(data=result.get("data"))
    except PilotApiError as e:
        return ResultResponse.failure(5001, f"运单查询失败: {e.message}")