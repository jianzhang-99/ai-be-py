"""Unit tests for Order Router API endpoints."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


class FakePilotClient:
    """模拟 PilotClient，不发真实 HTTP 请求。"""

    async def create_order(self, order_data: dict) -> dict:
        return {
            "code": 0,
            "msg": "success",
            "data": {
                "id": "ORD-20260512-001",
                "loadingPort": order_data.get("loadingPort"),
                "unloadingPort": order_data.get("unloadingPort"),
            },
        }

    async def get_order(self, order_id: str) -> dict:
        return {
            "code": 0,
            "msg": "success",
            "data": {
                "id": order_id,
                "status": "预录入成功",
            },
        }


class FakeOrderTool:
    """模拟 OrderTool，不调用真实 LLM。"""

    async def run(self, payload: dict) -> dict:
        text = payload.get("message", "")
        if "武汉" in text and "南京" in text:
            return {
                "loading_port": "武汉",
                "discharge_port": "南京",
                "cargo_name": "煤",
                "tons": "5000",
                "shipping_date": "5月10日",
                "_detail": {
                    "loading_port": "武汉",
                    "discharge_port": "南京",
                    "cargo_name": "煤",
                    "tons": "5000",
                    "shipping_date": "5月10日",
                },
            }
        return {
            "loading_port": "未知",
            "discharge_port": "未知",
            "cargo_name": "未知",
            "tons": "0",
            "shipping_date": "待确认",
            "_detail": {},
        }


@pytest.fixture
def client():
    """返回 TestClient，使用 fake 客户端注入。"""
    with patch("backend.api.routers.order.get_pilot_client", return_value=FakePilotClient()):
        with patch("backend.tools.order.OrderTool", return_value=FakeOrderTool()):
            from backend.main import app
            with TestClient(app) as client_instance:
                yield client_instance


class TestOrderPreview:
    """POST /api/order/preview 测试。"""

    def test_preview_order_success(self, client: TestClient) -> None:
        """正常从自然语言抽取运单信息。"""
        response = client.post(
            "/api/order/preview",
            json={"message": "5月10日武汉装煤5000吨到南京"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 0
        assert "loading_port" in data["data"]

    def test_preview_order_empty_message(self, client: TestClient) -> None:
        """message 为空时返回参数错误。"""
        response = client.post(
            "/api/order/preview",
            json={"message": ""},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 1001


class TestOrderSubmit:
    """POST /api/order/submit 测试。"""

    def test_submit_order_success(self, client: TestClient) -> None:
        """正常提交运单数据到 Pilot API。"""
        response = client.post(
            "/api/order/submit",
            json={
                "loading_port": "武汉",
                "unloading_port": "南京",
                "loading_date": "2026-05-10",
                "cargo_name": "煤",
                "cargo_weight": 5000,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 0
        assert "id" in data["data"]

    def test_submit_order_missing_loading_port(self, client: TestClient) -> None:
        """loading_port 缺失时 Pydantic 返回 422。"""
        response = client.post(
            "/api/order/submit",
            json={
                # 故意不带 loading_port 字段
                "unloading_port": "南京",
                "loading_date": "2026-05-10",
                "cargo_name": "煤",
                "cargo_weight": 5000,
            },
        )
        assert response.status_code == 422  # Pydantic 验证失败

    def test_submit_order_invalid_weight(self, client: TestClient) -> None:
        """cargo_weight <= 0 时 Pydantic 返回 422。"""
        response = client.post(
            "/api/order/submit",
            json={
                "loading_port": "武汉",
                "unloading_port": "南京",
                "loading_date": "2026-05-10",
                "cargo_name": "煤",
                "cargo_weight": -100,
            },
        )
        assert response.status_code == 422  # Pydantic 验证失败


class TestOrderQuery:
    """GET /api/order/query 测试。"""

    def test_query_order_success(self, client: TestClient) -> None:
        """正常查询运单状态。"""
        response = client.get("/api/order/query", params={"order_id": "ORD-20260512-001"})
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 0
        assert data["data"]["status"] == "预录入成功"

    def test_query_order_missing_id(self, client: TestClient) -> None:
        """不传 order_id 参数时 FastAPI 返回 422。"""
        response = client.get("/api/order/query")
        assert response.status_code == 422  # 缺少必需参数