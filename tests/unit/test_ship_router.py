"""Unit tests for Ship Router API endpoints."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


class FakeBigDataClient:
    """模拟 BigDataClient，不发真实 HTTP 请求。"""

    async def search_ship_page_data(self, params: dict) -> dict:
        return {
            "code": 0,
            "msg": "success",
            "data": {
                "total": 1,
                "records": [
                    {
                        "ship_name": "测试船舶",
                        "mmsi": "123456789",
                        "status": "正常",
                    }
                ],
            },
        }


class FakePilotClient:
    """模拟 PilotClient，不发真实 HTTP 请求。"""

    async def get_bigdata_ship_by_name(self, ship_name: str) -> dict:
        return {
            "code": 0,
            "msg": "success",
            "data": {
                "ship_name": ship_name,
                "mmsi": "987654321",
                "length": 120.5,
                "width": 20.0,
            },
        }

    async def search_ship_by_name(self, ship_name: str) -> dict:
        return {
            "code": 0,
            "msg": "success",
            "data": [
                {"ship_name": f"{ship_name}号", "mmsi": "111222333"},
            ],
        }


@pytest.fixture
def fake_bigdata():
    return FakeBigDataClient()


@pytest.fixture
def fake_pilot():
    return FakePilotClient()


@pytest.fixture
def client():
    """返回 TestClient，使用 fake 客户端注入。"""
    with patch("backend.api.routers.ship.get_bigdata_client", return_value=FakeBigDataClient()):
        with patch("backend.api.routers.ship.get_pilot_client", return_value=FakePilotClient()):
            from backend.main import app
            with TestClient(app) as client_instance:
                yield client_instance


class TestShipSearch:
    """POST /api/ship/search 测试。"""

    def test_search_ship_success(self, client: TestClient) -> None:
        """正常搜索船舶，返回列表数据。"""
        response = client.post(
            "/api/ship/search",
            json={"ship_name": "测试", "page": 1, "page_size": 20},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 0
        assert "data" in data

    def test_search_ship_page_invalid(self, client: TestClient) -> None:
        """page < 1 时返回参数错误。"""
        response = client.post(
            "/api/ship/search",
            json={"ship_name": "测试", "page": 0, "page_size": 20},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 1001

    def test_search_ship_page_size_invalid(self, client: TestClient) -> None:
        """page_size > 100 时返回参数错误。"""
        response = client.post(
            "/api/ship/search",
            json={"ship_name": "测试", "page": 1, "page_size": 200},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 1001


class TestShipDetail:
    """GET /api/ship/detail 测试。"""

    def test_get_ship_detail_success(self, client: TestClient) -> None:
        """正常获取船舶详情。"""
        response = client.get("/api/ship/detail?ship_name=测试船")
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 0
        assert data["data"]["ship_name"] == "测试船"

    def test_get_ship_detail_missing_param(self, client: TestClient) -> None:
        """ship_name 为空时返回参数错误。"""
        response = client.get("/api/ship/detail?ship_name=")
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 1001


class TestShipFuzzySearch:
    """GET /api/ship/fuzzy-search 测试。"""

    def test_fuzzy_search_success(self, client: TestClient) -> None:
        """正常模糊搜索。"""
        response = client.get("/api/ship/fuzzy-search?ship_name=测试")
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 0
        assert isinstance(data["data"], list)

    def test_fuzzy_search_too_short(self, client: TestClient) -> None:
        """ship_name 少于 2 字符时返回参数错误。"""
        response = client.get("/api/ship/fuzzy-search?ship_name=船")
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 1001