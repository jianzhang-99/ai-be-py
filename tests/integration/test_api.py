from __future__ import annotations

"""Integration tests for phase-one API contracts."""

from fastapi.testclient import TestClient

from backend.main import app


client = TestClient(app)


def login_and_get_token() -> str:
    """Create a real auth session for protected endpoint tests."""

    response = client.post(
        "/auth/login",
        json={"phone": "13800138000", "password": "123456"},
    )

    payload = response.json()
    assert response.status_code == 200
    assert payload["code"] == 0
    return payload["data"]["token"]


def test_health_endpoint_returns_ok() -> None:
    """The liveness endpoint should expose a stable success payload."""

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_endpoint_returns_intent_and_session() -> None:
    """The non-streaming chat API should match the documented response shape."""

    token = login_and_get_token()
    response = client.post(
        "/api/chat",
        json={"message": "帮我查一下武汉天气", "user_id": "u_001"},
        headers={"Authorization": f"Bearer {token}"},
    )

    payload = response.json()
    assert response.status_code == 200
    assert payload["intent"] == "QUERY_WEATHER"
    assert payload["session_id"]
    assert "武汉" in payload["message"]


def test_chat_stream_endpoint_returns_sse_events() -> None:
    """The streaming API should emit the documented phase-one events."""

    token = login_and_get_token()
    with client.stream(
        "POST",
        "/api/chat/stream",
        json={"message": "5月10日武汉装煤5000吨到南京", "user_id": "u_001"},
        headers={"Authorization": f"Bearer {token}"},
    ) as response:
        body = response.read().decode()

    assert response.status_code == 200
    assert "event: intent" in body
    assert "event: tool_start" in body
    assert "event: tool_result" in body
    assert "event: response" in body
    assert "event: done" in body


def test_login_and_current_user_flow() -> None:
    """Login should return a token that can be used on current-user lookup."""

    token = login_and_get_token()
    response = client.get(
        "/auth/current",
        headers={"Authorization": f"Bearer {token}"},
    )

    payload = response.json()
    assert response.status_code == 200
    assert payload["code"] == 0
    assert payload["data"]["userId"] == 1
    assert payload["data"]["phone"] == "13800138000"


def test_protected_route_returns_unauthorized_without_token() -> None:
    """Protected manager routes should reject requests without Bearer auth."""

    response = client.post("/api/chat", json={"message": "你好"})

    assert response.status_code == 401
    assert response.json() == {
        "code": 401,
        "msg": "未登录或登录已过期，请重新登录",
        "data": None,
    }
