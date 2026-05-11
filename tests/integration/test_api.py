from __future__ import annotations

"""Integration tests – auth endpoints with in-memory stub, no real DB."""

import pytest
import bcrypt
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from backend.auth.schemas import AuthUser


# ---- In-memory fake repository -----------------------------------------------

class FakeSysUserRepository:
    """Patches the real SysUserRepository; backed by a dict in memory."""

    def __init__(self, users: dict[str, AuthUser]) -> None:
        self._users = users
        self._id_index = {u.id: u for u in users.values()}
        self._update_calls: list[int] = []

    async def find_by_phone(self, phone: str) -> AuthUser | None:
        return self._users.get(phone)

    async def find_by_id(self, user_id: int) -> AuthUser | None:
        return self._id_index.get(user_id)

    async def update_last_login(self, user_id: int, login_ip: str = "") -> None:
        self._update_calls.append(user_id)


# ---- Pre-built test user (password = "123456") ------------------------------
_bcrypt_hash = bcrypt.hashpw("123456".encode(), bcrypt.gensalt()).decode()

ALICE = AuthUser(
    id=1,
    phone="15888888888",
    password_hash=_bcrypt_hash,
    nick_name="测试用户",
    status=1,  # 1 = 正常（匹配 ld_test.sys_user.status 语义）
    is_delete=False,
    last_login_time=None,
)

_fake_repo = FakeSysUserRepository({ALICE.phone: ALICE})


@pytest.fixture(autouse=True)
def patch_auth_deps(monkeypatch):
    """Replace get_sys_user_repository with in-memory stub for every test."""
    from backend.auth import deps
    monkeypatch.setattr(deps, "get_sys_user_repository", lambda: _fake_repo)


# ---- Tests -------------------------------------------------------------------

def test_health_endpoint_returns_ok() -> None:
    """The liveness endpoint is public and never touches auth."""
    from backend.main import app
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_protected_route_returns_401_without_token() -> None:
    """Unauthenticated requests to protected routes get 401."""
    from backend.main import app
    client = TestClient(app)
    response = client.post("/api/chat", json={"message": "你好"})
    assert response.status_code == 401
    assert response.json() == {
        "code": 401,
        "msg": "未登录或登录已过期，请重新登录",
        "data": None,
    }


def test_login_and_current_user_flow() -> None:
    """Login returns a token usable for /auth/current."""
    from backend.main import app
    client = TestClient(app)

    login_resp = client.post(
        "/auth/login",
        json={"phone": "15888888888", "password": "123456"},
    )
    assert login_resp.status_code == 200
    login_payload = login_resp.json()
    assert login_payload["code"] == 0, f"login failed: {login_payload}"
    token = login_payload["data"]["token"]

    current_resp = client.get(
        "/auth/current",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert current_resp.status_code == 200
    current_payload = current_resp.json()
    assert current_payload["code"] == 0
    assert current_payload["data"]["userId"] == 1
    assert current_payload["data"]["phone"] == "15888888888"


def test_login_bad_phone_format_returns_1001() -> None:
    """Malformed phone is rejected before hitting the DB."""
    from backend.main import app
    client = TestClient(app)
    resp = client.post("/auth/login", json={"phone": "bad", "password": "123456"})
    assert resp.json()["code"] == 1001


def test_login_wrong_password_returns_1002() -> None:
    """Wrong password is rejected with 1002."""
    from backend.main import app
    client = TestClient(app)
    resp = client.post("/auth/login", json={"phone": "15888888888", "password": "wrong"})
    assert resp.json()["code"] == 1002


def test_login_unknown_user_returns_1002() -> None:
    """Unknown phone is rejected with 1002 (no info leak)."""
    from backend.main import app
    client = TestClient(app)
    resp = client.post("/auth/login", json={"phone": "19900000000", "password": "123456"})
    assert resp.json()["code"] == 1002


def test_logout_succeeds_and_invalidates_token() -> None:
    """Logout returns success and the token is no longer valid."""
    from backend.main import app
    client = TestClient(app)

    login_resp = client.post(
        "/auth/login",
        json={"phone": "15888888888", "password": "123456"},
    )
    token = login_resp.json()["data"]["token"]

    logout_resp = client.post(
        "/auth/logout",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert logout_resp.status_code == 200
    assert logout_resp.json()["code"] == 0

    current_resp = client.get(
        "/auth/current",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert current_resp.status_code == 401