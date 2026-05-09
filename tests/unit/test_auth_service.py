from __future__ import annotations

"""Unit tests for the migrated authentication service."""

import pytest

from backend.auth.schemas import LoginRequest
from backend.auth.service import AuthError, AuthService


def test_auth_service_login_success_returns_token() -> None:
    """A valid phone/password pair should issue a token session."""

    service = AuthService()

    result = service.login(LoginRequest(phone="13800138000", password="123456"))

    assert result.userId == 1
    assert result.phone == "13800138000"
    assert result.token


def test_auth_service_login_rejects_invalid_phone() -> None:
    """Phone format validation should match the Java service behavior."""

    service = AuthService()

    with pytest.raises(AuthError) as error:
        service.login(LoginRequest(phone="admin", password="123456"))

    assert error.value.code == 1001


def test_auth_service_revokes_token_on_logout() -> None:
    """A logged-out token should no longer resolve to a current user."""

    service = AuthService()
    login_result = service.login(LoginRequest(phone="13800138000", password="123456"))

    service.logout(login_result.token)

    with pytest.raises(AuthError) as error:
        service.get_current_user(login_result.token)

    assert error.value.code == 401
