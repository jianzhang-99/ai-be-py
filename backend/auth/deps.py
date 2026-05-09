from __future__ import annotations

"""Dependency helpers for the authentication module."""

from functools import lru_cache

from backend.auth.service import AuthService


@lru_cache(maxsize=1)
def get_auth_service() -> AuthService:
    """Reuse one auth service instance so issued tokens stay valid in-process."""

    return AuthService()
