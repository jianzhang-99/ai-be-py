from __future__ import annotations

"""Authentication service migrated from the Java login/auth flow."""

import re
import secrets
from datetime import datetime

import bcrypt

from backend.auth.schemas import AuthUser, CurrentUserResponse, LoginRequest, LoginResponse, SessionInfo

PHONE_PATTERN = re.compile(r"^1[3-9]\d{9}$")
STATUS_NORMAL = 1
STATUS_DISABLE = 2


class AuthError(Exception):
    """Expected auth-domain error with a Java-style business code."""

    def __init__(self, code: int, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(message)


class AuthSessionStore:
    """In-memory token store that mirrors the Java session-based flow."""

    def __init__(self) -> None:
        self._sessions_by_token: dict[str, SessionInfo] = {}
        self._tokens_by_user_id: dict[int, str] = {}

    def create_session(self, user_id: int, phone: str) -> SessionInfo:
        """Issue a fresh token and replace any previous active session."""

        old_token = self._tokens_by_user_id.get(user_id)
        if old_token is not None:
            self._sessions_by_token.pop(old_token, None)

        token = secrets.token_urlsafe(32)
        session = SessionInfo(
            token=token,
            user_id=user_id,
            phone=phone,
            created_at=datetime.now(),
        )
        self._sessions_by_token[token] = session
        self._tokens_by_user_id[user_id] = token
        return session

    def get_session(self, token: str) -> SessionInfo | None:
        """Resolve a token to its session info."""

        return self._sessions_by_token.get(token)

    def revoke_token(self, token: str) -> None:
        """Invalidate the given token if it exists."""

        session = self._sessions_by_token.pop(token, None)
        if session is not None:
            self._tokens_by_user_id.pop(session.user_id, None)


class AuthService:
    """Service layer for login, logout, and current-user lookup."""

    def __init__(self) -> None:
        self._users_by_phone = self._load_default_users()
        self._users_by_id = {user.id: user for user in self._users_by_phone.values()}
        self._sessions = AuthSessionStore()

    def _hash_password(self, password: str) -> str:
        """Hash a plaintext password with bcrypt for stored demo users."""

        return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

    def _load_default_users(self) -> dict[str, AuthUser]:
        """Bootstrap manager users until a real database-backed user module lands."""

        users = [
            AuthUser(
                id=1,
                phone="13800138000",
                password_hash=self._hash_password("123456"),
                nick_name="测试用户",
            ),
            AuthUser(
                id=2,
                phone="13900000000",
                password_hash=self._hash_password("admin123"),
                nick_name="管理员",
            ),
        ]
        return {user.phone: user for user in users}

    def login(self, request: LoginRequest) -> LoginResponse:
        """Validate phone/password and issue a token."""

        if not PHONE_PATTERN.match(request.phone):
            raise AuthError(1001, "手机号码不正确")

        user = self._users_by_phone.get(request.phone)
        if user is None or user.is_delete or user.status == STATUS_DISABLE:
            raise AuthError(1002, "用户名或密码不正确")

        password_ok = bcrypt.checkpw(
            request.password.encode("utf-8"),
            user.password_hash.encode("utf-8"),
        )
        if not password_ok:
            raise AuthError(1002, "用户名或密码不正确")

        user.last_login_time = datetime.now()
        session = self._sessions.create_session(user_id=user.id, phone=user.phone)
        return LoginResponse(userId=user.id, phone=user.phone, token=session.token)

    def logout(self, token: str | None) -> None:
        """Revoke the current token if the caller is logged in."""

        if token:
            self._sessions.revoke_token(token)

    def get_current_user(self, token: str) -> CurrentUserResponse:
        """Load the current user from the presented token."""

        session = self._sessions.get_session(token)
        if session is None:
            raise AuthError(401, "未登录或登录已过期，请重新登录")

        user = self._users_by_id.get(session.user_id)
        if user is None or user.is_delete or user.status == STATUS_DISABLE:
            self._sessions.revoke_token(token)
            raise AuthError(401, "未登录或登录已过期，请重新登录")

        return CurrentUserResponse(
            userId=user.id,
            phone=user.phone,
            token=token,
            nickName=user.nick_name,
            lastLoginTime=user.last_login_time,
        )

    def authenticate_token(self, token: str) -> CurrentUserResponse:
        """Alias used by middleware for token validation."""

        return self.get_current_user(token)
