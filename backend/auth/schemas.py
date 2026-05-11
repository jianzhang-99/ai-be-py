from __future__ import annotations

"""Schemas used by the authentication module."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class ResultResponse(BaseModel):
    """Java-style API result wrapper used by manager endpoints."""

    code: int
    data: Optional[object] = None
    msg: str

    @classmethod
    def success(cls, data: Optional[object] = None, msg: str = "成功") -> "ResultResponse":
        """Build a successful result payload."""

        return cls(code=0, data=data, msg=msg)

    @classmethod
    def failure(cls, code: int, msg: str, data: Optional[object] = None) -> "ResultResponse":
        """Build a failed result payload."""

        return cls(code=code, data=data, msg=msg)


class LoginRequest(BaseModel):
    """Login request with the same field names used by the Java version."""

    phone: str = Field(..., description="手机号")
    password: str = Field(..., min_length=1, description="登录密码")


class LoginResponse(BaseModel):
    """Login response returned after token issuance."""

    userId: int
    phone: str
    token: str


class CurrentUserResponse(BaseModel):
    """Current authenticated user information."""

    userId: int
    phone: str
    token: str
    nickName: Optional[str] = None
    lastLoginTime: Optional[datetime] = None


class AuthUser(BaseModel):
    """Internal authenticated user model (maps to ld_test.sys_user)."""

    id: int
    phone: str
    password_hash: str
    nick_name: Optional[str] = None
    status: int = 1  # 0=正常, 1=禁用
    is_delete: bool = False
    last_login_time: Optional[datetime] = None


class SessionInfo(BaseModel):
    """Server-side token session info."""

    token: str
    user_id: int
    phone: str
    created_at: datetime