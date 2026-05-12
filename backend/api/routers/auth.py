from __future__ import annotations

"""Authentication router migrated from the Java manager module."""

from typing import Annotated

from fastapi import APIRouter, Depends, Request, Response

from backend.auth.constants import TOKEN_COOKIE_NAME
from backend.auth.deps import get_auth_service, get_sys_user_repository
from backend.auth.schemas import LoginRequest, ResultResponse, CurrentUserResponse
from backend.auth.service import AuthError, AuthService
from backend.infra.database.repositories.sys_user_repository import SysUserRepository

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login", response_model=ResultResponse)
async def login(
    request: LoginRequest,
    response: Response,
    auth_service: Annotated[AuthService, Depends(get_auth_service)],
    user_repo: Annotated[SysUserRepository, Depends(get_sys_user_repository)],
) -> ResultResponse:
    """Authenticate the manager user and return a session token."""

    try:
        result = await auth_service.login(request, user_repo)
    except AuthError as error:
        return ResultResponse.failure(code=error.code, msg=error.message)
    response.set_cookie(
        key=TOKEN_COOKIE_NAME,
        value=result.token,
        httponly=True,
        samesite="lax",
        path="/",
    )
    return ResultResponse.success(data=result)


@router.post("/logout", response_model=ResultResponse)
async def logout(
    request: Request,
    response: Response,
    auth_service: Annotated[AuthService, Depends(get_auth_service)],
) -> ResultResponse:
    """Log the current user out without requiring a strict pre-check."""

    authorization = request.headers.get("Authorization", "")
    token = authorization.removeprefix("Bearer ").strip() or request.cookies.get(TOKEN_COOKIE_NAME)
    await auth_service.logout(token)
    response.delete_cookie(TOKEN_COOKIE_NAME, path="/")
    return ResultResponse.success(data="退出登录成功")


@router.get("/current", response_model=ResultResponse)
async def get_current_user(
    request: Request,
    auth_service: Annotated[AuthService, Depends(get_auth_service)],
    user_repo: Annotated[SysUserRepository, Depends(get_sys_user_repository)],
) -> ResultResponse:
    """Return the authenticated user context extracted by middleware."""

    current_user: CurrentUserResponse = request.state.current_user
    # 重新从 DB 拉一次最新信息（昵称、状态等可能已变）
    fresh = await auth_service.get_current_user(current_user.token, user_repo)
    return ResultResponse.success(data=fresh)
