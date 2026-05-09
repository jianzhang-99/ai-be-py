from __future__ import annotations

"""Authentication router migrated from the Java manager module."""

from fastapi import APIRouter, Depends, Request

from backend.auth.deps import get_auth_service
from backend.auth.schemas import LoginRequest, ResultResponse
from backend.auth.service import AuthError, AuthService

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login", response_model=ResultResponse)
async def login(
    request: LoginRequest,
    auth_service: AuthService = Depends(get_auth_service),
) -> ResultResponse:
    """Authenticate the manager user and return a session token."""

    try:
        result = auth_service.login(request)
    except AuthError as error:
        return ResultResponse.failure(code=error.code, msg=error.message)
    return ResultResponse.success(data=result)


@router.post("/logout", response_model=ResultResponse)
async def logout(
    request: Request,
    auth_service: AuthService = Depends(get_auth_service),
) -> ResultResponse:
    """Log the current user out without requiring a strict pre-check."""

    authorization = request.headers.get("Authorization", "")
    token = authorization.removeprefix("Bearer ").strip() or None
    auth_service.logout(token)
    return ResultResponse.success(data="退出登录成功")


@router.get("/current", response_model=ResultResponse)
async def get_current_user(request: Request) -> ResultResponse:
    """Return the authenticated user context extracted by middleware."""

    current_user = request.state.current_user
    return ResultResponse.success(data=current_user)
