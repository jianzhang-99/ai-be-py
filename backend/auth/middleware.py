from __future__ import annotations

"""Global authentication middleware for protected manager endpoints."""

from collections.abc import Awaitable, Callable

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, Response

from backend.auth.constants import AUTHORIZATION, BEARER_PREFIX, PHONE, USER_ID
from backend.auth.deps import get_auth_service, get_sys_user_repository
from backend.auth.service import AuthError

PUBLIC_PATHS = {
    "/",
    "/health",
    "/openapi.json",
    "/docs",
    "/docs/oauth2-redirect",
    "/redoc",
    "/auth/login",
    "/auth/logout",
}

PUBLIC_PREFIXES = (
    "/swagger-ui",
    "/v2",
    "/v3",
    "/webjars",
    "/swagger-resources",
)


class AuthMiddleware(BaseHTTPMiddleware):
    """Validate Bearer tokens for protected routes and attach user context."""

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Allow public paths through and enforce Bearer auth elsewhere."""

        if self._is_public(request.url.path):
            return await call_next(request)

        token = self._extract_bearer_token(request)
        if token is None:
            return self._unauthorized_response()

        auth_service = get_auth_service()
        user_repo = get_sys_user_repository()

        try:
            current_user = await auth_service.authenticate_token(token, user_repo)
        except AuthError:
            return self._unauthorized_response()

        request.state.current_user = current_user
        setattr(request.state, USER_ID, current_user.userId)
        setattr(request.state, PHONE, current_user.phone)
        return await call_next(request)

    def _is_public(self, path: str) -> bool:
        """Check whether the request path should skip auth."""

        if path in PUBLIC_PATHS:
            return True
        return any(path.startswith(prefix) for prefix in PUBLIC_PREFIXES)

    def _extract_bearer_token(self, request: Request) -> str | None:
        """Read the Authorization header and normalize the Bearer token."""

        authorization = request.headers.get(AUTHORIZATION)
        if not authorization or not authorization.startswith(BEARER_PREFIX):
            return None
        token = authorization[len(BEARER_PREFIX):].strip()
        return token or None

    def _unauthorized_response(self) -> JSONResponse:
        """Return the same shape the Java interceptor writes."""

        return JSONResponse(
            status_code=401,
            content={"code": 401, "msg": "未登录或登录已过期，请重新登录", "data": None},
        )