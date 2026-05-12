"""结构化日志配置。

使用 structlog 输出 JSON 格式日志，包含 request_id、session_id 等必记字段。
"""

from __future__ import annotations

import logging
import sys
import time
from typing import Any

import structlog

from backend.config import get_settings


def setup_logging() -> None:
    """初始化结构化日志配置。

    必须在应用启动时调用一次。
    """

    settings = get_settings()

    # 配置标准库 logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=logging.DEBUG if settings.debug else logging.INFO,
    )

    # 配置 structlog
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.DEBUG if settings.debug else logging.INFO
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.BoundLogger:
    """获取结构化日志实例。

    Args:
        name: 可选的日志名称，通常传入 __name__

    Returns:
        配置好的 structlog logger
    """

    logger = structlog.get_logger(name)
    return logger


class RequestContext:
    """请求上下文，用于在一次请求范围内传递日志字段。"""

    _context_vars = structlog.contextvars

    @classmethod
    def bind(
        cls,
        request_id: str | None = None,
        session_id: str | None = None,
        user_id: int | None = None,
        **kwargs: Any,
    ) -> None:
        """绑定请求级上下文变量。

        Args:
            request_id: 单次请求唯一标识
            session_id: 会话标识
            user_id: 用户 ID
            **kwargs: 其他自定义字段
        """

        bindings = {
            "request_id": request_id or _generate_request_id(),
            "session_id": session_id,
            "user_id": user_id,
            **kwargs,
        }
        # 过滤 None 值
        bindings = {k: v for k, v in bindings.items() if v is not None}
        cls._context_vars.bind_contextvars(**bindings)

    @classmethod
    def unbind(cls, *keys: str) -> None:
        """解绑指定的上下文变量。"""

        cls._context_vars.unbind_contextvars(*keys)

    @classmethod
    def clear(cls) -> None:
        """清除所有上下文变量。"""

        cls._context_vars.clear_contextvars()


def _generate_request_id() -> str:
    """生成唯一的请求 ID。"""

    import uuid
    return f"req-{uuid.uuid4().hex[:16]}"


# 日志工具函数，封装常见日志场景


def log_request_received(
    message: str,
    request_id: str | None = None,
    session_id: str | None = None,
    **kwargs: Any,
) -> None:
    """记录收到请求。"""

    logger = get_logger("api")
    logger.info(
        "request_received",
        message=message,
        request_id=request_id or _generate_request_id(),
        session_id=session_id,
        **kwargs,
    )


def log_request_completed(
    latency_ms: int,
    status_code: int,
    request_id: str | None = None,
    session_id: str | None = None,
    **kwargs: Any,
) -> None:
    """记录请求完成。"""

    logger = get_logger("api")
    logger.info(
        "request_completed",
        request_id=request_id,
        session_id=session_id,
        latency_ms=latency_ms,
        status_code=status_code,
        **kwargs,
    )


def log_llm_call(
    model: str,
    latency_ms: int,
    request_id: str | None = None,
    session_id: str | None = None,
    **kwargs: Any,
) -> None:
    """记录 LLM 调用。"""

    logger = get_logger("llm")
    logger.info(
        "llm_call",
        request_id=request_id,
        session_id=session_id,
        model=model,
        latency_ms=latency_ms,
        **kwargs,
    )


def log_tool_called(
    tool_name: str,
    latency_ms: int,
    success: bool = True,
    request_id: str | None = None,
    session_id: str | None = None,
    error: str | None = None,
    **kwargs: Any,
) -> None:
    """记录工具调用。"""

    logger = get_logger("tool")
    event_type = "tool_called" if success else "tool_failed"
    log_fn = logger.info if success else logger.warning
    log_fn(
        event_type,
        request_id=request_id,
        session_id=session_id,
        tool_name=tool_name,
        latency_ms=latency_ms,
        error=error,
        **kwargs,
    )


def log_slow_operation(
    operation: str,
    latency_ms: int,
    threshold_ms: int,
    request_id: str | None = None,
    session_id: str | None = None,
    **kwargs: Any,
) -> None:
    """记录慢操作告警。"""

    logger = get_logger("slow")
    logger.warning(
        "slow_operation",
        request_id=request_id,
        session_id=session_id,
        operation=operation,
        latency_ms=latency_ms,
        threshold_ms=threshold_ms,
        **kwargs,
    )


def log_error(
    error: Exception,
    operation: str,
    request_id: str | None = None,
    session_id: str | None = None,
    **kwargs: Any,
) -> None:
    """记录错误日志。"""

    logger = get_logger("error")
    logger.error(
        "error",
        request_id=request_id,
        session_id=session_id,
        operation=operation,
        error=str(error),
        error_type=type(error).__name__,
        **kwargs,
    )
