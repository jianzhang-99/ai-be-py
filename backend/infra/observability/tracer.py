"""OpenTelemetry 链路追踪模块。

提供分布式追踪能力，支持请求追踪、Span 创建和传播。
"""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from typing import Any, Generator, Optional

from backend.config import get_settings

# OpenTelemetry 相关导入（可选，未安装时不启用）
try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.trace import Span, Status, StatusCode
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    trace = None
    Span = None
    Status = None
    StatusCode = None
    TraceContextTextMapPropagator = None


# 全局 tracer 实例
_tracer: Optional["trace.tracer"] = None
_propagator: Optional["TraceContextTextMapPropagator"] = None


def setup_tracer(service_name: str = "ai-be-py") -> None:
    """初始化 OpenTelemetry tracer。

    Args:
        service_name: 服务名称
    """
    global _tracer, _propagator

    if not OTEL_AVAILABLE:
        return

    # 创建 tracer provider
    provider = TracerProvider()

    # 添加 Console exporter（开发环境）
    settings = get_settings()
    if settings.debug:
        processor = BatchSpanProcessor(ConsoleSpanExporter())
        provider.add_span_processor(processor)

    # 设置全局 provider
    trace.set_tracer_provider(provider)

    # 获取 tracer
    _tracer = trace.get_tracer(service_name)
    _propagator = TraceContextTextMapPropagator()


def get_tracer() -> Optional["trace.tracer"]:
    """获取全局 tracer 实例。"""
    global _tracer
    if _tracer is None and OTEL_AVAILABLE:
        _tracer = trace.get_tracer("ai-be-py")
    return _tracer


class RequestTracer:
    """请求级追踪器，管理单个请求的 span 生命周期。"""

    def __init__(self, request_id: Optional[str] = None) -> None:
        self.request_id = request_id or self._generate_request_id()
        self._root_span: Optional["Span"] = None
        self._current_span: Optional["Span"] = None

    @staticmethod
    def _generate_request_id() -> str:
        """生成请求 ID。"""
        return f"req-{uuid.uuid4().hex[:16]}"

    def start_root_span(self, name: str, attributes: Optional[dict[str, Any]] = None) -> "RequestTracer":
        """启动根 span。

        Args:
            name: span 名称
            attributes: span 属性
        """
        tracer = get_tracer()
        if tracer is None:
            return self

        self._root_span = tracer.start_span(name, attributes=attributes)
        self._current_span = self._root_span
        return self

    def start_span(self, name: str, attributes: Optional[dict[str, Any]] = None) -> "RequestTracer":
        """启动子 span。

        Args:
            name: span 名称
            attributes: span 属性
        """
        tracer = get_tracer()
        if tracer is None or self._current_span is None:
            return self

        ctx = self._current_span.get_span_context()
        self._current_span = tracer.start_span(
            name,
            context=trace.set_span_in_context(self._current_span),
            attributes=attributes,
        )
        return self

    def end_span(self) -> None:
        """结束当前 span。"""
        if self._current_span is not None:
            self._current_span.end()
            self._current_span = self._root_span  # 回到父 span

    def end(self) -> None:
        """结束根 span。"""
        if self._root_span is not None:
            self._root_span.end()
            self._root_span = None
            self._current_span = None

    def set_attribute(self, key: str, value: Any) -> None:
        """设置 span 属性。"""
        if self._current_span is not None:
            self._current_span.set_attribute(key, value)

    def set_status(self, status_code: str, description: str = "") -> None:
        """设置 span 状态。

        Args:
            status_code: OK / ERROR
            description: 状态描述
        """
        if self._current_span is not None and OTEL_AVAILABLE:
            code = StatusCode.OK if status_code == "OK" else StatusCode.ERROR
            self._current_span.set_status(Status(code, description))

    def record_exception(self, exception: Exception) -> None:
        """记录异常到当前 span。"""
        if self._current_span is not None and OTEL_AVAILABLE:
            self._current_span.record_exception(exception)
            self._current_span.set_status(Status(StatusCode.ERROR, str(exception)))

    def add_event(self, name: str, attributes: Optional[dict[str, Any]] = None) -> None:
        """添加 span 事件。"""
        if self._current_span is not None:
            self._current_span.add_event(name, attributes=attributes)


@contextmanager
def trace_request(
    name: str,
    request_id: Optional[str] = None,
    attributes: Optional[dict[str, Any]] = None,
) -> Generator[RequestTracer, None, None]:
    """追踪请求的上下文管理器。

    Args:
        name: span 名称
        request_id: 请求 ID
        attributes: span 属性

    Usage:
        with trace_request("chat_stream", request_id="req-xxx") as tracer:
            tracer.set_attribute("session_id", "sess-xxx")
            # do something
            tracer.set_status("OK")
    """
    tracer_instance = RequestTracer(request_id)
    tracer_instance.start_root_span(name, attributes)
    try:
        yield tracer_instance
    except Exception as e:
        tracer_instance.record_exception(e)
        tracer_instance.set_status("ERROR", str(e))
        raise
    finally:
        tracer_instance.end()


@contextmanager
def trace_operation(
    name: str,
    operation_type: str,
    attributes: Optional[dict[str, Any]] = None,
) -> Generator[RequestTracer, None, None]:
    """追踪单个操作的上下文管理器。

    Args:
        name: 操作名称
        operation_type: 操作类型（如 "llm_call", "tool_call", "db_query"）
        attributes: span 属性

    Usage:
        with trace_operation("query_weather", "tool_call", {"tool": "weather"}) as tracer:
            # do something
            pass
    """
    tracer_instance = RequestTracer()
    attrs = {"operation_type": operation_type, **(attributes or {})}
    tracer_instance.start_root_span(name, attrs)
    try:
        yield tracer_instance
    except Exception as e:
        tracer_instance.record_exception(e)
        tracer_instance.set_status("ERROR", str(e))
        raise
    finally:
        tracer_instance.end()


def extract_trace_context(carrier: dict[str, str]) -> Optional[dict]:
    """从 HTTP header 等 carrier 中提取追踪上下文。

    Args:
        carrier: 包含 trace context 的字典（如 HTTP headers）

    Returns:
        trace context 字典
    """
    if not OTEL_AVAILABLE or _propagator is None:
        return None

    ctx = _propagator.extract(carrier)
    return ctx


def inject_trace_context(carrier: dict[str, str]) -> dict[str, str]:
    """将追踪上下文注入到 carrier（如 HTTP headers）。

    Args:
        carrier: 目标字典

    Returns:
        注入后的 carrier
    """
    if not OTEL_AVAILABLE or _propagator is None:
        return carrier

    _propagator.inject(carrier)
    return carrier


def get_current_trace_id() -> Optional[str]:
    """获取当前 span 的 trace ID。"""
    if not OTEL_AVAILABLE:
        return None

    current_span = trace.get_current_span()
    if current_span is None:
        return None

    span_context = current_span.get_span_context()
    if span_context.is_valid:
        return format(span_context.trace_id, "032x")
    return None


def get_current_span_id() -> Optional[str]:
    """获取当前 span 的 span ID。"""
    if not OTEL_AVAILABLE:
        return None

    current_span = trace.get_current_span()
    if current_span is None:
        return None

    span_context = current_span.get_span_context()
    if span_context.is_valid:
        return format(span_context.span_id, "016x")
    return None
