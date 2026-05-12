"""可观测性模块。

包含结构化日志、Prometheus 指标、链路追踪等组件。
"""

from backend.infra.observability.logger import (
    RequestContext,
    get_logger,
    log_error,
    log_llm_call,
    log_request_completed,
    log_request_received,
    log_slow_operation,
    log_tool_called,
    setup_logging,
)
from backend.infra.observability.metrics import (
    get_metrics,
    get_metrics_content_type,
    inc_active_sessions,
    dec_active_sessions,
    record_cache_hit,
    record_cache_miss,
    set_active_sessions,
    track_chat,
    track_external_api,
    track_llm,
    track_tool,
)
from backend.infra.observability.tracer import (
    extract_trace_context,
    get_current_span_id,
    get_current_trace_id,
    get_tracer,
    inject_trace_context,
    setup_tracer,
    trace_operation,
    trace_request,
)

__all__ = [
    # logger
    "setup_logging",
    "get_logger",
    "RequestContext",
    "log_request_received",
    "log_request_completed",
    "log_llm_call",
    "log_tool_called",
    "log_slow_operation",
    "log_error",
    # metrics
    "get_metrics",
    "get_metrics_content_type",
    "track_llm",
    "track_tool",
    "track_chat",
    "track_external_api",
    "set_active_sessions",
    "inc_active_sessions",
    "dec_active_sessions",
    "record_cache_hit",
    "record_cache_miss",
    # tracer
    "setup_tracer",
    "get_tracer",
    "trace_request",
    "trace_operation",
    "extract_trace_context",
    "inject_trace_context",
    "get_current_trace_id",
    "get_current_span_id",
]
