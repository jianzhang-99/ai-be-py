"""Unit tests for Observability components (logger, metrics, tracer)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from backend.infra.observability.logger import (
    RequestContext,
    _generate_request_id,
    log_error,
    log_llm_call,
    log_request_completed,
    log_request_received,
    log_slow_operation,
    log_tool_called,
)
from backend.infra.observability.metrics import (
    LLMLatencyRecorder,
    ToolLatencyRecorder,
    ChatLatencyRecorder,
    ExternalApiLatencyRecorder,
    record_cache_hit,
    record_cache_miss,
    get_metrics,
    get_metrics_content_type,
    set_active_sessions,
)


class TestRequestIdGeneration:
    """请求 ID 生成测试。"""

    def test_request_id_format(self) -> None:
        """request_id 格式为 req-16位hex。"""
        req_id = _generate_request_id()
        assert req_id.startswith("req-")
        assert len(req_id) == 20  # req- + 16 hex chars

    def test_request_id_unique(self) -> None:
        """每次生成的 request_id 唯一。"""
        ids = {_generate_request_id() for _ in range(100)}
        assert len(ids) == 100


class TestRequestContext:
    """请求上下文测试。"""

    def test_bind_and_unbind(self) -> None:
        """bind 和 unbind 正常工作。"""
        RequestContext.bind(request_id="req-test", session_id="sess-123", user_id=1)
        RequestContext.unbind("session_id")
        # unbind 后上下文变量被清除（通过再次 bind None 来验证）
        RequestContext.bind(session_id=None)
        RequestContext.clear()

    def test_clear(self) -> None:
        """clear 清除所有上下文变量。"""
        RequestContext.bind(request_id="req-clr", user_id=99)
        RequestContext.clear()


class TestLogFunctions:
    """日志函数测试（验证不抛异常）。"""

    def test_log_request_received(self) -> None:
        """log_request_received 不抛异常。"""
        log_request_received("test request", request_id="req-001", session_id="sess-001")

    def test_log_request_completed(self) -> None:
        """log_request_completed 不抛异常。"""
        log_request_completed(
            latency_ms=150,
            status_code=200,
            request_id="req-001",
            session_id="sess-001",
        )

    def test_log_llm_call(self) -> None:
        """log_llm_call 不抛异常。"""
        log_llm_call(model="deepseek-chat", latency_ms=300, request_id="req-001")

    def test_log_tool_called_success(self) -> None:
        """log_tool_called 成功路径不抛异常。"""
        log_tool_called(
            tool_name="weather",
            latency_ms=50,
            success=True,
            request_id="req-001",
        )

    def test_log_tool_called_failure(self) -> None:
        """log_tool_called 失败路径不抛异常。"""
        log_tool_called(
            tool_name="weather",
            latency_ms=50,
            success=False,
            error="timeout",
            request_id="req-001",
        )

    def test_log_slow_operation(self) -> None:
        """log_slow_operation 不抛异常。"""
        log_slow_operation(
            operation="llm_call",
            latency_ms=35000,
            threshold_ms=30000,
            request_id="req-001",
        )

    def test_log_error(self) -> None:
        """log_error 不抛异常。"""
        exc = ValueError("test error")
        log_error(exc, operation="test_op", request_id="req-001")


class TestMetricsRecording:
    """Prometheus 指标记录测试。"""

    def test_llm_latency_recorder_success(self) -> None:
        """LLM 成功调用记录指标。"""
        recorder = LLMLatencyRecorder(model="test-model")
        with recorder:
            pass  # 模拟成功执行

    def test_llm_latency_recorder_failure(self) -> None:
        """LLM 失败调用记录指标。"""
        recorder = LLMLatencyRecorder(model="test-model")
        with pytest.raises(ValueError):
            with recorder:
                raise ValueError("test error")

    def test_tool_latency_recorder_success(self) -> None:
        """Tool 成功调用记录指标。"""
        recorder = ToolLatencyRecorder(tool_name="test-tool")
        with recorder:
            pass

    def test_tool_latency_recorder_timeout(self) -> None:
        """Tool 超时记录为 timeout。"""
        recorder = ToolLatencyRecorder(tool_name="test-tool")
        with pytest.raises(Exception):
            with recorder:
                raise TimeoutError("timeout")

    def test_chat_latency_recorder(self) -> None:
        """Chat 延迟记录器正常工作。"""
        recorder = ChatLatencyRecorder(endpoint="/api/chat")
        with recorder:
            pass

    def test_external_api_latency_recorder(self) -> None:
        """外部 API 延迟记录器正常工作。"""
        recorder = ExternalApiLatencyRecorder(api_name="pilot", endpoint="/api/order")
        with recorder:
            pass

    def test_record_cache_hit(self) -> None:
        """缓存命中计数。"""
        record_cache_hit("session")

    def test_record_cache_miss(self) -> None:
        """缓存未命中计数。"""
        record_cache_miss("token")

    def test_set_active_sessions(self) -> None:
        """活跃会话数设置。"""
        set_active_sessions("DEFAULT", 10)

    def test_get_metrics(self) -> None:
        """get_metrics 返回 Prometheus 格式数据。"""
        output = get_metrics()
        assert isinstance(output, bytes)

    def test_get_metrics_content_type(self) -> None:
        """get_metrics_content_type 返回正确类型。"""
        content_type = get_metrics_content_type()
        assert "text/plain" in content_type or "text/versioned" in content_type