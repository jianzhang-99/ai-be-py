"""Prometheus 指标暴露模块。

定义并暴露 LLM 延迟、工具延迟、QPS 等关键指标。
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Generator

from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST

# ============== 指标定义 ==============

# LLM 相关指标
LLM_REQUESTS_TOTAL = Counter(
    "ai_llm_requests_total",
    "LLM 请求总数",
    ["model", "status"],  # success / failure
)

LLM_LATENCY_SECONDS = Histogram(
    "ai_llm_latency_seconds",
    "LLM 延迟分布（秒）",
    ["model"],
    buckets=(0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 30.0, 60.0),
)

# 工具调用指标
TOOL_CALLS_TOTAL = Counter(
    "ai_tool_calls_total",
    "工具调用总数",
    ["tool_name", "status"],  # success / failure / timeout
)

TOOL_LATENCY_SECONDS = Histogram(
    "ai_tool_latency_seconds",
    "工具延迟分布（秒）",
    ["tool_name"],
    buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)

# 对话请求指标
CHAT_REQUESTS_TOTAL = Counter(
    "ai_chat_requests_total",
    "对话请求总数",
    ["endpoint", "status"],  # success / failure
)

CHAT_LATENCY_SECONDS = Histogram(
    "ai_chat_latency_seconds",
    "对话处理延迟分布（秒）",
    ["endpoint"],
    buckets=(0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0),
)

# 活跃会话数
ACTIVE_SESSIONS = Gauge(
    "ai_active_sessions",
    "当前活跃会话数",
    ["scene"],
)

# Token 使用量
TOKEN_USAGE_TOTAL = Counter(
    "ai_token_usage_total",
    "Token 使用总量",
    ["model", "type"],  # prompt / completion
)

# 外部 API 调用指标
EXTERNAL_API_REQUESTS_TOTAL = Counter(
    "ai_external_api_requests_total",
    "外部 API 请求总数",
    ["api_name", "endpoint", "status"],
)

EXTERNAL_API_LATENCY_SECONDS = Histogram(
    "ai_external_api_latency_seconds",
    "外部 API 延迟分布（秒）",
    ["api_name", "endpoint"],
    buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)

# Redis 缓存指标
CACHE_HITS_TOTAL = Counter(
    "ai_cache_hits_total",
    "缓存命中总数",
    ["cache_type"],  # session / token
)

CACHE_MISSES_TOTAL = Counter(
    "ai_cache_misses_total",
    "缓存未命中总数",
    ["cache_type"],
)

# ============== 指标记录工具 ==============


class LLMLatencyRecorder:
    """LLM 延迟记录器。"""

    def __init__(self, model: str) -> None:
        self.model = model
        self._start_time: float = 0

    def __enter__(self) -> "LLMLatencyRecorder":
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        latency = time.perf_counter() - self._start_time
        status = "success" if exc_type is None else "failure"
        LLM_REQUESTS_TOTAL.labels(model=self.model, status=status).inc()
        LLM_LATENCY_SECONDS.labels(model=self.model).observe(latency)

    def record_tokens(self, prompt_tokens: int = 0, completion_tokens: int = 0) -> None:
        """记录 Token 使用量。"""
        if prompt_tokens > 0:
            TOKEN_USAGE_TOTAL.labels(model=self.model, type="prompt").inc(prompt_tokens)
        if completion_tokens > 0:
            TOKEN_USAGE_TOTAL.labels(model=self.model, type="completion").inc(completion_tokens)


class ToolLatencyRecorder:
    """工具调用延迟记录器。"""

    def __init__(self, tool_name: str) -> None:
        self.tool_name = tool_name
        self._start_time: float = 0

    def __enter__(self) -> "ToolLatencyRecorder":
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        latency = time.perf_counter() - self._start_time
        if exc_type is not None:
            if "TimeoutError" in str(exc_type) or "TimeoutError" in str(exc_val):
                status = "timeout"
            else:
                status = "failure"
        else:
            status = "success"

        TOOL_CALLS_TOTAL.labels(tool_name=self.tool_name, status=status).inc()
        TOOL_LATENCY_SECONDS.labels(tool_name=self.tool_name).observe(latency)


class ChatLatencyRecorder:
    """对话延迟记录器。"""

    def __init__(self, endpoint: str) -> None:
        self.endpoint = endpoint
        self._start_time: float = 0

    def __enter__(self) -> "ChatLatencyRecorder":
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        latency = time.perf_counter() - self._start_time
        status = "success" if exc_type is None else "failure"
        CHAT_REQUESTS_TOTAL.labels(endpoint=self.endpoint, status=status).inc()
        CHAT_LATENCY_SECONDS.labels(endpoint=self.endpoint).observe(latency)


class ExternalApiLatencyRecorder:
    """外部 API 延迟记录器。"""

    def __init__(self, api_name: str, endpoint: str) -> None:
        self.api_name = api_name
        self.endpoint = endpoint
        self._start_time: float = 0

    def __enter__(self) -> "ExternalApiLatencyRecorder":
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        latency = time.perf_counter() - self._start_time
        status = "success" if exc_type is None else "failure"
        EXTERNAL_API_REQUESTS_TOTAL.labels(
            api_name=self.api_name, endpoint=self.endpoint, status=status
        ).inc()
        EXTERNAL_API_LATENCY_SECONDS.labels(api_name=self.api_name, endpoint=self.endpoint).observe(
            latency
        )


# ============== 便捷上下文管理器 ==============


@contextmanager
def track_llm(model: str) -> Generator[LLMLatencyRecorder, None, None]:
    """追踪 LLM 调用的上下文管理器。"""
    recorder = LLMLatencyRecorder(model)
    yield recorder


@contextmanager
def track_tool(tool_name: str) -> Generator[ToolLatencyRecorder, None, None]:
    """追踪工具调用的上下文管理器。"""
    recorder = ToolLatencyRecorder(tool_name)
    yield recorder


@contextmanager
def track_chat(endpoint: str) -> Generator[ChatLatencyRecorder, None, None]:
    """追踪对话处理的上下文管理器。"""
    recorder = ChatLatencyRecorder(endpoint)
    yield recorder


@contextmanager
def track_external_api(api_name: str, endpoint: str) -> Generator[ExternalApiLatencyRecorder, None, None]:
    """追踪外部 API 调用的上下文管理器。"""
    recorder = ExternalApiLatencyRecorder(api_name, endpoint)
    yield recorder


# ============== Gauge 操作 ==============


def set_active_sessions(scene: str, count: int) -> None:
    """设置活跃会话数。"""
    ACTIVE_SESSIONS.labels(scene=scene).set(count)


def inc_active_sessions(scene: str) -> None:
    """增加一个活跃会话。"""
    ACTIVE_SESSIONS.labels(scene=scene).inc()


def dec_active_sessions(scene: str) -> None:
    """减少一个活跃会话。"""
    ACTIVE_SESSIONS.labels(scene=scene).dec()


def record_cache_hit(cache_type: str) -> None:
    """记录缓存命中。"""
    CACHE_HITS_TOTAL.labels(cache_type=cache_type).inc()


def record_cache_miss(cache_type: str) -> None:
    """记录缓存未命中。"""
    CACHE_MISSES_TOTAL.labels(cache_type=cache_type).inc()


# ============== 指标暴露 ==============


def get_metrics() -> bytes:
    """获取 Prometheus 指标数据。"""
    return generate_latest()


def get_metrics_content_type() -> str:
    """获取 Prometheus 指标的内容类型。"""
    return CONTENT_TYPE_LATEST
