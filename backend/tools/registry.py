"""工具注册表，支持动态注册、熔断降级和超时控制。"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from enum import Enum
from typing import Any, Optional

from backend.tools.order import OrderTool
from backend.tools.ship import ShipTool
from backend.tools.weather import WeatherTool

logger = logging.getLogger(__name__)

ToolHandler = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


class CircuitState(Enum):
    """熔断器状态枚举。"""

    CLOSED = "closed"  # 正常状态
    OPEN = "open"  # 熔断状态
    HALF_OPEN = "half_open"  # 半开状态（试探恢复）


class ToolDegradedError(Exception):
    """工具降级异常，当熔断器打开时抛出。"""

    def __init__(self, tool_name: str) -> None:
        self.tool_name = tool_name
        super().__init__(f"工具 {tool_name} 已降级（熔断器打开）")


class ToolTimeoutError(Exception):
    """工具执行超时异常。"""

    def __init__(self, tool_name: str, timeout: float) -> None:
        self.tool_name = tool_name
        self.timeout = timeout
        super().__init__(f"工具 {tool_name} 执行超时（{timeout}s）")


class CircuitBreaker:
    """工具熔断器，防止外部系统故障影响主流程。"""

    def __init__(
        self,
        threshold: int = 3,
        timeout: int = 60,
    ) -> None:
        """初始化熔断器。

        Args:
            threshold: 连续失败次数阈值，达到后打开熔断器
            timeout: 熔断器打开后的恢复超时时间（秒）
        """

        self._threshold = threshold
        self._timeout = timeout
        self._failures = 0
        self._state = CircuitState.CLOSED
        self._opened_at: Optional[float] = None

    @property
    def state(self) -> CircuitState:
        """返回当前熔断器状态。"""

        if self._state == CircuitState.OPEN and self._opened_at is not None:
            # 检查是否超时，可以尝试恢复
            import time

            if time.time() - self._opened_at >= self._timeout:
                self._state = CircuitState.HALF_OPEN
                logger.info("circuit breaker half_open, will try recovery")
        return self._state

    async def call(
        self,
        func: Callable[..., Awaitable[Any]],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """通过熔断器执行异步函数。

        Args:
            func: 异步函数
            *args: 位置参数
            **kwargs: 关键字参数

        Returns:
            函数返回值

        Raises:
            ToolDegradedError: 熔断器打开时
        """

        if self.state == CircuitState.OPEN:
            raise ToolDegradedError("circuit open")

        try:
            result = await func(*args, **kwargs)
            # 成功时重置失败计数
            self._failures = 0
            return result
        except Exception as e:
            self._failures += 1
            logger.warning(
                "circuit breaker call failed",
                extra={"failures": self._failures, "threshold": self._threshold},
            )

            if self._failures >= self._threshold:
                import time

                self._state = CircuitState.OPEN
                self._opened_at = time.time()
                logger.error(
                    "circuit breaker opened",
                    extra={"tool": getattr(func, "__name__", "unknown")},
                )
            raise


# 全局熔断器字典，按工具名管理
_tool_circuit_breakers: dict[str, CircuitBreaker] = {}


def get_circuit_breaker(tool_name: str) -> CircuitBreaker:
    """获取或创建指定工具的熔断器。"""

    if tool_name not in _tool_circuit_breakers:
        _tool_circuit_breakers[tool_name] = CircuitBreaker(
            threshold=3,
            timeout=60,
        )
    return _tool_circuit_breakers[tool_name]


class ToolRegistry:
    """通过内存注册表解析和执行工具，支持动态注册、熔断和超时。"""

    SCENE_TO_TOOL = {
        "QUERY_WEATHER": "weather",
        "QUERY_SHIP": "ship",
        "SAVE_ORDER": "order",
    }

    def __init__(self) -> None:
        self._tools: dict[str, ToolHandler] = {
            "weather": WeatherTool().run,
            "ship": ShipTool().run,
            "order": OrderTool().run,
        }
        # 工具熔断器配置
        self._circuit_breakers: dict[str, CircuitBreaker] = {}
        # 工具默认超时配置（秒）
        self._timeouts: dict[str, float] = {
            "weather": 30.0,
            "ship": 30.0,
            "order": 30.0,
        }

    def register(
        self,
        tool_name: str,
        handler: ToolHandler,
        circuit_breaker: Optional[CircuitBreaker] = None,
        timeout: float = 30.0,
    ) -> None:
        """动态注册工具。

        Args:
            tool_name: 工具名称
            handler: 异步处理函数
            circuit_breaker: 可选的熔断器实例
            timeout: 执行超时时间（秒）
        """

        self._tools[tool_name] = handler
        if circuit_breaker is not None:
            self._circuit_breakers[tool_name] = circuit_breaker
        self._timeouts[tool_name] = timeout
        logger.info(
            "tool registered",
            extra={"tool_name": tool_name, "timeout": timeout},
        )

    def register_factory(
        self,
        tool_name: str,
        factory: Callable[[], ToolHandler],
    ) -> None:
        """通过工厂函数注册工具。

        Args:
            tool_name: 工具名称
            factory: 返回处理函数的工厂
        """

        self._tools[tool_name] = factory()
        self._timeouts[tool_name] = 30.0
        logger.info("tool factory registered", extra={"tool_name": tool_name})

    def get_tool_name(self, scene: Optional[str]) -> Optional[str]:
        """返回映射到场景的工具名称。"""

        if scene is None:
            return None
        return self.SCENE_TO_TOOL.get(scene)

    def has_tool(self, tool_name: str) -> bool:
        """检查注册表是否包含给定工具。"""

        return tool_name in self._tools

    def list_tools(self) -> list[str]:
        """返回所有已注册工具的名称列表。"""

        return list(self._tools.keys())

    async def call_tool(
        self,
        tool_name: str,
        payload: dict[str, Any],
        timeout: Optional[float] = None,
    ) -> dict[str, Any]:
        """执行工具并返回其标准化结果，支持熔断和超时。

        Args:
            tool_name: 工具名称
            payload: 工具输入参数
            timeout: 可选的执行超时时间（秒）

        Returns:
            工具执行结果

        Raises:
            ValueError: 工具不存在
            ToolDegradedError: 熔断器打开
            ToolTimeoutError: 执行超时
        """

        if tool_name not in self._tools:
            raise ValueError(f"Unknown tool: {tool_name}")

        handler = self._tools[tool_name]
        cb = self._circuit_breakers.get(
            tool_name,
            get_circuit_breaker(tool_name),
        )
        exec_timeout = timeout or self._timeouts.get(tool_name, 30.0)

        try:
            # 使用 asyncio.wait_for 实现超时控制
            result = await asyncio.wait_for(
                cb.call(handler, payload),
                timeout=exec_timeout,
            )
            return result
        except asyncio.TimeoutError:
            logger.error(
                "tool execution timeout",
                extra={"tool_name": tool_name, "timeout": exec_timeout},
            )
            raise ToolTimeoutError(tool_name, exec_timeout) from None