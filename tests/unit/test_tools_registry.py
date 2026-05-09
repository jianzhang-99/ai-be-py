from __future__ import annotations

"""Unit tests for the phase-one tool registry."""

import pytest

from backend.tools.registry import ToolRegistry


@pytest.mark.asyncio
async def test_registry_maps_scene_to_weather_tool() -> None:
    """Scene mapping should resolve the configured weather tool."""

    registry = ToolRegistry()

    assert registry.get_tool_name("QUERY_WEATHER") == "weather"
    assert registry.has_tool("weather") is True


@pytest.mark.asyncio
async def test_registry_executes_order_tool() -> None:
    """Registry execution should return a structured order preview."""

    registry = ToolRegistry()

    result = await registry.call_tool("order", {"text": "5月10日武汉装煤5000吨到南京"})

    assert result["loading_port"] == "武汉"
    assert result["discharge_port"] == "南京"
    assert result["cargo_name"] == "煤"
    assert result["tons"] == "5000"
