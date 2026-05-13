from __future__ import annotations

"""Unit tests for phase-one intent recognition."""

import pytest

from backend.graph.state.agent_state import SceneEnum
from backend.services.intent_service import IntentService


@pytest.mark.asyncio
async def test_rule_intent_matches_weather_keyword() -> None:
    """Weather questions should be resolved without calling the custom model."""

    service = IntentService()

    result = await service.recognize_by_rule("帮我查一下武汉天气", {})

    assert result == {
        "intent": SceneEnum.QUERY_WEATHER,
        "confidence": 1.0,
        "method": "rule",
    }


@pytest.mark.asyncio
async def test_rule_intent_supports_waiting_scene_continuation() -> None:
    """Waiting continuation should reuse the current business scene."""

    service = IntentService()

    result = await service.recognize_by_rule(
        "明天下午也要一条",
        {"current_scene": SceneEnum.SAVE_ORDER, "state": SceneEnum.WAITING_USER},
    )

    assert result == {
        "intent": SceneEnum.SAVE_ORDER,
        "confidence": 1.0,
        "method": "rule",
    }


@pytest.mark.asyncio
async def test_model_intent_uses_custom_model_result(monkeypatch: pytest.MonkeyPatch) -> None:
    """Rule miss should delegate to the custom intent model and keep SceneEnum output."""

    service = IntentService()

    monkeypatch.setattr(
        service.model_runtime,
        "predict",
        lambda user_input, history: {
            "intent": SceneEnum.QUERY_SHIP,
            "confidence": 0.93,
        },
    )

    result = await service.recognize_by_model("帮我看看这条船到哪了", [])

    assert result == {
        "intent": SceneEnum.QUERY_SHIP,
        "confidence": 0.93,
        "method": "custom_model",
    }
