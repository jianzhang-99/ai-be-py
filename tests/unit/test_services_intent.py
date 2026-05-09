from __future__ import annotations

"""Unit tests for phase-one intent recognition."""

import pytest

from backend.graph.state.agent_state import SceneEnum
from backend.services.intent_service import IntentService


@pytest.mark.asyncio
async def test_rule_intent_matches_weather_keyword() -> None:
    """Weather questions should be resolved without calling the LLM."""

    service = IntentService()

    result = await service.recognize_by_rule("帮我查一下武汉天气", {})

    assert result == {"intent": SceneEnum.QUERY_WEATHER}


@pytest.mark.asyncio
async def test_rule_intent_supports_waiting_scene_continuation() -> None:
    """Waiting continuation should reuse the current business scene."""

    service = IntentService()

    result = await service.recognize_by_rule(
        "明天下午也要一条",
        {"current_scene": SceneEnum.SAVE_ORDER, "state": SceneEnum.WAITING_USER},
    )

    assert result == {"intent": SceneEnum.SAVE_ORDER}
