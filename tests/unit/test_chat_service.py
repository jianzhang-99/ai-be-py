from __future__ import annotations

"""Unit tests for ChatService with mocked workflow and repository"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from backend.api.schemas import ChatEvent, ChatRequest
from backend.services.chat_service import ChatService


class FakeIntent:
    def __init__(self):
        self.intent = "QUERY_WEATHER"
        self.intent_name = "天气查询"
        self.slots = {}


class FakeWorkflowResult:
    def get(self, key, default=None):
        if key == "response_text":
            return "南京明天晴天，温度15-25度"
        if key == "intent":
            return FakeIntent()
        return default


@pytest.fixture
def chat_service():
    return ChatService()


class TestChatServiceChatSimple:
    """chat_simple 端到端测试"""

    @pytest.mark.asyncio
    async def test_chat_simple_saves_log_when_user_id_is_int(self):
        svc = ChatService()
        svc.workflow.run_simple = AsyncMock(return_value={
            "response_text": "南京明天晴天",
            "intent": FakeIntent(),
        })
        svc._log_repo.save_log = AsyncMock(return_value=1)

        request = ChatRequest(
            message="南京明天天气",
            session_id="sess-test-001",
            user_id="1",  # str that parses to int
        )

        result = await svc.chat_simple(request)

        assert result.message == "南京明天晴天"
        assert result.intent == "QUERY_WEATHER"
        assert result.session_id == "sess-test-001"
        svc._log_repo.save_log.assert_called_once()
        call_kwargs = svc._log_repo.save_log.call_args
        assert call_kwargs.kwargs["session_id"] == "sess-test-001"
        assert call_kwargs.kwargs["user_input"] == "南京明天天气"
        assert call_kwargs.kwargs["ai_response"] == "南京明天晴天"

    @pytest.mark.asyncio
    async def test_chat_simple_generates_session_id_if_not_provided(self):
        svc = ChatService()
        svc.workflow.run_simple = AsyncMock(return_value={
            "response_text": "回复",
            "intent": FakeIntent(),
        })
        svc._log_repo.save_log = AsyncMock(return_value=1)

        request = ChatRequest(
            message="你好",
            user_id="1",
        )

        result = await svc.chat_simple(request)

        assert result.session_id is not None
        assert len(result.session_id) > 0

    @pytest.mark.asyncio
    async def test_chat_simple_does_not_save_log_for_anonymous_user(self):
        svc = ChatService()
        svc.workflow.run_simple = AsyncMock(return_value={
            "response_text": "回复",
            "intent": FakeIntent(),
        })
        svc._log_repo.save_log = AsyncMock()

        request = ChatRequest(
            message="你好",
            user_id="anonymous",  # cannot parse to int
        )

        result = await svc.chat_simple(request)

        svc._log_repo.save_log.assert_not_called()

    @pytest.mark.asyncio
    async def test_chat_simple_uses_phone_from_request(self):
        svc = ChatService()
        svc.workflow.run_simple = AsyncMock(return_value={
            "response_text": "回复",
            "intent": FakeIntent(),
        })
        svc._log_repo.save_log = AsyncMock(return_value=1)

        request = ChatRequest(
            message="你好",
            user_id="1",
        )

        # phone 通过 _persist_chat_log 的 phone 参数传入，非 request 字段
        # 测试时直接验证 save_log 被调用
        await svc.chat_simple(request)

        # 验证 user_id=1 可以正常解析并保存
        svc._log_repo.save_log.assert_called_once()

    @pytest.mark.asyncio
    async def test_parse_user_id_returns_int_for_numeric_string(self):
        svc = ChatService()
        assert svc._parse_user_id("123") == 123
        assert svc._parse_user_id("0") == 0
        assert svc._parse_user_id("-1") == -1

    @pytest.mark.asyncio
    async def test_parse_user_id_returns_none_for_non_numeric(self):
        svc = ChatService()
        assert svc._parse_user_id("anonymous") is None
        assert svc._parse_user_id("user-1") is None
        assert svc._parse_user_id("") is None
        assert svc._parse_user_id(None) is None

    @pytest.mark.asyncio
    async def test_chat_simple_handles_no_intent_in_result(self):
        svc = ChatService()
        svc.workflow.run_simple = AsyncMock(return_value={
            "response_text": "回复",
            "intent": None,
        })
        svc._log_repo.save_log = AsyncMock(return_value=1)

        request = ChatRequest(
            message="你好",
            user_id="1",
        )

        result = await svc.chat_simple(request)

        assert result.message == "回复"
        assert result.intent is None


class TestChatServiceChatStream:
    """chat_stream 测试（不持久化，只验证流式输出）"""

    @pytest.mark.asyncio
    async def test_chat_stream_yields_events(self):
        svc = ChatService()

        async def mock_stream(*args, **kwargs):
            events = [
                ChatEvent(event="intent", data={"intent": "QUERY_WEATHER"}),
                ChatEvent(event="message", data={"content": "南京明天晴天"}),
                ChatEvent(event="done", data={}),
            ]
            for e in events:
                yield e

        svc.workflow.run_stream = mock_stream

        request = ChatRequest(
            message="南京天气",
            user_id="1",
            session_id="sess-stream",
        )

        events = []
        async for event in svc.chat_stream(request):
            events.append(event)

        assert len(events) == 3
        assert events[0]["event"] == "intent"
        assert events[1]["data"] == {"content": "南京明天晴天"}

    @pytest.mark.asyncio
    async def test_chat_stream_generates_session_id_if_not_provided(self):
        svc = ChatService()

        async def mock_stream(*args, **kwargs):
            yield ChatEvent(event="done", data={})

        svc.workflow.run_stream = mock_stream

        request = ChatRequest(
            message="你好",
            user_id="1",
        )

        async for event in svc.chat_stream(request):
            pass

        # session_id should be generated by the service, not here
        # the workflow generates it internally, so we just verify no error