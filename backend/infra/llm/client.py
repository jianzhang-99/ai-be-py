from __future__ import annotations

"""用于本地阶段一开发的 LLM 客户端，带有 mock 回退机制。"""

from collections.abc import AsyncGenerator
from typing import Any, Optional

from backend.config import get_settings


class LLMClient:
    """供服务和节点使用的轻量级 LLM 抽象层。"""

    def __init__(self):
        self.settings = get_settings()
        self.provider = self.settings.llm_provider
        self._client = None

    @property
    def client(self):
        """延迟创建真实客户端，以便本地测试可以使用 mock 路径。"""

        if self._client is None:
            if self.provider == "deepseek":
                from langchain_deepseek import ChatDeepSeek

                self._client = ChatDeepSeek(
                    model=self.settings.deepseek_model,
                    api_key=self.settings.deepseek_api_key,
                    temperature=0.7,
                    max_tokens=2000,
                )
            elif self.provider == "qwen":
                from langchain_openai import ChatOpenAI

                self._client = ChatOpenAI(
                    model=self.settings.qwen_model,
                    base_url=self.settings.qwen_base_url,
                    api_key=self.settings.qwen_api_key,
                    temperature=0.7,
                )
            else:
                raise ValueError(f"Unsupported LLM provider: {self.provider}")
        return self._client

    def _should_use_mock(self) -> bool:
        """在阶段一验证期间避免对外部密钥的硬依赖。"""

        return self.settings.enable_mock_llm and not self.settings.deepseek_api_key

    def _build_mock_response(self, user_message: str) -> str:
        """返回可预测的文本，用于本地开发和测试。"""

        normalized = user_message.lower()
        if "query_weather" in normalized or "天气" in normalized:
            return "QUERY_WEATHER"
        if "query_ship" in normalized or "查船" in normalized or "船舶" in normalized:
            return "QUERY_SHIP"
        if "save_order" in normalized or "录单" in normalized or "运单" in normalized:
            return "SAVE_ORDER"
        return "我是航运助手小吨，当前运行在本地 MVP 模式，可以先回答闲聊并串联工具结果。"

    def _build_messages(
        self,
        system_prompt: str,
        user_message: str,
        messages: Optional[list[dict[str, str]]] = None,
    ) -> list[Any]:
        """将普通历史记录转换为 LangChain 消息。"""

        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        langchain_messages: list[Any] = [
            SystemMessage(content=system_prompt)
        ]

        if messages:
            for msg in messages[-10:]:
                role = msg.get("role")
                content = msg.get("content", "")
                if role == "user":
                    langchain_messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    langchain_messages.append(AIMessage(content=content))
                else:
                    langchain_messages.append(SystemMessage(content=content))

        langchain_messages.append(HumanMessage(content=user_message))
        return langchain_messages

    async def chat(
        self,
        system_prompt: str,
        user_message: str,
        messages: Optional[list[dict[str, str]]] = None,
    ) -> str:
        """返回单个模型响应字符串。"""

        if self._should_use_mock():
            return self._build_mock_response(user_message)

        langchain_messages = self._build_messages(system_prompt, user_message, messages)
        response = await self.client.ainvoke(langchain_messages)
        return response.content

    async def chat_stream(
        self,
        system_prompt: str,
        user_message: str,
        messages: Optional[list[dict[str, str]]] = None,
    ) -> AsyncGenerator[str, None]:
        """为流式 API 生成响应块。"""

        if self._should_use_mock():
            yield self._build_mock_response(user_message)
            return

        langchain_messages = self._build_messages(system_prompt, user_message, messages)
        async for chunk in self.client.astream(langchain_messages):
            if chunk.content:
                yield chunk.content
