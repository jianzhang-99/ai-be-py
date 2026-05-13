"""统一 LLM 客户端，支持 DeepSeek 和通义千问。"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from typing import Any, Optional

from backend.config import get_settings

logger = logging.getLogger(__name__)


class LLMClient:
    """统一 LLM 客户端，参考 Java 版 LlmClient 架构。"""

    def __init__(self):
        self.settings = get_settings()
        self._client = None

    @property
    def client(self):
        """根据配置的 provider 返回对应客户端。"""
        if self._client is None:
            provider = self.settings.llm_provider
            if provider == "deepseek":
                from langchain_deepseek import ChatDeepSeek

                self._client = ChatDeepSeek(
                    model=self.settings.deepseek_model,
                    api_key=self.settings.deepseek_api_key,
                    temperature=0.7,
                    max_tokens=2000,
                )
            elif provider == "qwen":
                # qwen 通过 OpenAI 兼容接口调用
                from langchain_openai import ChatOpenAI

                self._client = ChatOpenAI(
                    model=self.settings.qwen_model,
                    base_url=self.settings.qwen_base_url,
                    api_key=self.settings.qwen_api_key,
                    temperature=0.7,
                )
            else:
                raise ValueError(f"Unsupported LLM provider: {provider}")
        return self._client

    def _build_messages(
        self,
        system_prompt: str,
        user_message: str,
        messages: Optional[list[dict[str, str]]] = None,
    ) -> list[Any]:
        """将普通历史记录转换为 LangChain 消息。"""
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        langchain_messages: list[Any] = [SystemMessage(content=system_prompt)]

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
        """调用大模型返回单个响应。"""
        provider = self.settings.llm_provider

        if provider == "tongyi":
            return await self._chat_tongyi(system_prompt, user_message, messages)

        # DeepSeek / Qwen 使用 LangChain
        langchain_messages = self._build_messages(system_prompt, user_message, messages)
        response = await self.client.ainvoke(langchain_messages)
        return response.content

    async def _chat_tongyi(
        self,
        system_prompt: str,
        user_message: str,
        messages: Optional[list[dict[str, str]]] = None,
    ) -> str:
        """使用通义千问 DashScope SDK 调用。"""
        import dashscope
        from dashscope import Generation

        api_key = self.settings.tongyi_api_key
        if not api_key:
            raise ValueError("通义千问 API Key 未配置 (TONGYI_API_KEY)")

        dashscope.api_key = api_key

        # 构建消息列表
        msgs = []
        if messages:
            for msg in messages[-10:]:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "user":
                    msgs.append({"role": "user", "content": content})
                elif role == "assistant":
                    msgs.append({"role": "assistant", "content": content})
                elif role == "system":
                    msgs.append({"role": "system", "content": content})

        msgs.append({"role": "user", "content": user_message})

        response = Generation.call(
            model=self.settings.tongyi_model,
            messages=[
                {"role": "system", "content": system_prompt},
                *msgs,
            ],
        )

        if response.status_code != 200:
            logger.error(f"通义千问调用失败: {response.message}")
            raise ValueError(f"通义千问调用失败: {response.message}")

        return response.output.text

    async def chat_stream(
        self,
        system_prompt: str,
        user_message: str,
        messages: Optional[list[dict[str, str]]] = None,
    ) -> AsyncGenerator[str, None]:
        """流式调用大模型。"""
        provider = self.settings.llm_provider

        if provider == "tongyi":
            async for chunk in self._chat_stream_tongyi(system_prompt, user_message, messages):
                yield chunk
            return

        # DeepSeek / Qwen 使用 LangChain
        langchain_messages = self._build_messages(system_prompt, user_message, messages)
        async for chunk in self.client.astream(langchain_messages):
            if chunk.content:
                yield chunk.content

    async def _chat_stream_tongyi(
        self,
        system_prompt: str,
        user_message: str,
        messages: Optional[list[dict[str, str]]] = None,
    ) -> AsyncGenerator[str, None]:
        """使用通义千问 DashScope SDK 流式调用。"""
        import dashscope
        from dashscope import Generation

        api_key = self.settings.tongyi_api_key
        if not api_key:
            raise ValueError("通义千问 API Key 未配置 (TONGYI_API_KEY)")

        dashscope.api_key = api_key

        # 构建消息列表
        msgs = []
        if messages:
            for msg in messages[-10:]:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "user":
                    msgs.append({"role": "user", "content": content})
                elif role == "assistant":
                    msgs.append({"role": "assistant", "content": content})
                elif role == "system":
                    msgs.append({"role": "system", "content": content})

        msgs.append({"role": "user", "content": user_message})

        responses = Generation.call(
            model=self.settings.tongyi_model,
            messages=[
                {"role": "system", "content": system_prompt},
                *msgs,
            ],
            stream=True,
            incremental_output=True,
        )

        for response in responses:
            if response.status_code == 200:
                yield response.output.text
            else:
                logger.error(f"通义千问流式调用失败: {response.message}")


# 全局单例
_llm_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """获取 LLM 客户端单例。"""
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client