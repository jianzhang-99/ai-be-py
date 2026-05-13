"""提示词加载器 - 从数据库读取提示词，参考 Java 版 PromptLoader 设计。

支持：
1. 数据库读取（ai_prompt 表）
2. 代码 fallback（prompt_templates.py）
3. 缓存 + 动态刷新
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

from backend.infra.database.repositories.prompt_repo import PromptRepository
from backend.infra.llm.prompt_templates import (
    ORDER_ASSISTANT,
    ADDRESS_EXTRACT,
    CITY_PARAMS_EXTRACT,
    DISPATCH_ASSISTANT,
    FREIGHT_PARAMS_EXTRACT,
    FREIGHT_SUMMARY,
    INTELLIGENT_MCP_TOOL_CALL,
    INTENT_RECOGNIZER_SYSTEM,
    INTENT_RECOGNIZER_USER,
    LOCATION_SUMMARY,
    MEMORY_EXTRACT,
    QUICK_QUESTION_GENERATE,
    QUERY_ORDER_PARAMS_EXTRACT,
    SHIP_NAME_EXTRACT,
    SHIPPING_ROUTE_PARAMS_EXTRACT,
    TALK_SYSTEM,
    TALK_USER_CONTEXT,
    WATER_LEVEL_PARAMS_EXTRACT,
    WATER_LEVEL_SUMMARY,
    WEATHER_ASSISTANT,
    WEATHER_PARAMS_EXTRACT,
    get_now_date,
    render_template,
)

logger = logging.getLogger(__name__)

# 代码里的 fallback 提示词，与 Java 版一致
_FALLBACK_PROMPTS: dict[str, dict[str, str]] = {
    "intent_recognizer": {
        "system": INTENT_RECOGNIZER_SYSTEM,
        "user": INTENT_RECOGNIZER_USER,
    },
    "talk_system": {
        "system": TALK_SYSTEM,
        "user": TALK_USER_CONTEXT,
    },
    "order_assistant": {
        "system": ORDER_ASSISTANT,
        "user": "",
    },
    "weather_assistant": {
        "system": WEATHER_ASSISTANT,
        "user": "",
    },
    "weather_params_extract": {
        "system": WEATHER_PARAMS_EXTRACT,
        "user": "",
    },
    "dispatch_assistant": {
        "system": DISPATCH_ASSISTANT,
        "user": "",
    },
    "address_extract": {
        "system": ADDRESS_EXTRACT,
        "user": "",
    },
    "ship_name_extract": {
        "system": SHIP_NAME_EXTRACT,
        "user": "",
    },
    "shipping_route_params_extract": {
        "system": SHIPPING_ROUTE_PARAMS_EXTRACT,
        "user": "",
    },
    "freight_params_extract": {
        "system": FREIGHT_PARAMS_EXTRACT,
        "user": "",
    },
    "freight_summary": {
        "system": FREIGHT_SUMMARY,
        "user": "",
    },
    "city_params_extract": {
        "system": CITY_PARAMS_EXTRACT,
        "user": "",
    },
    "memory_extract": {
        "system": MEMORY_EXTRACT,
        "user": "",
    },
    "quick_question_generate": {
        "system": QUICK_QUESTION_GENERATE,
        "user": "",
    },
    "query_order_params_extract": {
        "system": QUERY_ORDER_PARAMS_EXTRACT,
        "user": "",
    },
    "water_level_params_extract": {
        "system": WATER_LEVEL_PARAMS_EXTRACT,
        "user": "",
    },
    "water_level_summary": {
        "system": WATER_LEVEL_SUMMARY,
        "user": "",
    },
    "location_summary": {
        "system": LOCATION_SUMMARY,
        "user": "",
    },
    "intelligent_mcp_tool_call": {
        "system": INTELLIGENT_MCP_TOOL_CALL,
        "user": "",
    },
}


class PromptLoader:
    """统一提示词加载器，优先从数据库读取，不存在则用代码 fallback。"""

    def __init__(self):
        self._repo = PromptRepository()
        self._cache: dict[str, dict[str, str]] = {}
        self._db_available: bool | None = None

    async def _ensure_db_available(self) -> bool:
        """检查数据库是否可用。"""
        if self._db_available is not None:
            return self._db_available
        try:
            await self._repo.list_all_enabled()
            self._db_available = True
        except Exception as e:
            logger.warning(f"数据库不可用，使用代码 fallback 提示词: {e}")
            self._db_available = False
        return self._db_available

    async def _load_from_db(self, code: str) -> dict[str, str] | None:
        """从数据库加载提示词。"""
        if not await self._ensure_db_available():
            return None
        try:
            row = await self._repo.get_enabled_by_code(code)
            if row and row.get("system_prompt"):
                return {
                    "system": (row.get("system_prompt") or "").strip(),
                    "user": (row.get("user_prompt") or "").strip(),
                }
        except Exception as e:
            logger.warning(f"从数据库加载提示词失败 {code}: {e}")
        return None

    def _load_from_code(self, code: str) -> dict[str, str] | None:
        """从代码 fallback 加载提示词。"""
        return _FALLBACK_PROMPTS.get(code)

    def _fill_variables(
        self,
        template: str,
        variables: dict[str, Any],
    ) -> str:
        """填充模板变量，支持 {{var}} 语法。"""
        if not variables:
            return template
        result = template
        for key, value in variables.items():
            placeholder = "{{" + key + "}}"
            result = result.replace(placeholder, str(value) if value is not None else "")
        return result

    async def get_prompt(
        self,
        code: str,
        variables: dict[str, Any] | None = None,
    ) -> tuple[str, str]:
        """获取提示词，返回 (system_prompt, user_prompt) 元组。

        Args:
            code: 提示词编码
            variables: 模板变量

        Returns:
            (system_prompt, user_prompt) 元组
        """
        # 先从缓存
        cached = self._cache.get(code)
        if cached is None:
            # 尝试从数据库加载
            db_prompt = await self._load_from_db(code)
            if db_prompt:
                self._cache[code] = db_prompt
                cached = db_prompt
            else:
                # 使用代码 fallback
                fallback = self._load_from_code(code)
                if fallback:
                    self._cache[code] = fallback
                    cached = fallback

        if cached is None:
            logger.error(f"提示词不存在: {code}")
            return "", ""

        system = cached.get("system", "")
        user = cached.get("user", "")

        if variables:
            system = self._fill_variables(system, variables)
            user = self._fill_variables(user, variables)

        return system, user

    async def get_system_prompt(self, code: str) -> str:
        """仅获取 system prompt。"""
        system, _ = await self.get_prompt(code)
        return system

    async def get_user_prompt(self, code: str, variables: dict[str, Any] | None = None) -> str:
        """仅获取 user prompt（已填充变量）。"""
        _, user = await self.get_prompt(code, variables)
        return user

    async def evict(self, code: str) -> None:
        """清除指定提示词的缓存。"""
        self._cache.pop(code, None)
        logger.info(f"提示词缓存已清除: {code}")

    async def evict_all(self) -> None:
        """清除所有提示词缓存。"""
        self._cache.clear()
        logger.info("提示词缓存已全部清除")


# 全局单例
_prompt_loader: PromptLoader | None = None


def get_prompt_loader() -> PromptLoader:
    """获取提示词加载器单例。"""
    global _prompt_loader
    if _prompt_loader is None:
        _prompt_loader = PromptLoader()
    return _prompt_loader


# 便捷函数，与 prompt_templates.py 接口兼容
async def intent_recognizer_prompt(
    user_utterance: str,
    current_scene: str = "unknown",
    in_progress: str = "none",
    task_mode: str = "normal",
    working_task_title: str = "",
    working_task_status: str = "",
    preprocess_hints: str = "",
) -> tuple[str, str]:
    """构建意图识别提示词。"""
    loader = get_prompt_loader()
    return await loader.get_prompt(
        "intent_recognizer",
        variables={
            "current_scene": current_scene,
            "in_progress": in_progress,
            "task_mode": task_mode,
            "working_task_title": working_task_title,
            "working_task_status": working_task_status,
            "user_utterance": user_utterance,
            "preprocess_hints": preprocess_hints,
        },
    )


async def talk_system_prompt(
    user_profile: str = "",
    chat_summaries: str = "",
    memory_hint: str = "",
) -> tuple[str, str]:
    """构建主对话提示词。"""
    loader = get_prompt_loader()
    return await loader.get_prompt(
        "talk_system",
        variables={
            "userProfile": user_profile or "未知用户",
            "chatSummaries": chat_summaries or "暂无历史对话",
            "memoryHint": memory_hint or "",
        },
    )