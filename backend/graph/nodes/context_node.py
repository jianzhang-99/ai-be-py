from __future__ import annotations

"""上下文装配节点。"""

from typing import Any

from backend.graph.state.agent_state import AgentState
from backend.infra.cache.session_store import SessionStore, get_session_store
from backend.infra.database.repositories.chat_log_repo import ChatLogRepository
from backend.infra.database.repositories.chat_summary_repo import ChatSummaryRepository
from backend.infra.database.repositories.user_memory_profile_repo import (
    UserMemoryProfileRepository,
)

L1_MAX_ROUNDS = 10
L2_MAX_SUMMARIES = 5
L3_MAX_PROFILES = 10
MEMORY_HINT = "回答用户问题时，如相关则参考以上信息，但不要主动提及或暴露这些信息的存在。"


class ContextNode:
    """为后续节点装配会话上下文。"""

    def __init__(
        self,
        chat_log_repo: ChatLogRepository | None = None,
        chat_summary_repo: ChatSummaryRepository | None = None,
        user_memory_repo: UserMemoryProfileRepository | None = None,
        session_store: SessionStore | None = None,
    ) -> None:
        self.chat_log_repo = chat_log_repo or ChatLogRepository()
        self.chat_summary_repo = chat_summary_repo or ChatSummaryRepository()
        self.user_memory_repo = user_memory_repo or UserMemoryProfileRepository()
        self.session_store = session_store or get_session_store()

    async def entrypoint(self, state: AgentState) -> AgentState:
        """读取并合并当前请求可用的上下文。"""

        session_id = state.get("session_id", "")
        user_id = state.get("user_id", "")
        request_history = state.get("history", [])
        parsed_user_id = self._parse_user_id(user_id)
        phone = self._infer_phone(user_id, parsed_user_id)

        persisted_history = await self._load_persisted_history(session_id)
        working_memory = await self._load_working_memory(session_id)
        chat_summaries = await self._load_chat_summaries(session_id)
        user_profile = await self._load_user_profile(parsed_user_id, phone)
        merged_history = self._merge_history(persisted_history, request_history)

        return {
            "history": merged_history,
            "working_memory": working_memory,
            "chat_summaries": chat_summaries,
            "user_profile": user_profile,
            "memory_hint": MEMORY_HINT,
        }

    def _parse_user_id(self, user_id: str | None) -> int | None:
        try:
            return int(user_id) if user_id is not None else None
        except (TypeError, ValueError):
            return None

    def _infer_phone(self, user_id: str | None, parsed_user_id: int | None) -> str | None:
        if parsed_user_id is not None and user_id is not None:
            return user_id
        if user_id and user_id != "anonymous":
            return user_id
        return None

    async def _load_persisted_history(self, session_id: str) -> list[dict[str, str]]:
        if not session_id:
            return []

        try:
            rows = await self.chat_log_repo.list_by_session_id(
                session_id=session_id,
                limit=L1_MAX_ROUNDS,
            )
        except Exception:
            return []
        history: list[dict[str, str]] = []
        for row in rows[-L1_MAX_ROUNDS:]:
            user_input = str(row.get("user_input") or "").strip()
            ai_response = str(row.get("ai_response") or "").strip()
            if user_input:
                history.append({"role": "user", "content": user_input})
            if ai_response:
                history.append({"role": "assistant", "content": ai_response})
        return history

    async def _load_working_memory(self, session_id: str) -> dict[str, Any]:
        if not session_id:
            return {}

        try:
            data = await self.session_store.get_session_data(session_id)
        except Exception:
            return {}
        return data if isinstance(data, dict) else {}

    async def _load_chat_summaries(self, session_id: str) -> str:
        if not session_id:
            return ""

        try:
            rows = await self.chat_summary_repo.list_by_session_id(
                session_id=session_id,
                limit=L2_MAX_SUMMARIES,
            )
        except Exception:
            return ""

        lines = [
            f"- {str(row.get('summary_content') or '').strip()}"
            for row in reversed(rows)
            if str(row.get("summary_content") or "").strip()
        ]
        return "\n".join(lines)

    async def _load_user_profile(self, user_id: int | None, phone: str | None) -> str:
        try:
            rows = await self.user_memory_repo.list_active(
                user_id=user_id,
                phone=phone,
                limit=L3_MAX_PROFILES,
            )
        except Exception:
            return ""

        lines = [
            f"- {str(row.get('memory_content') or '').strip()}"
            for row in rows
            if str(row.get("memory_content") or "").strip()
        ]
        return "\n".join(lines)

    def _merge_history(
        self,
        persisted_history: list[dict[str, str]],
        request_history: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        if not persisted_history:
            return list(request_history)
        if not request_history:
            return list(persisted_history)
        if persisted_history == request_history:
            return list(persisted_history)

        if len(request_history) >= len(persisted_history):
            tail = request_history[-len(persisted_history):]
            if tail == persisted_history:
                return list(request_history)

        if len(persisted_history) >= len(request_history):
            tail = persisted_history[-len(request_history):]
            if tail == request_history:
                return list(persisted_history)

        return [*persisted_history, *request_history]
