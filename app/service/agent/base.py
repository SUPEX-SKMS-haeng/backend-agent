"""/app/service/agent/base.py"""

import json
import uuid
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator

from core.log.logging import get_logging
from infra.database.repository import agent_log as agent_log_repo
from service.model.agent import AgentRequest, AgentResponse, ChatHistory
from sqlalchemy.orm import Session

logger = get_logging()


class BaseAgent(ABC):
    """
    에이전트 기본 클래스

    모든 에이전트는 이 클래스를 상속하며 3가지 응답 모드를 구현합니다:
    - invoke: 비스트리밍 (JSON 응답)
    - stream: 스트리밍 (SSE)
    - post_process_stream: 후처리 후 스트리밍 (SSE)

    모든 모드에서 실행 로그가 DB에 자동 저장됩니다.
    """

    name: str = ""
    version: str = ""
    description: str = ""

    @abstractmethod
    async def invoke(
        self, request: AgentRequest, user: dict, *, db: Session, response_mode: str
    ) -> AgentResponse:
        """비스트리밍 응답"""

    @abstractmethod
    async def stream(
        self, request: AgentRequest, user: dict, *, db: Session, response_mode: str
    ) -> AsyncGenerator[str, None]:
        """스트리밍 응답"""

    @abstractmethod
    async def post_process_stream(
        self, request: AgentRequest, user: dict, *, db: Session, response_mode: str
    ) -> AsyncGenerator[str, None]:
        """후처리 후 스트리밍 응답"""

    def _save_log(
        self,
        db: Session,
        request: AgentRequest,
        user: dict,
        response_mode: str,
        answer: str | None = None,
        sources: list[dict] | None = None,
        log_metadata: dict | None = None,
    ) -> str:
        """에이전트 실행 로그를 DB에 저장하고 trace_id를 반환"""
        trace_id = str(uuid.uuid4())
        try:
            agent_log_repo.create(
                db,
                trace_id=trace_id,
                session_id=request.session_id,
                user_id=user.get("user_id"),
                org_id=request.org_id,
                agent_name=self.name,
                agent_version=self.version,
                response_mode=response_mode,
                query=request.query,
                answer=answer,
                sources={"items": sources} if sources else None,
                log_metadata=log_metadata,
                provider=request.provider,
                model=request.model,
            )
            logger.info(f"[{self.name}-{self.version}] log saved: trace_id={trace_id}")
        except Exception as e:
            logger.error(f"[{self.name}-{self.version}] log save failed: {e}")
        return trace_id

    def _build_messages(
        self,
        system_prompt: str,
        query: str,
        chat_history: list[ChatHistory],
        context: str = "",
    ) -> list[ChatHistory]:
        """시스템 프롬프트 + 히스토리 + 컨텍스트 + 쿼리 조합"""
        messages = [ChatHistory(role="system", content=system_prompt)]

        for msg in chat_history:
            messages.append(msg)

        if context:
            user_content = f"[참고 문서]\n{context}\n\n[질문]\n{query}"
        else:
            user_content = query

        messages.append(ChatHistory(role="user", content=user_content))
        return messages

    def _format_sse(self, data: dict[str, Any]) -> str:
        """SSE 형식 포맷"""
        return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

    def _format_sse_done(self) -> str:
        return "data: [DONE]\n\n"
