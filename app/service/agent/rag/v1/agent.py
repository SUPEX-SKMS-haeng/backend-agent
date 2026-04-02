"""/app/service/agent/rag/v1/agent.py"""

from typing import AsyncGenerator

from common.util.llm_gateway_client import LLMGatewayClient
from common.util.search_client import search_documents
from core.config import get_setting
from core.log.logging import get_logging
from service.agent.base import BaseAgent
from service.agent.rag.v1.prompts import SYSTEM_PROMPT
from service.agent.registry import register
from service.model.agent import AgentRequest, AgentResponse
from sqlalchemy.orm import Session

logger = get_logging()
settings = get_setting()


@register(name="rag", version="v1")
class RagAgentV1(BaseAgent):
    description = "RAG 기반 문서 검색 + 응답 생성 (v1)"

    def _get_client(self) -> LLMGatewayClient:
        return LLMGatewayClient(llm_gateway_url=settings.LLM_GATEWAY_URL)

    async def _retrieve(self, query: str, metadata: dict | None = None) -> tuple[str, list[dict]]:
        """Azure AI Search로 문서 검색"""
        return await search_documents(query)

    def _build_log_metadata(self, sources: list[dict]) -> dict:
        """로그에 저장할 메타데이터 구성"""
        return {
            "prompts": {
                "system": SYSTEM_PROMPT,
            },
            "sources_count": len(sources),
        }

    async def invoke(self, request: AgentRequest, user: dict, *, db: Session, response_mode: str) -> AgentResponse:
        """비스트리밍: RAG 검색 → LLM 호출 → JSON 응답"""
        client = self._get_client()
        context, sources = await self._retrieve(request.query, request.metadata)

        messages = self._build_messages(
            system_prompt=SYSTEM_PROMPT,
            query=request.query,
            chat_history=request.chat_history,
            context=context,
        )

        result = await client.call_completions_non_stream(
            user_id=user.get("user_id", ""),
            org_id=request.org_id,
            provider=request.provider,
            model=request.model,
            messages=messages,
            prompt_variables=None,
            agent_name=f"{self.name}-{self.version}",
        )

        answer = ""
        if "choices" in result:
            answer = result["choices"][0].get("message", {}).get("content", "")
        elif "content" in result:
            answer = result["content"]

        # DB 저장
        self._save_log(
            db, request, user, response_mode,
            answer=answer,
            sources=sources,
            log_metadata=self._build_log_metadata(sources),
        )

        return AgentResponse(answer=answer, sources=sources)

    async def stream(
        self, request: AgentRequest, user: dict, *, db: Session, response_mode: str
    ) -> AsyncGenerator[str, None]:
        """스트리밍: RAG 검색 → LLM 스트리밍 → SSE 전달"""
        client = self._get_client()
        context, sources = await self._retrieve(request.query, request.metadata)

        messages = self._build_messages(
            system_prompt=SYSTEM_PROMPT,
            query=request.query,
            chat_history=request.chat_history,
            context=context,
        )

        if sources:
            yield self._format_sse({"type": "sources", "sources": sources})

        async for chunk in client.call_completions_stream(
            user_id=user.get("user_id", ""),
            org_id=request.org_id,
            provider=request.provider,
            model=request.model,
            messages=messages,
            prompt_variables=None,
            agent_name=f"{self.name}-{self.version}",
        ):
            yield chunk

        # 스트리밍은 전체 answer를 모으기 어려우므로 query + metadata만 저장
        self._save_log(
            db, request, user, response_mode,
            sources=sources,
            log_metadata=self._build_log_metadata(sources),
        )

        yield self._format_sse_done()

    async def post_process_stream(
        self, request: AgentRequest, user: dict, *, db: Session, response_mode: str
    ) -> AsyncGenerator[str, None]:
        """후처리 후 스트리밍: RAG 검색 → LLM 완성 → DB 저장 → SSE 전달"""
        response = await self.invoke(request, user, db=db, response_mode=response_mode)

        if response.sources:
            yield self._format_sse({"type": "sources", "sources": response.sources})

        yield self._format_sse({"type": "answer", "content": response.answer})
        yield self._format_sse({"type": "metadata", "metadata": response.metadata})
        yield self._format_sse_done()
