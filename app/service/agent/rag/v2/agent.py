"""/app/service/agent/rag/v2/agent.py

LangGraph 기반 Agentic RAG (v2)

흐름:
  route → [retrieve → grade → (rewrite → retrieve)* → generate] or [direct_generate]
"""

from typing import AsyncGenerator

from common.util.llm_gateway_client import LLMGatewayClient
from common.util.search_client import search_documents
from core.config import get_setting
from core.log.logging import get_logging
from service.agent.base import BaseAgent
from service.agent.rag.v2.graph import build_rag_graph
from service.agent.rag.v2.prompts import (
    DIRECT_PROMPT,
    GENERATE_PROMPT,
    GRADE_PROMPT,
    REWRITE_PROMPT,
    ROUTE_PROMPT,
)
from service.agent.registry import register
from service.model.agent import AgentRequest, AgentResponse, ChatHistory
from sqlalchemy.orm import Session

logger = get_logging()
settings = get_setting()


@register(name="rag", version="v2")
class RagAgentV2(BaseAgent):
    description = "LangGraph Agentic RAG — 자동 라우팅/검색 품질 평가/쿼리 리라이팅 (v2)"

    def _get_client(self) -> LLMGatewayClient:
        return LLMGatewayClient(llm_gateway_url=settings.LLM_GATEWAY_URL)

    async def _retrieve(self, query: str, metadata: dict | None = None) -> tuple[str, list[dict]]:
        """Azure AI Search로 문서 검색"""
        return await search_documents(query)

    def _build_log_metadata(self, graph_result: dict) -> dict:
        """로그에 저장할 메타데이터 구성 (사용된 프롬프트 포함)"""
        route = graph_result.get("route_decision", "")
        return {
            "route": route,
            "grade": graph_result.get("grade_decision", ""),
            "retry_count": graph_result.get("retry_count", 0),
            "prompts": {
                "route": ROUTE_PROMPT,
                "grade": GRADE_PROMPT if route == "retrieve" else None,
                "rewrite": REWRITE_PROMPT if graph_result.get("retry_count", 0) > 0 else None,
                "generate": GENERATE_PROMPT if route == "retrieve" else DIRECT_PROMPT,
            },
            "sources_count": len(graph_result.get("sources", [])),
        }

    async def _run_graph(self, request: AgentRequest, user: dict) -> dict:
        """LangGraph 실행하여 최종 상태를 반환"""
        client = self._get_client()

        graph = build_rag_graph(
            client=client,
            user_id=user.get("user_id", ""),
            org_id=request.org_id,
            provider=request.provider,
            model=request.model,
            retrieve_fn=self._retrieve,
        )

        initial_state = {
            "query": request.query,
            "original_query": request.query,
            "chat_history": [
                {"role": msg.role.value, "content": msg.content}
                for msg in request.chat_history
            ],
            "context": "",
            "sources": [],
            "answer": "",
            "route_decision": "",
            "grade_decision": "",
            "retry_count": 0,
        }

        return await graph.ainvoke(initial_state)

    async def invoke(self, request: AgentRequest, user: dict, *, db: Session, response_mode: str) -> AgentResponse:
        """비스트리밍: 그래프 실행 → DB 저장 → JSON 응답"""
        result = await self._run_graph(request, user)

        log_metadata = self._build_log_metadata(result)

        # DB 저장
        self._save_log(
            db, request, user, response_mode,
            answer=result.get("answer", ""),
            sources=result.get("sources", []),
            log_metadata=log_metadata,
        )

        return AgentResponse(
            answer=result.get("answer", ""),
            sources=result.get("sources", []),
            metadata={
                "route": result.get("route_decision", ""),
                "grade": result.get("grade_decision", ""),
                "retry_count": result.get("retry_count", 0),
            },
        )

    async def stream(
        self, request: AgentRequest, user: dict, *, db: Session, response_mode: str
    ) -> AsyncGenerator[str, None]:
        """스트리밍: 그래프로 컨텍스트 확보 → 최종 생성만 스트리밍"""
        client = self._get_client()
        result = await self._run_graph(request, user)
        sources = result.get("sources", [])

        if sources:
            yield self._format_sse({"type": "sources", "sources": sources})

        yield self._format_sse({
            "type": "metadata",
            "metadata": {
                "route": result.get("route_decision", ""),
                "grade": result.get("grade_decision", ""),
                "retry_count": result.get("retry_count", 0),
            },
        })

        # 최종 응답을 스트리밍으로 재생성
        messages = [ChatHistory(role="system", content=GENERATE_PROMPT)]
        for msg in request.chat_history:
            messages.append(msg)

        context = result.get("context", "")
        query = request.query
        if context:
            user_content = f"[참고 문서]\n{context}\n\n[질문]\n{query}"
        else:
            user_content = query
        messages.append(ChatHistory(role="user", content=user_content))

        async for chunk in client.call_completions_stream(
            user_id=user.get("user_id", ""),
            org_id=request.org_id,
            provider=request.provider,
            model=request.model,
            messages=messages,
            prompt_variables=None,
            agent_name=f"rag-{self.version}",
        ):
            yield chunk

        # DB 저장 (스트리밍이라 answer는 그래프 결과 사용)
        self._save_log(
            db, request, user, response_mode,
            answer=result.get("answer", ""),
            sources=sources,
            log_metadata=self._build_log_metadata(result),
        )

        yield self._format_sse_done()

    async def post_process_stream(
        self, request: AgentRequest, user: dict, *, db: Session, response_mode: str
    ) -> AsyncGenerator[str, None]:
        """후처리 후 스트리밍: 그래프 실행 → DB 저장 → SSE 전달"""
        response = await self.invoke(request, user, db=db, response_mode=response_mode)

        if response.sources:
            yield self._format_sse({"type": "sources", "sources": response.sources})

        yield self._format_sse({"type": "answer", "content": response.answer})
        yield self._format_sse({"type": "metadata", "metadata": response.metadata})
        yield self._format_sse_done()
