"""/app/service/agent/mentor/v1/agent.py

SK 멘토링 에이전트 (mentor/v1)

LangGraph 기반 7단계 Agentic RAG:
  intent_classify → query_expand → retrieve → grade
    → (rewrite → query_expand → retrieve)*
    → generate → validate → END
"""

from typing import AsyncGenerator

from common.util.llm_gateway_client import LLMGatewayClient
from common.util.search_client import search_documents
from core.config import get_setting
from core.log.logging import get_logging
from service.agent.base import BaseAgent
from service.agent.mentor.v1.graph import build_mentor_graph
from service.agent.mentor.v1.prompts import GENERATE_PROMPT, GRADE_PROMPT, INTENT_PROMPT, REWRITE_PROMPT
from service.agent.registry import register
from service.model.agent import AgentRequest, AgentResponse, ChatHistory
from sqlalchemy.orm import Session

logger = get_logging()
settings = get_setting()


@register(name="mentor", version="v1")
class MentoringAgent(BaseAgent):
    description = "LangGraph SK 멘토링 에이전트 — 인텐트 분류/멀티쿼리 검색/SKMS 검증"

    def _get_client(self) -> LLMGatewayClient:
        return LLMGatewayClient(llm_gateway_url=settings.LLM_GATEWAY_URL)

    # 인텐트별 우선 검색 인덱스 매핑 (설정이 없으면 기본 인덱스 사용)
    _INTENT_INDEX_MAP: dict[str, str | None] = {
        "strategy": "skms-strategy",
        "culture":  "skms-culture",
        "crisis":   "skms-crisis",
        "supex":    "skms-supex",
        "hr":       "skms-hr",
        "general":  None,  # 기본 인덱스
    }

    async def _retrieve(self, query: str, metadata: dict | None = None) -> tuple[str, list[dict]]:
        """Azure AI Search 문서 검색 — metadata.intent로 인덱스 선택"""
        intent = (metadata or {}).get("intent", "general")
        index_name = self._INTENT_INDEX_MAP.get(intent)  # None이면 settings 기본값 사용
        return await search_documents(query, index_name=index_name)

    def _build_log_metadata(self, graph_result: dict) -> dict:
        """로그에 저장할 메타데이터 구성"""
        return {
            "intent":        graph_result.get("intent", ""),
            "confidence":    graph_result.get("intent_confidence", 0.0),
            "grade":         graph_result.get("grade_decision", ""),
            "retry_count":   graph_result.get("retry_count", 0),
            "validate":      graph_result.get("validate_result", ""),
            "sources_count": len(graph_result.get("sources", [])),
            "prompts": {
                "intent":   INTENT_PROMPT,
                "grade":    GRADE_PROMPT,
                "rewrite":  REWRITE_PROMPT if graph_result.get("retry_count", 0) > 0 else None,
                "generate": GENERATE_PROMPT,
            },
        }

    async def _run_graph(self, request: AgentRequest, user: dict) -> dict:
        """LangGraph 실행하여 최종 상태를 반환"""
        client = self._get_client()

        graph = build_mentor_graph(
            client=client,
            user_id=user.get("user_id", ""),
            org_id=request.org_id,
            provider=request.provider,
            model=request.model,
            retrieve_fn=self._retrieve,
            agent_name=f"{self.name}-{self.version}",
        )

        initial_state = {
            "query":             request.query,
            "original_query":    request.query,
            "chat_history": [
                {"role": msg.role.value, "content": msg.content}
                for msg in request.chat_history
            ],
            "intent":            "general",
            "intent_confidence": 0.0,
            "search_queries":    [],
            "context":           "",
            "sources":           [],
            "grade_decision":    "insufficient",
            "rewritten_query":   "",
            "retry_count":       0,
            "answer":            "",
            "validate_result":   "pass",
            "validate_reason":   "",
            "validate_count":    0,
        }

        return await graph.ainvoke(initial_state)

    async def invoke(
        self, request: AgentRequest, user: dict, *, db: Session, response_mode: str
    ) -> AgentResponse:
        """비스트리밍: 그래프 실행 → DB 저장 → JSON 응답"""
        result = await self._run_graph(request, user)

        self._save_log(
            db, request, user, response_mode,
            answer=result.get("answer", ""),
            sources=result.get("sources", []),
            log_metadata=self._build_log_metadata(result),
        )

        return AgentResponse(
            answer=result.get("answer", ""),
            sources=result.get("sources", []),
            metadata={
                "intent":      result.get("intent", ""),
                "grade":       result.get("grade_decision", ""),
                "retry_count": result.get("retry_count", 0),
                "validate":    result.get("validate_result", ""),
            },
        )

    async def stream(
        self, request: AgentRequest, user: dict, *, db: Session, response_mode: str
    ) -> AsyncGenerator[str, None]:
        """스트리밍: 그래프로 컨텍스트/출처 확보 → 최종 생성만 스트리밍"""
        client = self._get_client()
        result = await self._run_graph(request, user)
        sources = result.get("sources", [])

        if sources:
            yield self._format_sse({"type": "sources", "sources": sources})

        yield self._format_sse({
            "type": "metadata",
            "metadata": {
                "intent":      result.get("intent", ""),
                "grade":       result.get("grade_decision", ""),
                "retry_count": result.get("retry_count", 0),
                "validate":    result.get("validate_result", ""),
            },
        })

        # 그래프에서 확보한 컨텍스트로 최종 답변을 스트리밍 재생성
        messages = [ChatHistory(role="system", content=GENERATE_PROMPT)]

        for msg in request.chat_history:
            messages.append(msg)

        context = result.get("context", "")
        query = request.query
        if context:
            user_content = f"## 참고 자료\n{context}\n\n## 질문\n{query}"
        else:
            user_content = f"## 질문\n{query}"

        messages.append(ChatHistory(role="user", content=user_content))

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
        """후처리 후 스트리밍: 그래프 전체 실행 → DB 저장 → SSE 전달"""
        response = await self.invoke(request, user, db=db, response_mode=response_mode)

        if response.sources:
            yield self._format_sse({"type": "sources", "sources": response.sources})

        yield self._format_sse({"type": "answer", "content": response.answer})
        yield self._format_sse({"type": "metadata", "metadata": response.metadata})
        yield self._format_sse_done()
