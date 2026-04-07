"""/app/service/agent/mentor/v1/agent.py

SK 멘토링 에이전트 (mentor/v1)

LangGraph 기반 7단계 Agentic RAG:
  intent_classify → query_expand → retrieve → grade
    → (rewrite → query_expand → retrieve)*
    → generate → validate → END
"""

import json
import re
import time
from typing import AsyncGenerator

from common.util.llm_gateway_client import LLMGatewayClient
from common.util.hybrid_search_client import hybrid_search_documents
from core.config import get_setting
from core.log.logging import get_logging
from service.agent.base import BaseAgent
from service.agent.mentor.v1.graph import (
    build_generate_messages,
    build_mentor_graph,
    parse_archetype,
    reinforce_ije,
    run_pre_generate,
)
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
        "strategy": "sk_books",
        "culture":  "sk_history_archive",
        "crisis":   "sk_skms",
        "supex":    None,  # 전용 인덱스 미생성 → 기본 인덱스 사용
        "hr":       None,  # 전용 인덱스 미생성 → 기본 인덱스 사용
        "general":  None,  # 기본 인덱스
    }

    async def _retrieve(self, query: str, metadata: dict | None = None) -> tuple[str, list[dict]]:
        """Azure AI Search 문서 검색 — metadata.intent로 인덱스 선택"""
        intent = (metadata or {}).get("intent", "general")
        index_name = self._INTENT_INDEX_MAP.get(intent)  # None이면 settings 기본값 사용
        return await hybrid_search_documents(query, index_name=index_name)

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

        chat_history = [
            {"role": msg.role.value, "content": msg.content}
            for msg in request.chat_history
        ]

        # chat_history에서 사용자 턴 수 역산
        user_turn_count = len([m for m in chat_history if m["role"] == "user"])

        initial_state = {
            "query":             request.query,
            "original_query":    request.query,
            "chat_history":      chat_history,
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
            # 응답 아키타입 관리
            "response_archetype":  "",
            "previous_archetypes": [],
            "turn_count":          user_turn_count,
            "previous_answer":     "",
        }

        return await graph.ainvoke(initial_state)

    async def invoke(
        self, request: AgentRequest, user: dict, *, db: Session, response_mode: str
    ) -> AgentResponse:
        """비스트리밍: 그래프 실행 → DB 저장 → JSON 응답"""
        start_time = time.time()
        result = await self._run_graph(request, user)
        elapsed_seconds = round(time.time() - start_time, 1)

        log_meta = self._build_log_metadata(result)
        log_meta["elapsed_seconds"] = elapsed_seconds
        self._save_log(
            db, request, user, response_mode,
            answer=result.get("answer", ""),
            sources=result.get("sources", []),
            log_metadata=log_meta,
        )

        return AgentResponse(
            answer=result.get("answer", ""),
            sources=result.get("sources", []),
            metadata={
                "intent":          result.get("intent", ""),
                "grade":           result.get("grade_decision", ""),
                "retry_count":     result.get("retry_count", 0),
                "validate":        result.get("validate_result", ""),
                "elapsed_seconds": elapsed_seconds,
            },
        )

    async def stream(
        self, request: AgentRequest, user: dict, *, db: Session, response_mode: str
    ) -> AsyncGenerator[str, None]:
        """스트리밍: 검색 파이프라인 실행 → generate를 실제 LLM 스트리밍으로 전송"""
        start_time = time.time()
        client = self._get_client()

        chat_history = [
            {"role": msg.role.value, "content": msg.content}
            for msg in request.chat_history
        ]
        user_turn_count = len([m for m in chat_history if m["role"] == "user"])

        initial_state = {
            "query":             request.query,
            "original_query":    request.query,
            "chat_history":      chat_history,
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
            "response_archetype":  "",
            "previous_archetypes": [],
            "turn_count":          user_turn_count,
            "previous_answer":     "",
        }

        # 1단계: generate 직전까지 실행 (검색 파이프라인)
        state = await run_pre_generate(
            client=client,
            user_id=user.get("user_id", ""),
            org_id=request.org_id,
            provider=request.provider,
            model=request.model,
            retrieve_fn=self._retrieve,
            agent_name=f"{self.name}-{self.version}",
            initial_state=initial_state,
        )

        sources = state.get("sources", [])
        logger.debug(f"[stream] run_pre_generate state keys={list(state.keys())} sources_count={len(sources)}")
        if sources:
            yield self._format_sse({"type": "sources", "sources": sources})

        # 2단계: generate를 실제 스트리밍으로 실행
        messages = build_generate_messages(state)
        full_answer = ""
        # 아키타입 태그(<!--archetype:xxx-->)를 스트리밍 중 필터링하기 위한 버퍼
        tag_buffer = ""

        async for sse_line in client.call_completions_stream(
            user_id=user.get("user_id", ""),
            org_id=request.org_id,
            provider=request.provider,
            model=request.model,
            messages=messages,
            prompt_variables=None,
            agent_name=f"{self.name}-{self.version}",
        ):
            # SSE raw line에서 content 추출하여 실시간 전달
            line = sse_line.strip()
            if line.startswith("data: "):
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                    content = ""
                    if "choices" in data:
                        delta = data["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                    elif "content" in data:
                        content = data["content"]
                    if content:
                        full_answer += content
                        # <!--archetype:xxx--> 태그 필터링
                        tag_buffer += content
                        # 태그가 완성되면 제거 후 나머지만 전송
                        if "-->" in tag_buffer:
                            cleaned = re.sub(r'<!--archetype:\w+-->', '', tag_buffer)
                            if cleaned:
                                yield self._format_sse({
                                    "choices": [{"index": 0, "delta": {"content": cleaned}}]
                                })
                            tag_buffer = ""
                        # 태그 시작이 감지되지 않으면 바로 전송
                        elif "<" not in tag_buffer:
                            yield self._format_sse({
                                "choices": [{"index": 0, "delta": {"content": tag_buffer}}]
                            })
                            tag_buffer = ""
                        # "<"가 있지만 아직 "-->"가 안 왔으면 버퍼에 유지
                except (json.JSONDecodeError, KeyError, IndexError):
                    pass

        # 버퍼에 남은 텍스트 전송 (태그가 아닌 경우)
        if tag_buffer:
            cleaned = re.sub(r'<!--archetype:\w+-->', '', tag_buffer)
            if cleaned:
                yield self._format_sse({
                    "choices": [{"index": 0, "delta": {"content": cleaned}}]
                })

        # 아키타입 파싱 및 후처리
        archetype, clean_answer = parse_archetype(full_answer)
        clean_answer = reinforce_ije(clean_answer)
        state["answer"] = clean_answer
        state["response_archetype"] = archetype

        elapsed_seconds = round(time.time() - start_time, 1)

        yield self._format_sse({
            "type": "metadata",
            "metadata": {
                "intent":          state.get("intent", ""),
                "grade":           state.get("grade_decision", ""),
                "retry_count":     state.get("retry_count", 0),
                "validate":        "skip",
                "elapsed_seconds": elapsed_seconds,
            },
        })

        log_meta = self._build_log_metadata(state)
        log_meta["elapsed_seconds"] = elapsed_seconds
        self._save_log(
            db, request, user, response_mode,
            answer=clean_answer,
            sources=sources,
            log_metadata=log_meta,
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
