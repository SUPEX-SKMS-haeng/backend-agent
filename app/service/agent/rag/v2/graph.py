"""/app/service/agent/rag/v2/graph.py

LangGraph 기반 Agentic RAG 워크플로우

흐름:
  START → route → [retrieve → grade → (rewrite → retrieve)* → generate] or [direct_generate] → END

  route:            검색이 필요한지 LLM으로 판단
  retrieve:         문서 검색 (mock)
  grade:            검색 결과 품질 평가
  rewrite:          쿼리 리라이팅 (검색 결과 부족 시)
  generate:         검색 결과 기반 최종 응답 생성
  direct_generate:  검색 없이 바로 응답
"""

from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from common.util.llm_gateway_client import LLMGatewayClient
from core.log.logging import get_logging
from service.agent.rag.v2.prompts import (
    DIRECT_PROMPT,
    GENERATE_PROMPT,
    GRADE_PROMPT,
    REWRITE_PROMPT,
    ROUTE_PROMPT,
)
from service.model.agent import ChatHistory

logger = get_logging()

MAX_RETRIES = 2


class AgentState(TypedDict):
    """그래프 상태"""
    query: str
    original_query: str
    chat_history: list[dict]
    context: str
    sources: list[dict]
    answer: str
    route_decision: str  # "retrieve" or "direct"
    grade_decision: str  # "sufficient" or "insufficient"
    retry_count: int


def build_rag_graph(
    client: LLMGatewayClient,
    user_id: str,
    org_id: str | None,
    provider: str,
    model: str,
    retrieve_fn: Any,
) -> StateGraph:
    """
    Agentic RAG 그래프를 빌드합니다.

    Args:
        client: LLM Gateway 클라이언트
        user_id: 사용자 ID
        org_id: 조직 ID
        provider: LLM 프로바이더
        model: 모델명
        retrieve_fn: 검색 함수 (async def fn(query, metadata) -> (context, sources))
    """

    async def _call_llm(messages: list[ChatHistory]) -> str:
        """LLM Gateway 비스트리밍 호출 헬퍼"""
        result = await client.call_completions_non_stream(
            user_id=user_id,
            org_id=org_id,
            provider=provider,
            model=model,
            messages=messages,
            prompt_variables=None,
            agent_name="rag-v2",
        )
        if "choices" in result:
            return result["choices"][0].get("message", {}).get("content", "")
        return result.get("content", "")

    # ── 노드 정의 ──

    async def route(state: AgentState) -> AgentState:
        """검색 필요 여부 판단"""
        query = state["query"]
        prompt = ROUTE_PROMPT.format(query=query)
        messages = [ChatHistory(role="user", content=prompt)]

        decision = await _call_llm(messages)
        decision = decision.strip().lower()

        if "retrieve" in decision:
            state["route_decision"] = "retrieve"
        else:
            state["route_decision"] = "direct"

        logger.info(f"[route] query='{query[:30]}...' → {state['route_decision']}")
        return state

    async def retrieve(state: AgentState) -> AgentState:
        """문서 검색"""
        query = state["query"]
        context, sources = await retrieve_fn(query, None)
        state["context"] = context
        state["sources"] = sources
        logger.info(f"[retrieve] query='{query[:30]}...' → {len(sources)} sources")
        return state

    async def grade(state: AgentState) -> AgentState:
        """검색 결과 품질 평가"""
        prompt = GRADE_PROMPT.format(query=state["query"], context=state["context"])
        messages = [ChatHistory(role="user", content=prompt)]

        decision = await _call_llm(messages)
        decision = decision.strip().lower()

        if "sufficient" in decision:
            state["grade_decision"] = "sufficient"
        else:
            state["grade_decision"] = "insufficient"

        logger.info(f"[grade] → {state['grade_decision']} (retry: {state['retry_count']})")
        return state

    async def rewrite(state: AgentState) -> AgentState:
        """쿼리 리라이팅"""
        prompt = REWRITE_PROMPT.format(query=state["query"])
        messages = [ChatHistory(role="user", content=prompt)]

        rewritten = await _call_llm(messages)
        state["query"] = rewritten.strip()
        state["retry_count"] += 1
        logger.info(f"[rewrite] '{state['original_query'][:20]}...' → '{state['query'][:30]}...'")
        return state

    async def generate(state: AgentState) -> AgentState:
        """검색 결과 기반 최종 응답 생성"""
        messages = [ChatHistory(role="system", content=GENERATE_PROMPT)]

        for msg in state["chat_history"]:
            messages.append(ChatHistory(role=msg["role"], content=msg["content"]))

        context = state.get("context", "")
        query = state["original_query"]
        if context:
            user_content = f"[참고 문서]\n{context}\n\n[질문]\n{query}"
        else:
            user_content = query

        messages.append(ChatHistory(role="user", content=user_content))

        state["answer"] = await _call_llm(messages)
        logger.info(f"[generate] answer length={len(state['answer'])}")
        return state

    async def direct_generate(state: AgentState) -> AgentState:
        """검색 없이 바로 응답"""
        messages = [ChatHistory(role="system", content=DIRECT_PROMPT)]

        for msg in state["chat_history"]:
            messages.append(ChatHistory(role=msg["role"], content=msg["content"]))

        messages.append(ChatHistory(role="user", content=state["query"]))

        state["answer"] = await _call_llm(messages)
        logger.info(f"[direct_generate] answer length={len(state['answer'])}")
        return state

    # ── 조건부 엣지 ──

    def after_route(state: AgentState) -> str:
        return "retrieve" if state["route_decision"] == "retrieve" else "direct_generate"

    def after_grade(state: AgentState) -> str:
        if state["grade_decision"] == "sufficient":
            return "generate"
        if state["retry_count"] >= MAX_RETRIES:
            logger.info(f"[grade] max retries reached ({MAX_RETRIES}), proceeding to generate")
            return "generate"
        return "rewrite"

    # ── 그래프 조립 ──

    graph = StateGraph(AgentState)

    graph.add_node("route", route)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade", grade)
    graph.add_node("rewrite", rewrite)
    graph.add_node("generate", generate)
    graph.add_node("direct_generate", direct_generate)

    graph.set_entry_point("route")

    graph.add_conditional_edges("route", after_route, {
        "retrieve": "retrieve",
        "direct_generate": "direct_generate",
    })

    graph.add_edge("retrieve", "grade")

    graph.add_conditional_edges("grade", after_grade, {
        "generate": "generate",
        "rewrite": "rewrite",
    })

    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("generate", END)
    graph.add_edge("direct_generate", END)

    return graph.compile()
