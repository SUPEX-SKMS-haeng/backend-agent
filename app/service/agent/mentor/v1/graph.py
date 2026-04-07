"""/app/service/agent/mentor/v1/graph.py

SK 멘토링 에이전트 — LangGraph 워크플로우

흐름:
  START
    → intent_classify  (LLM: 질문 유형 분류)
    → query_expand     (규칙 기반: 인텐트별 검색 쿼리 확장)
    → retrieve         (문서 검색)
    → grade            (LLM: 검색 결과 평가)
    ┌─ sufficient  → generate   (LLM: 최종 답변 생성)
    │                 → validate (LLM: SKMS 준수 여부 검증)
    │                 ┌─ pass    → END
    │                 └─ fail    → generate (validate_reason 반영 재생성, 1회 한도)
    └─ insufficient → rewrite   (LLM: 쿼리 리라이팅, 최대 2회)
                      → retrieve (재검색 루프)
                      (한도 초과 시 → generate fallback)
"""

import asyncio
import json
import random
import re

from langgraph.graph import END, StateGraph

from common.util.llm_gateway_client import LLMGatewayClient
from core.log.logging import get_logging
from service.agent.mentor.v1.prompts import (
    GENERATE_PROMPT,
    GRADE_PROMPT,
    INTENT_PROMPT,
    REWRITE_PROMPT,
    VALIDATE_PROMPT,
)
from service.agent.mentor.v1.state import MentoringState
from service.model.agent import ChatHistory

logger = get_logging()

MAX_RETRIES = 2    # 재검색 최대 횟수
MAX_VALIDATE = 1   # 검증 실패 후 재생성 최대 횟수

# ── 인텐트별 검색 쿼리 접두어 (규칙 기반, LLM 비용 없음) ─────
_INTENT_SEARCH_PREFIX: dict[str, list[str]] = {
    "strategy": ["전략", "사업 방향"],
    "culture":  ["조직문화", "VWBE 구성원"],
    "crisis":   ["위기", "리스크 관리"],
    "supex":    ["SUPEX 패기", "도전 목표"],
    "hr":       ["인재 육성", "리더십"],
    "general":  [],
}


def build_mentor_graph(
    client: LLMGatewayClient,
    user_id: str,
    org_id: str | None,
    provider: str,
    model: str,
    retrieve_fn,
    agent_name: str = "mentor-v1",
) -> StateGraph:
    """
    SK 멘토링 에이전트 그래프를 빌드합니다.

    Args:
        client      : LLM Gateway 클라이언트
        user_id     : 사용자 ID
        org_id      : 조직 ID
        provider    : LLM 프로바이더
        model       : 모델명
        retrieve_fn : 검색 함수 async def fn(query, metadata) -> (context_str, sources_list)
        agent_name  : LLM Gateway 에이전트 이름 (프롬프트 조회 키)
    """

    async def _call_llm(messages: list[ChatHistory]) -> str:
        """LLM Gateway 비스트리밍 호출 헬퍼 (내부 판단 노드 공통 사용)"""
        result = await client.call_completions_non_stream(
            user_id=user_id,
            org_id=org_id,
            provider=provider,
            model=model,
            messages=messages,
            prompt_variables=None,
            agent_name=agent_name,
        )
        if "choices" in result:
            return result["choices"][0].get("message", {}).get("content", "")
        return result.get("content", "")

    # ── 노드 0: 대화 라우팅 ─────────────────────────────────

    async def conversation_router(state: MentoringState) -> MentoringState:
        """
        새 질문인지 핑퐁 답변인지 판단.
        - chat_history 없음 → new_query
        - 마지막 assistant 발화가 질문으로 끝났고 + 현재 query가 짧으면 → followup
        - 그 외 → new_query
        """
        history = state.get("chat_history", [])
        query = state["query"]

        if not history:
            state["route"] = "new_query"
            return state

        last_assistant = next(
            (m["content"] for m in reversed(history) if m["role"] == "assistant"),
            ""
        )
        is_question_end = last_assistant.rstrip().endswith("?") or last_assistant.rstrip().endswith("까?")

        if is_question_end and len(query.strip()) < 80:
            state["route"] = "followup"
        else:
            state["route"] = "new_query"

        logger.info(f"[conversation_router] route={state['route']} query_len={len(query)}")
        return state

    # ── 노드 1: 인텐트 분류 ─────────────────────────────────

    async def intent_classify(state: MentoringState) -> MentoringState:
        """LLM으로 질문 유형 분류 → intent, intent_confidence"""
        prompt = INTENT_PROMPT.format(query=state["query"])
        messages = [ChatHistory(role="user", content=prompt)]

        raw = await _call_llm(messages)
        try:
            result = json.loads(raw)
            intent = result.get("intent", "general")
            confidence = float(result.get("confidence", 0.5))
        except (json.JSONDecodeError, ValueError):
            # 파싱 실패 시 general로 폴백
            intent = "general"
            confidence = 0.5

        # 유효하지 않은 카테고리 방어
        if intent not in _INTENT_SEARCH_PREFIX:
            intent = "general"

        state["intent"] = intent
        state["intent_confidence"] = confidence
        logger.info(f"[intent_classify] intent={intent} confidence={confidence:.2f}")
        return state

    # ── 노드 2: 쿼리 확장 (규칙 기반, LLM 미사용) ──────────

    async def query_expand(state: MentoringState) -> MentoringState:
        """인텐트별 고정 접두어로 검색 쿼리 확장 (비용 없음)"""
        query = state.get("rewritten_query") or state["original_query"]
        prefixes = _INTENT_SEARCH_PREFIX.get(state["intent"], [])

        search_queries = [query]
        for prefix in prefixes[:2]:
            search_queries.append(f"{prefix} {query}")

        state["search_queries"] = search_queries
        logger.info(f"[query_expand] {len(search_queries)} queries for intent={state['intent']}")
        return state

    # ── 노드 3: 문서 검색 ────────────────────────────────────

    async def retrieve(state: MentoringState) -> MentoringState:
        """확장된 쿼리로 병렬 검색 후 컨텍스트/출처 누산"""
        all_context_parts: list[str] = []
        all_sources: list[dict] = []
        seen_titles: set[str] = set()

        # 모든 쿼리를 병렬로 실행
        results = await asyncio.gather(
            *(retrieve_fn(query, {"intent": state["intent"]})
              for query in state["search_queries"])
        )

        for ctx, srcs in results:
            if ctx:
                all_context_parts.append(ctx)
            for src in srcs:
                title = src.get("title", "")
                if title not in seen_titles:
                    seen_titles.add(title)
                    all_sources.append(src)

        state["context"] = "\n\n---\n\n".join(all_context_parts)
        state["sources"] = all_sources
        logger.info(f"[retrieve] queries={len(state['search_queries'])} sources={len(all_sources)}")
        return state

    # ── 노드 4: 검색 결과 평가 ───────────────────────────────

    async def grade(state: MentoringState) -> MentoringState:
        """검색 컨텍스트가 질문에 답하기 충분한지 규칙 기반으로 판단 (LLM 호출 제거)"""
        context = state.get("context", "")
        sources = state.get("sources", [])

        # 규칙 기반 판단: 소스가 있고 컨텍스트가 최소 길이 이상이면 sufficient
        if len(sources) >= 1 and len(context) >= 100:
            state["grade_decision"] = "sufficient"
        else:
            state["grade_decision"] = "insufficient"

        logger.info(
            f"[grade] → {state['grade_decision']} "
            f"(retry={state['retry_count']}, sources={len(sources)}, ctx_len={len(context)})"
        )
        return state

    # ── 노드 5: 쿼리 리라이팅 ───────────────────────────────

    async def rewrite(state: MentoringState) -> MentoringState:
        """검색 결과 불충분 시 SKMS 키워드 기반으로 쿼리 재작성"""
        prompt = REWRITE_PROMPT.format(query=state["query"])
        messages = [ChatHistory(role="user", content=prompt)]

        rewritten = await _call_llm(messages)
        rewritten = rewritten.strip()

        state["rewritten_query"] = rewritten
        state["retry_count"] += 1
        # 재검색을 위해 이전 결과 초기화
        state["context"] = ""
        state["sources"] = []

        logger.info(
            f"[rewrite] attempt={state['retry_count']} "
            f"'{state['original_query'][:20]}' → '{rewritten[:30]}'"
        )
        return state

    # ── 노드 6: 최종 답변 생성 (PERSONA 주입 대상) ──────────

    def _reinforce_ije(text: str) -> str:
        """'이제' 접속어가 부족하면 자연스러운 위치에 삽입"""
        sentences = text.split('. ')
        ije_count = text.count('이제')
        target_count = max(1, len(sentences) // 4)  # 4문장당 최소 1회

        if ije_count >= target_count:
            return text

        # 삽입 가능한 접속어 패턴
        ije_variants = ['이제 ', '이제, ', '이제 말이야, ']
        transition_words = ['그래서', '그런데', '또', '그리고']

        inserted = 0
        for i in range(3, len(sentences), 4):  # 3번째 문장부터 4문장 간격
            if inserted >= (target_count - ije_count):
                break
            sentence = sentences[i].lstrip()
            # 이미 접속어로 시작하면 교체, 아니면 앞에 추가
            has_transition = any(sentence.startswith(tw) for tw in transition_words)
            if has_transition:
                for tw in transition_words:
                    if sentence.startswith(tw):
                        sentences[i] = sentence.replace(tw, random.choice(ije_variants), 1)
                        inserted += 1
                        break
            elif not sentence.startswith('이제'):
                sentences[i] = random.choice(ije_variants) + sentence
                inserted += 1

        return '. '.join(sentences)

    async def generate(state: MentoringState) -> MentoringState:
        """
        검색 컨텍스트 + 질문으로 멘토링 답변 생성.
        LLM Gateway가 GENERATE_PROMPT 앞에 PERSONA/GUARDRAIL/RAG/FEWSHOT 자동 삽입.
        """
        # 동적 프롬프트 주입: 아키타입 이력 및 턴 카운트
        turn_count = state.get("turn_count", 0)
        previous_archetypes = state.get("previous_archetypes", [])

        # 직전 턴 질문 종결 여부 계산
        prev_answer = state.get("previous_answer", "")
        prev_ends_with_question = "YES" if prev_answer.strip().endswith("?") else "NO"

        dynamic_prompt = GENERATE_PROMPT.format(
            turn_count=turn_count,
            previous_archetypes=previous_archetypes,
            prev_ends_with_question=prev_ends_with_question,
            chat_history="",  # chat_history는 메시지로 별도 주입
            context=state.get("context", ""),
            query=state["original_query"],
        )

        messages = [ChatHistory(role="system", content=dynamic_prompt)]

        # 이전 대화 이력 (최근 6턴)
        for msg in state["chat_history"][-6:]:
            messages.append(ChatHistory(role=msg["role"], content=msg["content"]))

        # 검증 실패 피드백이 있으면 재생성 요청으로 추가
        if state.get("validate_reason"):
            messages.append(ChatHistory(
                role="system",
                content=f"[이전 답변 수정 요청] {state['validate_reason']}",
            ))

        # 참고 자료 + 질문 구성 (LLM Gateway RAG 프롬프트가 이 구조를 처리)
        context = state.get("context", "")
        query = state["original_query"]
        if context:
            user_content = f"## 참고 자료\n{context}\n\n## 질문\n{query}"
        else:
            user_content = f"## 질문\n{query}"

        messages.append(ChatHistory(role="user", content=user_content))

        answer = await _call_llm(messages)

        # 아키타입 태그 파싱 및 제거
        archetype_match = re.search(r'<!--archetype:(\w+)-->', answer)
        current_archetype = archetype_match.group(1) if archetype_match else "standard"
        clean_answer = re.sub(r'<!--archetype:\w+-->', '', answer).strip()

        # "이제" 접속어 후처리 보강
        clean_answer = _reinforce_ije(clean_answer)

        # state 업데이트
        state["answer"] = clean_answer
        state["previous_answer"] = clean_answer
        state["response_archetype"] = current_archetype
        prev = state.get("previous_archetypes", [])
        prev.append(current_archetype)
        state["previous_archetypes"] = prev[-3:]  # 최근 3턴만 유지
        state["turn_count"] = state.get("turn_count", 0) + 1

        logger.info(
            f"[generate] answer_length={len(clean_answer)} "
            f"archetype={current_archetype} turn={state['turn_count']} "
            f"validate_count={state['validate_count']}"
        )
        return state

    # ── 노드 7: 답변 검증 ────────────────────────────────────

    async def validate(state: MentoringState) -> MentoringState:
        """생성된 답변이 SKMS 철학 및 사실 관계에 부합하는지 검증"""
        # 이미 재생성을 1회 수행했으면 무조건 pass 처리 (무한루프 방지)
        if state.get("validate_count", 0) >= MAX_VALIDATE:
            state["validate_result"] = "pass"
            state["validate_reason"] = ""
            logger.info("[validate] max_validate reached → force pass")
            return state

        # 인텐트 분류 신뢰도가 높으면 검증 스킵 (LLM 호출 절약)
        if state.get("intent_confidence", 0) >= 0.8 and state.get("grade_decision") == "sufficient":
            state["validate_result"] = "pass"
            state["validate_reason"] = ""
            logger.info(f"[validate] high confidence ({state['intent_confidence']:.2f}) → skip validation")
            return state

        prompt = VALIDATE_PROMPT.format(
            context=state.get("context", "")[:1000],
            answer=state["answer"],
            response_archetype=state.get("response_archetype", "standard"),
        )
        messages = [ChatHistory(role="user", content=prompt)]

        raw = await _call_llm(messages)
        raw = raw.strip().lower()

        if raw.startswith("fail"):
            reason = raw.split(":", 1)[1].strip() if ":" in raw else "SKMS 철학 위반"
            state["validate_result"] = "fail"
            state["validate_reason"] = reason
        else:
            state["validate_result"] = "pass"
            state["validate_reason"] = ""

        state["validate_count"] = state.get("validate_count", 0) + 1
        logger.info(f"[validate] result={state['validate_result']} count={state['validate_count']}")
        return state

    # ── 조건부 엣지 ──────────────────────────────────────────

    def after_grade(state: MentoringState) -> str:
        if state["grade_decision"] == "sufficient":
            return "generate"
        if state["retry_count"] >= MAX_RETRIES:
            logger.info(f"[grade] max retries ({MAX_RETRIES}) → fallback generate")
            return "generate"
        return "rewrite"

    def after_validate(state: MentoringState) -> str:
        if state["validate_result"] == "pass":
            return END
        return "generate"  # validate_reason 포함 재생성

    # ── 그래프 조립 ──────────────────────────────────────────

    graph = StateGraph(MentoringState)

    graph.add_node("conversation_router", conversation_router)
    graph.add_node("intent_classify", intent_classify)
    graph.add_node("query_expand",    query_expand)
    graph.add_node("retrieve",        retrieve)
    graph.add_node("grade",           grade)
    graph.add_node("rewrite",         rewrite)
    graph.add_node("generate",        generate)
    graph.add_node("validate",        validate)

    graph.set_entry_point("conversation_router")

    graph.add_conditional_edges(
        "conversation_router",
        lambda s: s["route"],
        {
            "new_query": "intent_classify",
            "followup":  "generate",   # RAG 전체 스킵, 이전 context 재사용
        }
    )

    graph.add_edge("intent_classify", "query_expand")
    graph.add_edge("query_expand",    "retrieve")
    graph.add_edge("retrieve",        "grade")

    graph.add_conditional_edges("grade", after_grade, {
        "generate": "generate",
        "rewrite":  "rewrite",
    })

    graph.add_edge("rewrite",  "query_expand")   # 리라이팅 후 쿼리 재확장
    graph.add_edge("generate", "validate")

    graph.add_conditional_edges("validate", after_validate, {
        END:        END,
        "generate": "generate",
    })

    return graph.compile()
