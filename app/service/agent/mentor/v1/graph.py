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
    DYNAMIC_PROMPT_TEMPLATE,
    GENERATE_PROMPT,
    GRADE_PROMPT,
    INTENT_PROMPT,
    INTENT_TALK_TYPE_MAP,
    REWRITE_PROMPT,
    STATIC_GENERATOR_PROMPT,
    TALK_TYPE_GUIDES,
    VALIDATE_PROMPT,
)
from service.agent.mentor.v1.state import MentoringState
from service.model.agent import ChatHistory

logger = get_logging()

MAX_RETRIES = 2    # 재검색 최대 횟수
MAX_VALIDATE = 1   # 검증 실패 후 재생성 최대 횟수

# ── Grade 임계값 상수 (모듈 레벨) ─────────────────────────────
# RRF raw score 임계값: 양쪽 검색 모두 상위권에 등장해야 도달하는 수준
# k=60, 양쪽 rank 1일 때 max = 0.5/61 + 0.5/61 ≈ 0.01639
# 한쪽만 rank 1일 때 = 0.5/61 ≈ 0.008197
# 임계값 0.005 = 한쪽 검색에서 최소 상위 40등 이내
# 데이터 기반 튜닝 결과 (2026-04-08):
# - 관련 질문 top-1 rrf_raw: 0.008~0.016, vec: 0.597~0.724
# - 무관 질문 top-1 rrf_raw: 0.014~0.016, vec: 0.546~0.701
# - RRF는 k=60 압축으로 구분력 낮음, 극단적 실패만 차단
# - Vector score도 임베딩 특성상 완벽 구분 불가, 키워드 관련성과 조합 필요
GRADE_RRF_RAW_THRESHOLD: float = 0.008   # BM25 단독 rank 1 수준
GRADE_VECTOR_SCORE_THRESHOLD: float = 0.55  # 무관 질문 최솟값(0.546) 바로 위

# ── 규칙 기반 수치 검증 (모듈 레벨, validate에서 사용) ─────────
_NUMBER_PATTERN = re.compile(
    r'(?:\d[\d,]*\.?\d*)\s*(?:조|억|만|원|%|퍼센트|명|개월|년|개|건|배)',
)


def _extract_numbers(text: str) -> set[str]:
    """텍스트에서 단위 포함 수치 표현을 추출"""
    return set(_NUMBER_PATTERN.findall(text))


def _rule_based_number_check(answer: str, context: str) -> tuple[bool, str]:
    """
    답변의 구체적 수치가 컨텍스트에 근거가 있는지 규칙 기반 체크.
    Returns: (pass여부, 사유)
    """
    answer_numbers = _extract_numbers(answer)
    if not answer_numbers:
        return True, ""

    # 답변의 수치 중 컨텍스트에 없는 것 찾기
    unsupported = []
    for num in answer_numbers:
        # 숫자 부분만 추출
        num_digits = re.sub(r'[^\d.]', '', num)
        if num_digits and len(num_digits) >= 2:  # 1자리 숫자는 무시
            # 정확히 같은 표현이 있거나, 단어 경계로 숫자가 컨텍스트에 존재하면 OK
            if num not in context and not re.search(
                r'(?<!\d)' + re.escape(num_digits) + r'(?!\d)', context
            ):
                unsupported.append(num)

    if len(unsupported) >= 3:
        return False, f"참고자료에 없는 수치 다수 포함: {', '.join(list(unsupported)[:5])}"
    return True, ""


def reinforce_ije(text: str) -> str:
    """'이제' 접속어가 부족하면 자연스러운 위치에 삽입"""
    sentences = text.split('. ')
    ije_count = text.count('이제')
    target_count = max(1, len(sentences) // 4)  # 4문장당 최소 1회

    if ije_count >= target_count:
        return text

    ije_variants = ['이제 ', '이제, ', '이제 말이야, ']
    transition_words = ['그래서', '그런데', '또', '그리고']

    inserted = 0
    for i in range(3, len(sentences), 4):
        if inserted >= (target_count - ije_count):
            break
        sentence = sentences[i].lstrip()
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

    # ── 키워드 기반 인텐트 분류 (LLM 호출 제거) ──────────────
    _INTENT_KEYWORDS: dict[str, list[str]] = {
        "strategy": [
            "전략", "사업 포트폴리오", "신사업", "M&A", "글로벌", "중장기",
            "사업 방향", "투자", "포트폴리오", "성장 동력", "비전",
        ],
        "culture": [
            "조직문화", "VWBE", "구성원 행복", "팀워크", "구성원 육성",
            "사기", "분위기", "소통", "협업", "행복",
        ],
        "crisis": [
            "위기", "리스크", "재무 위기", "대외 이슈", "위기관리",
            "리스크 대응", "위험", "불확실",
        ],
        "supex": [
            "SUPEX", "수펙스", "패기", "도전 목표", "실행 방법론",
            "목표 설정", "도전", "초일류",
        ],
        "hr": [
            "인재", "채용", "평가", "보상", "승진", "리더십 개발",
            "인사", "교육", "역량", "성과 관리",
        ],
        "off_topic": [
            "nft", "블록체인", "bitcoin", "비트코인", "암호화폐", "가상화폐",
            "이더리움", "코인", "채굴", "요리", "레시피", "점심", "저녁",
            "게임", "스포츠", "날씨", "주식 종목", "주가", "맛집",
        ],
    }

    async def intent_classify(state: MentoringState) -> MentoringState:
        """키워드 매칭으로 질문 유형 분류 (LLM 미사용, 즉시 반환)"""
        query = state["query"].lower()

        best_intent = "general"
        best_score = 0

        for intent, keywords in _INTENT_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw.lower() in query)
            if score > best_score:
                best_score = score
                best_intent = intent

        # "패기" 컨텍스트 분기: 도전/목표 맥락이면 supex, 사기/분위기면 culture
        if "패기" in query:
            if any(w in query for w in ["도전", "목표", "SUPEX", "수펙스", "실행"]):
                best_intent = "supex"
            elif any(w in query for w in ["사기", "분위기", "문화", "조직"]):
                best_intent = "culture"

        confidence = min(1.0, best_score * 0.3 + 0.4) if best_score > 0 else 0.3

        state["intent"] = best_intent
        state["intent_confidence"] = confidence
        logger.info(f"[intent_classify] intent={best_intent} confidence={confidence:.2f} (keyword-based)")
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

    async def retrieve(state: MentoringState) -> dict:
        """확장된 쿼리로 병렬 검색 후 컨텍스트/출처 누산 — partial dict 반환"""
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

        merged_context = "\n\n---\n\n".join(all_context_parts)
        logger.info(f"[retrieve] queries={len(state['search_queries'])} sources={len(all_sources)}")
        return {"context": merged_context, "sources": all_sources}

    # ── 노드 4: 검색 결과 평가 ───────────────────────────────

    async def grade(state: MentoringState) -> MentoringState:
        """검색 컨텍스트가 질문에 답하기 충분한지 다층 관련성 기반으로 판단"""
        context = state.get("context", "")
        sources = state.get("sources", [])
        query = state.get("original_query", state.get("query", ""))

        # 0단계: off_topic 인텐트 즉시 차단
        if state.get("intent") == "off_topic":
            state["grade_decision"] = "insufficient"
            logger.info("[grade] off_topic 질문으로 분류되어 즉시 insufficient")
            return state

        # 1단계: 기본 필터 (소스 없거나 너무 짧으면 바로 insufficient)
        if len(sources) < 1 or len(context) < 100:
            state["grade_decision"] = "insufficient"
            logger.info(
                f"[grade] → insufficient (basic filter) "
                f"(retry={state['retry_count']}, sources={len(sources)}, ctx_len={len(context)})"
            )
            return state

        # 2단계: RRF 절대 점수 체크 (LLM 호출 없음)
        top_rrf_raw = sources[0].get("rrf_score_raw", 0) if sources else 0
        top_vector_score = sources[0].get("vector_score") if sources else None

        rrf_pass = top_rrf_raw >= GRADE_RRF_RAW_THRESHOLD
        vector_pass = (
            top_vector_score is not None
            and top_vector_score >= GRADE_VECTOR_SCORE_THRESHOLD
        )

        if not rrf_pass and not vector_pass:
            state["grade_decision"] = "insufficient"
            logger.info(
                f"[grade] → insufficient (absolute score) "
                f"(retry={state['retry_count']}, rrf_raw={top_rrf_raw:.6f}, "
                f"vector_score={top_vector_score}, "
                f"thresholds: rrf={GRADE_RRF_RAW_THRESHOLD}, vector={GRADE_VECTOR_SCORE_THRESHOLD})"
            )
            return state

        # 3단계: absolute_relevance 절대 신뢰도 체크
        # 모든 소스의 absolute_relevance가 0.4 미만이면 insufficient
        # 단, 소스 자체가 0개면 이 단계 스킵 (폴백 유지)
        if sources and all(s.get("absolute_relevance", 0.0) < 0.4 for s in sources):
            state["grade_decision"] = "insufficient"
            logger.info("[grade] 3단계 실패: 모든 소스의 absolute_relevance < 0.4")
            return state

        # 4단계: 질문-컨텍스트 키워드 관련성 평가
        _STOPWORDS = {
            "입니다", "합니다", "있습니다", "했습니다", "하면", "때는", "어떻게",
            "무엇", "왜", "어떤", "이걸", "그걸", "저는", "우리", "자네",
            "회장님", "선대회장님", "하고", "되고", "잡으", "내라고",
        }
        query_tokens = set(query.replace("?", "").replace(".", "").split())
        query_keywords = {t for t in query_tokens if len(t) >= 2 and t not in _STOPWORDS}

        context_lower = context.lower()
        matched = sum(1 for kw in query_keywords if kw.lower() in context_lower)
        relevance = matched / max(len(query_keywords), 1)

        # 최소 관련성 30% 이상이고 소스 2개 이상이어야 sufficient
        if relevance >= 0.3 and len(sources) >= 2:
            state["grade_decision"] = "sufficient"
        elif relevance >= 0.5:
            state["grade_decision"] = "sufficient"
        else:
            state["grade_decision"] = "insufficient"

        logger.info(
            f"[grade] → {state['grade_decision']} "
            f"(retry={state['retry_count']}, sources={len(sources)}, "
            f"ctx_len={len(context)}, relevance={relevance:.2f}, "
            f"query_kw={len(query_keywords)}, matched={matched}, "
            f"rrf_raw={top_rrf_raw:.6f}, vector_score={top_vector_score})"
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

    async def generate(state: MentoringState) -> MentoringState:
        """
        검색 컨텍스트 + 질문으로 멘토링 답변 생성.
        LLM Gateway가 STATIC_GENERATOR_PROMPT 앞에 PERSONA/GUARDRAIL/RAG/FEWSHOT 자동 삽입.
        messages[0]은 순수 정적 문자열 → Azure OpenAI auto prefix caching 대상.
        """
        # 동적 변수 준비
        turn_count = state.get("turn_count", 0)
        previous_archetypes = state.get("previous_archetypes", [])
        prev_answer = state.get("previous_answer", "")
        prev_ends_with_question = "YES" if prev_answer.strip().endswith("?") else "NO"
        context = state.get("context", "")
        query = state["original_query"]

        # messages[0]: 정적 시스템 프롬프트 (캐시 대상, f-string 없음)
        messages = [ChatHistory(role="system", content=STATIC_GENERATOR_PROMPT)]

        # 이전 대화 이력 (최근 6턴)
        for msg in state["chat_history"][-6:]:
            messages.append(ChatHistory(role=msg["role"], content=msg["content"]))

        # 검증 실패 피드백이 있으면 재생성 요청으로 추가
        if state.get("validate_reason"):
            messages.append(ChatHistory(
                role="system",
                content=f"[이전 답변 수정 요청] {state['validate_reason']}",
            ))

        # 인텐트 기반 토크타입 가이드 선택
        intent = state.get("intent", "general")
        talk_type_key = INTENT_TALK_TYPE_MAP.get(intent, "mindset")
        talk_type_guide = TALK_TYPE_GUIDES.get(talk_type_key, "")

        # 동적 사용자 메시지: 세션 정보 + 토크타입 가이드 + 컨텍스트 + 질문
        dynamic_content = DYNAMIC_PROMPT_TEMPLATE.format(
            turn_count=turn_count,
            previous_archetypes=previous_archetypes,
            prev_ends_with_question=prev_ends_with_question,
            talk_type_guide=talk_type_guide,
            context=context,
            query=query,
        )
        messages.append(ChatHistory(role="user", content=dynamic_content))

        # generate는 캐시 로깅을 위해 직접 client 호출
        result = await client.call_completions_non_stream(
            user_id=user_id, org_id=org_id,
            provider=provider, model=model,
            messages=messages, prompt_variables=None,
            agent_name=agent_name,
        )
        if "choices" in result:
            answer = result["choices"][0].get("message", {}).get("content", "")
        else:
            answer = result.get("content", "")

        # 캐시 히트 로깅 (Azure OpenAI auto prefix caching)
        try:
            usage = result.get("usage", {})
            cached = (
                usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
                if isinstance(usage, dict) else 0
            )
            logger.info(f"[generate] cached_tokens={cached}, total_input={usage.get('prompt_tokens', 'N/A')}")
        except Exception:
            pass

        # 아키타입 태그 파싱 및 제거
        archetype_match = re.search(r'<!--archetype:(\w+)-->', answer)
        current_archetype = archetype_match.group(1) if archetype_match else "standard"
        clean_answer = re.sub(r'<!--archetype:\w+-->', '', answer).strip()

        # "이제" 접속어 후처리 보강
        clean_answer = reinforce_ije(clean_answer)

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
        # 1순위: followup 경로는 validate 무조건 스킵
        if state.get("route") == "followup":
            state["validate_result"] = "pass"
            state["validate_reason"] = ""
            logger.info("[validate] followup route → skip validation")
            return state

        # 이미 재생성을 1회 수행했으면 무조건 pass 처리 (무한루프 방지)
        if state.get("validate_count", 0) >= MAX_VALIDATE:
            state["validate_result"] = "pass"
            state["validate_reason"] = ""
            logger.info("[validate] max_validate reached → force pass")
            return state

        # ── 사전 검증: 규칙 기반 수치 체크 (LLM 호출 전, 비용 0) ──
        answer = state["answer"]
        context = state.get("context", "")
        number_pass, number_reason = _rule_based_number_check(answer, context)

        if not number_pass:
            state["validate_result"] = "fail"
            state["validate_reason"] = number_reason
            state["validate_count"] = state.get("validate_count", 0) + 1
            logger.info(
                f"[validate] FAIL (rule-based number check) "
                f"reason={number_reason} count={state['validate_count']}"
            )
            return state

        # ── LLM 기반 검증 ──
        prompt = VALIDATE_PROMPT.format(
            context=context[:1000],
            answer=answer,
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


def build_pre_generate_graph(
    client: LLMGatewayClient,
    user_id: str,
    org_id: str | None,
    provider: str,
    model: str,
    retrieve_fn,
    agent_name: str = "mentor-v1",
) -> StateGraph:
    """
    generate/validate를 제외한 검색 파이프라인 그래프.
    스트리밍 모드에서 검색까지만 실행 후, generate를 실제 스트리밍으로 처리하기 위해 사용.
    """
    # build_mentor_graph와 동일한 노드를 재생성 (클로저 필요)
    full_graph_compiled = build_mentor_graph(
        client=client, user_id=user_id, org_id=org_id,
        provider=provider, model=model,
        retrieve_fn=retrieve_fn, agent_name=agent_name,
    )
    # 대신 간단히 파이프라인 함수로 구현
    return full_graph_compiled  # 아래 run_pre_generate에서 직접 처리


async def run_pre_generate(
    client: LLMGatewayClient,
    user_id: str,
    org_id: str | None,
    provider: str,
    model: str,
    retrieve_fn,
    agent_name: str,
    initial_state: dict,
) -> dict:
    """generate 직전까지의 파이프라인을 실행하고 state를 반환."""
    from service.agent.mentor.v1.state import MentoringState as MS

    state = dict(initial_state)

    async def _call_llm(messages):
        result = await client.call_completions_non_stream(
            user_id=user_id, org_id=org_id,
            provider=provider, model=model,
            messages=messages, prompt_variables=None,
            agent_name=agent_name,
        )
        if "choices" in result:
            return result["choices"][0].get("message", {}).get("content", "")
        return result.get("content", "")

    # 1. conversation_router
    history = state.get("chat_history", [])
    query = state["query"]
    if not history:
        state["route"] = "new_query"
    else:
        last_assistant = next(
            (m["content"] for m in reversed(history) if m["role"] == "assistant"), ""
        )
        is_q = last_assistant.rstrip().endswith("?") or last_assistant.rstrip().endswith("까?")
        state["route"] = "followup" if is_q and len(query.strip()) < 80 else "new_query"

    if state["route"] == "followup":
        # followup: 인텐트/쿼리확장 생략, RAG 검색만 1회 수행하여 sources 확보
        ctx, srcs = await retrieve_fn(query, {"intent": state.get("intent", "general")})
        state["context"] = ctx or ""
        state["sources"] = srcs or []
        state["grade_decision"] = "sufficient" if srcs else "insufficient"
        logger.info(f"[run_pre_generate:followup] sources={len(srcs)} ctx_len={len(state['context'])}")
        return state

    # 2. intent_classify (키워드 기반)
    _INTENT_KEYWORDS = {
        "strategy": ["전략", "사업 포트폴리오", "신사업", "M&A", "글로벌", "중장기", "사업 방향", "투자", "포트폴리오", "성장 동력", "비전"],
        "culture": ["조직문화", "VWBE", "구성원 행복", "팀워크", "구성원 육성", "사기", "분위기", "소통", "협업", "행복"],
        "crisis": ["위기", "리스크", "재무 위기", "대외 이슈", "위기관리", "리스크 대응", "위험", "불확실"],
        "supex": ["SUPEX", "수펙스", "패기", "도전 목표", "실행 방법론", "목표 설정", "도전", "초일류"],
        "hr": ["인재", "채용", "평가", "보상", "승진", "리더십 개발", "인사", "교육", "역량", "성과 관리"],
        "off_topic": ["nft", "블록체인", "bitcoin", "비트코인", "암호화폐", "가상화폐", "이더리움", "코인", "채굴", "요리", "레시피", "점심", "저녁", "게임", "스포츠", "날씨", "주식 종목", "주가", "맛집"],
    }
    q_lower = query.lower()
    best_intent, best_score = "general", 0
    for intent, keywords in _INTENT_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw.lower() in q_lower)
        if score > best_score:
            best_score = score
            best_intent = intent
    if "패기" in query:
        if any(w in query for w in ["도전", "목표", "SUPEX", "수펙스", "실행"]):
            best_intent = "supex"
        elif any(w in query for w in ["사기", "분위기", "문화", "조직"]):
            best_intent = "culture"
    state["intent"] = best_intent
    state["intent_confidence"] = min(1.0, best_score * 0.3 + 0.4) if best_score > 0 else 0.3

    # 3. query_expand
    q = state.get("rewritten_query") or state["original_query"]
    prefixes = _INTENT_SEARCH_PREFIX.get(state["intent"], [])
    state["search_queries"] = [q] + [f"{p} {q}" for p in prefixes[:2]]

    # 4. retrieve (병렬) + rewrite 루프
    for attempt in range(MAX_RETRIES + 1):
        all_context_parts, all_sources, seen_titles = [], [], set()
        results = await asyncio.gather(
            *(retrieve_fn(sq, {"intent": state["intent"]}) for sq in state["search_queries"])
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
        logger.debug(f"[run_pre_generate] retrieve attempt={attempt} sources={len(all_sources)} ctx_len={len(state['context'])}")

        # 5. grade (다층 관련성 기반)
        # 5a. RRF 절대 점수 + 벡터 유사도 체크
        top_rrf_raw = all_sources[0].get("rrf_score_raw", 0) if all_sources else 0
        top_vec_score = all_sources[0].get("vector_score") if all_sources else None
        rrf_ok = top_rrf_raw >= GRADE_RRF_RAW_THRESHOLD
        vec_ok = top_vec_score is not None and top_vec_score >= GRADE_VECTOR_SCORE_THRESHOLD

        # off_topic 인텐트 즉시 차단
        if state.get("intent") == "off_topic":
            is_sufficient = False
            rel = 0.0
            logger.info("[run_pre_generate:grade] off_topic 질문으로 분류되어 insufficient")
        elif not rrf_ok and not vec_ok:
            is_sufficient = False
            rel = 0.0
            logger.info(
                f"[run_pre_generate:grade] insufficient (absolute score) "
                f"rrf_raw={top_rrf_raw:.6f}, vector_score={top_vec_score}"
            )
        elif all_sources and all(s.get("absolute_relevance", 0.0) < 0.4 for s in all_sources):
            is_sufficient = False
            rel = 0.0
            logger.info("[run_pre_generate:grade] insufficient (모든 소스 absolute_relevance < 0.4)")
        else:
            # 5b. 키워드 관련성 평가
            _STOPWORDS_PRE = {
                "입니다", "합니다", "있습니다", "했습니다", "하면", "때는", "어떻게",
                "무엇", "왜", "어떤", "이걸", "그걸", "저는", "우리", "자네",
                "회장님", "선대회장님", "하고", "되고", "잡으", "내라고",
            }
            q_tokens = set(query.replace("?", "").replace(".", "").split())
            q_kws = {t for t in q_tokens if len(t) >= 2 and t not in _STOPWORDS_PRE}
            ctx_lower = state["context"].lower()
            matched_kw = sum(1 for kw in q_kws if kw.lower() in ctx_lower)
            rel = matched_kw / max(len(q_kws), 1)

            is_sufficient = (
                len(all_sources) >= 1
                and len(state["context"]) >= 100
                and ((rel >= 0.3 and len(all_sources) >= 2) or rel >= 0.5)
            )
        if is_sufficient:
            state["grade_decision"] = "sufficient"
            logger.info(f"[run_pre_generate:grade] sufficient (relevance={rel:.2f}, sources={len(all_sources)})")
            break
        if attempt >= MAX_RETRIES:
            state["grade_decision"] = "insufficient"
            logger.info(f"[run_pre_generate:grade] insufficient after max retries (relevance={rel:.2f})")
            break

        # rewrite
        from service.model.agent import ChatHistory as CH
        rewrite_prompt = REWRITE_PROMPT.format(query=state["query"])
        rewritten = await _call_llm([CH(role="user", content=rewrite_prompt)])
        state["rewritten_query"] = rewritten.strip()
        state["retry_count"] += 1
        q = state["rewritten_query"]
        state["search_queries"] = [q] + [f"{p} {q}" for p in prefixes[:2]]

    logger.info(f"[run_pre_generate] intent={state['intent']} sources={len(state.get('sources', []))} retries={state['retry_count']}")
    logger.debug(f"[run_pre_generate] returning state keys={list(state.keys())} sources_count={len(state.get('sources', []))}")
    return state


def build_generate_messages(state: dict) -> list:
    """generate 노드용 메시지 리스트를 구성 (스트리밍에서 재사용).
    messages[0]은 순수 정적 문자열 → Azure OpenAI auto prefix caching 대상.
    """
    from service.model.agent import ChatHistory as CH

    turn_count = state.get("turn_count", 0)
    previous_archetypes = state.get("previous_archetypes", [])
    prev_answer = state.get("previous_answer", "")
    prev_ends_with_question = "YES" if prev_answer.strip().endswith("?") else "NO"
    context = state.get("context", "")
    query = state["original_query"]

    # messages[0]: 정적 시스템 프롬프트 (캐시 대상, f-string 없음)
    messages = [CH(role="system", content=STATIC_GENERATOR_PROMPT)]

    for msg in state["chat_history"][-6:]:
        messages.append(CH(role=msg["role"], content=msg["content"]))

    if state.get("validate_reason"):
        messages.append(CH(
            role="system",
            content=f"[이전 답변 수정 요청] {state['validate_reason']}",
        ))

    # 인텐트 기반 토크타입 가이드 선택
    intent = state.get("intent", "general")
    talk_type_key = INTENT_TALK_TYPE_MAP.get(intent, "mindset")
    talk_type_guide = TALK_TYPE_GUIDES.get(talk_type_key, "")

    # 동적 사용자 메시지: 세션 정보 + 토크타입 가이드 + 컨텍스트 + 질문
    dynamic_content = DYNAMIC_PROMPT_TEMPLATE.format(
        turn_count=turn_count,
        previous_archetypes=previous_archetypes,
        prev_ends_with_question=prev_ends_with_question,
        talk_type_guide=talk_type_guide,
        context=context,
        query=query,
    )
    messages.append(CH(role="user", content=dynamic_content))

    return messages


def parse_archetype(answer: str) -> tuple[str, str]:
    """아키타입 태그 파싱 및 제거. (archetype, clean_answer) 반환"""
    archetype_match = re.search(r'<!--archetype:(\w+)-->', answer)
    current_archetype = archetype_match.group(1) if archetype_match else "standard"
    clean_answer = re.sub(r'<!--archetype:\w+-->', '', answer).strip()
    return current_archetype, clean_answer
