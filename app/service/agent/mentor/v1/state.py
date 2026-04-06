"""/app/service/agent/mentor/v1/state.py

SK 멘토링 에이전트 그래프 상태 정의
"""

from typing import TypedDict


class MentoringState(TypedDict):
    """멘토링 에이전트 그래프 전체 상태"""

    # ── 입력 ─────────────────────────────────────────────────
    query: str
    original_query: str
    chat_history: list[dict]

    # ── 라우팅 ───────────────────────────────────────────────
    route: str             # "new_query" | "followup"

    # ── 노드 1: intent_classify ──────────────────────────────
    intent: str            # strategy | culture | crisis | supex | hr | general
    intent_confidence: float

    # ── 노드 2: query_expand ─────────────────────────────────
    search_queries: list[str]

    # ── 노드 3: retrieve ─────────────────────────────────────
    context: str           # 검색 결과 합산 컨텍스트 문자열
    sources: list[dict]    # 출처 목록

    # ── 노드 4: grade ────────────────────────────────────────
    grade_decision: str    # "sufficient" | "insufficient"

    # ── 노드 5: rewrite ──────────────────────────────────────
    rewritten_query: str
    retry_count: int

    # ── 노드 6: generate ─────────────────────────────────────
    answer: str

    # ── 노드 7: validate ─────────────────────────────────────
    validate_result: str   # "pass" | "fail"
    validate_reason: str
    validate_count: int

    # ── 응답 아키타입 관리 ────────────────────────────────────
    response_archetype: str          # 직전 턴에 사용한 응답 아키타입 (초기값: "")
    previous_archetypes: list[str]   # 최근 3턴의 아키타입 이력 (초기값: [])
    turn_count: int                  # 현재 대화 턴 수 (초기값: 0)
    previous_answer: str             # 직전 턴의 답변 원문 (초기값: "")
