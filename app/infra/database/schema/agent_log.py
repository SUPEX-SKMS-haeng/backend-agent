"""/app/infra/database/schema/agent_log.py"""

import datetime

from zoneinfo import ZoneInfo

KST = ZoneInfo("Asia/Seoul")

from infra.database.base import Base
from sqlalchemy import BIGINT, TEXT, Index, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column


class AgentChatLog(Base):
    """에이전트 실행 로그"""

    __tablename__ = "agent_chat_log"

    id: Mapped[int] = mapped_column(BIGINT, primary_key=True, autoincrement=True)
    trace_id: Mapped[str] = mapped_column(String(64), comment="요청 추적 ID (UUID)")
    session_id: Mapped[str | None] = mapped_column(String(64), nullable=True, comment="대화 세션 ID (멀티턴 그룹핑)")
    user_id: Mapped[str | None] = mapped_column(String(255), nullable=True, comment="사용자 ID")
    org_id: Mapped[int | None] = mapped_column(BIGINT, nullable=True, comment="조직 ID")
    agent_name: Mapped[str] = mapped_column(String(100), comment="에이전트 이름 (rag, summary 등)")
    agent_version: Mapped[str] = mapped_column(String(20), comment="에이전트 버전 (v1, v2 등)")
    response_mode: Mapped[str] = mapped_column(String(20), comment="응답 모드: invoke | stream | post_process")
    query: Mapped[str] = mapped_column(TEXT, comment="사용자 질문 원문")
    answer: Mapped[str | None] = mapped_column(TEXT, nullable=True, comment="에이전트 최종 응답")
    sources: Mapped[dict | None] = mapped_column(JSONB, nullable=True, comment="검색된 문서 출처")
    log_metadata: Mapped[dict | None] = mapped_column(
        JSONB, nullable=True, comment="실행 메타데이터 (route, grade, retry_count, prompts 등)"
    )
    provider: Mapped[str | None] = mapped_column(String(100), nullable=True, comment="LLM 프로바이더")
    model: Mapped[str | None] = mapped_column(String(255), nullable=True, comment="사용 모델명")
    create_dt: Mapped[datetime.datetime] = mapped_column(default=lambda: datetime.datetime.now(KST))

    __table_args__ = (
        Index("ix_agent_chat_log_trace_id", "trace_id"),
        Index("ix_agent_chat_log_session_id", "session_id"),
        Index("ix_agent_chat_log_user_id", "user_id"),
        Index("ix_agent_chat_log_org_id", "org_id"),
    )
