"""/app/infra/database/repository/agent_log.py"""

from infra.database.schema.agent_log import AgentChatLog
from sqlalchemy.orm import Session


def create(
    db: Session,
    *,
    trace_id: str,
    user_id: str | None,
    org_id: int | None,
    agent_name: str,
    agent_version: str,
    response_mode: str,
    query: str,
    answer: str | None = None,
    sources: dict | None = None,
    log_metadata: dict | None = None,
    provider: str | None = None,
    model: str | None = None,
) -> AgentChatLog:
    """에이전트 실행 로그 저장"""
    log = AgentChatLog(
        trace_id=trace_id,
        user_id=user_id,
        org_id=int(org_id) if org_id else None,
        agent_name=agent_name,
        agent_version=agent_version,
        response_mode=response_mode,
        query=query,
        answer=answer,
        sources=sources,
        log_metadata=log_metadata,
        provider=provider,
        model=model,
    )
    db.add(log)
    db.commit()
    db.refresh(log)
    return log


def get_by_trace_id(db: Session, trace_id: str) -> AgentChatLog | None:
    """trace_id로 로그 조회"""
    return db.query(AgentChatLog).filter(AgentChatLog.trace_id == trace_id).first()


def get_by_user_id(
    db: Session,
    user_id: str,
    offset: int = 0,
    limit: int = 20,
) -> list[AgentChatLog]:
    """사용자 ID로 로그 목록 조회"""
    return (
        db.query(AgentChatLog)
        .filter(AgentChatLog.user_id == user_id)
        .order_by(AgentChatLog.create_dt.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
