"""/app/infra/database/repository/agent_log.py"""

from infra.database.schema.agent_log import AgentChatLog
from sqlalchemy.orm import Session


def create(
    db: Session,
    *,
    trace_id: str,
    session_id: str | None = None,
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
        session_id=session_id,
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


def get_by_session_id(db: Session, session_id: str) -> list[AgentChatLog]:
    """session_id로 세션 내 모든 턴 조회 (시간순)"""
    return (
        db.query(AgentChatLog)
        .filter(AgentChatLog.session_id == session_id)
        .order_by(AgentChatLog.create_dt.asc())
        .all()
    )


def get_by_user_id(
    db: Session,
    user_id: str,
    offset: int = 0,
    limit: int = 20,
) -> list[AgentChatLog]:
    """사용자 ID로 로그 목록 조회 (세션별 첫 턴만)"""
    from sqlalchemy import func

    # session_id가 있으면 세션별 가장 오래된 로그만, 없으면 그대로 반환
    subquery = (
        db.query(func.min(AgentChatLog.id).label("min_id"))
        .filter(AgentChatLog.user_id == user_id, AgentChatLog.session_id.isnot(None))
        .group_by(AgentChatLog.session_id)
        .subquery()
    )

    session_logs = (
        db.query(AgentChatLog)
        .filter(AgentChatLog.id.in_(db.query(subquery.c.min_id)))
    )

    no_session_logs = (
        db.query(AgentChatLog)
        .filter(AgentChatLog.user_id == user_id, AgentChatLog.session_id.is_(None))
    )

    return (
        session_logs.union(no_session_logs)
        .order_by(AgentChatLog.create_dt.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
