"""/app/api/routes/agent.py"""

from contextlib import contextmanager

from api.deps import CurrentUserDep, DatabaseDep
from common.util.hybrid_search_client import hybrid_search_documents
from common.util.search_client import list_indexes, search_documents
from core.config import get_setting
from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse
from infra.database.repository import agent_log as agent_log_repo
from service.agent.registry import get_agent, list_agents
from service.model.agent import AgentRequest

settings = get_setting()

# Phoenix 트레이싱이 활성화된 경우에만 import
if settings.PHOENIX_ENABLED:
    from core.middleware.phoenix import phoenix_agent_context, phoenix_agent_stream
    from openinference.instrumentation import using_attributes
else:
    # Phoenix 비활성화 시 no-op fallback
    @contextmanager
    def phoenix_agent_context(project_name: str):
        yield

    async def phoenix_agent_stream(project_name, gen):
        async for item in gen:
            yield item

    @contextmanager
    def using_attributes(**kwargs):
        yield

router = APIRouter()


@router.get("/health")
async def health_check():
    return {"success": True, "data": {"status": "ok"}}


@router.get("/agents")
async def get_available_agents():
    """등록된 에이전트 목록 조회"""
    agents = list_agents()
    return {"success": True, "data": agents}


@router.get("/history")
async def get_history(current_user: CurrentUserDep, db: DatabaseDep, offset: int = Query(0), limit: int = Query(20)):
    """채팅 히스토리 목록 조회"""
    logs = agent_log_repo.get_by_user_id(db, user_id=current_user.get("user_id", ""), offset=offset, limit=limit)
    return {
        "success": True,
        "data": [
            {
                "id": log.id,
                "traceId": log.trace_id,
                "sessionId": log.session_id,
                "query": log.query,
                "answer": log.answer,
                "agentName": log.agent_name,
                "agentVersion": log.agent_version,
                "createDt": log.create_dt.isoformat() if log.create_dt else None,
            }
            for log in logs
        ],
    }


@router.get("/history/{trace_id}")
async def get_history_detail(trace_id: str, db: DatabaseDep):
    """채팅 히스토리 단건 조회 — session_id가 있으면 세션 전체 턴 반환"""
    log = agent_log_repo.get_by_trace_id(db, trace_id=trace_id)
    if not log:
        return {"success": False, "error": {"message": "not found"}}

    # session_id가 있으면 해당 세션의 모든 턴을 반환
    turns = []
    if log.session_id:
        session_logs = agent_log_repo.get_by_session_id(db, session_id=log.session_id)
        for s_log in session_logs:
            elapsed = None
            if s_log.log_metadata and isinstance(s_log.log_metadata, dict):
                elapsed = s_log.log_metadata.get("elapsed_seconds")
            raw_sources = s_log.sources
            source_items = raw_sources.get("items") if isinstance(raw_sources, dict) else raw_sources
            turns.append({
                "query": s_log.query,
                "answer": s_log.answer,
                "elapsedSeconds": elapsed,
                "sources": source_items,
            })

    return {
        "success": True,
        "data": {
            "id": log.id,
            "traceId": log.trace_id,
            "sessionId": log.session_id,
            "query": log.query,
            "answer": log.answer,
            "agentName": log.agent_name,
            "agentVersion": log.agent_version,
            "sources": log.sources.get("items") if isinstance(log.sources, dict) else log.sources,
            "logMetadata": log.log_metadata,
            "createDt": log.create_dt.isoformat() if log.create_dt else None,
            "turns": turns if turns else None,
        },
    }


@router.get("/search/indexes")
async def get_indexes():
    """Azure AI Search 인덱스 목록 조회"""
    indexes = await list_indexes()
    return {"success": True, "data": indexes}


@router.get("/search")
async def search(query: str = Query(..., description="검색 쿼리"), top: int = Query(5, description="결과 수")):
    """Azure AI Search 검색 테스트 (기존 키워드 검색)"""
    context, sources = await search_documents(query, top=top)
    return {
        "success": True,
        "data": {
            "context": context,
            "sources": sources,
            "count": len(sources),
        },
    }


@router.get("/search/hybrid")
async def hybrid_search(query: str = Query(..., description="검색 쿼리"), top: int = Query(5, description="결과 수")):
    """하이브리드 검색 테스트 (BM25 + Vector + Weighted RRF)"""
    context, sources = await hybrid_search_documents(query, top=top)
    return {
        "success": True,
        "data": {
            "context": context,
            "sources": sources,
            "count": len(sources),
        },
    }


@router.post("/invoke")
async def invoke(request: AgentRequest, current_user: CurrentUserDep, db: DatabaseDep):
    """비스트리밍 응답"""
    agent = get_agent(request.agent_name, request.version)
    project = f"{agent.name}-{agent.version}"
    user_id = current_user.get("user_id", "")

    with phoenix_agent_context(project), using_attributes(
        session_id=f"{user_id}",
        user_id=user_id,
        metadata={"agent_name": agent.name, "agent_version": agent.version},
        tags=[agent.name, agent.version, project],
    ):
        response = await agent.invoke(request, current_user, db=db, response_mode="invoke")

    return {"success": True, "data": response.model_dump()}


@router.post("/stream")
async def stream(request: AgentRequest, current_user: CurrentUserDep, db: DatabaseDep):
    """스트리밍 응답"""
    agent = get_agent(request.agent_name, request.version)
    project = f"{agent.name}-{agent.version}"
    user_id = current_user.get("user_id", "")

    async def _traced_stream():
        with using_attributes(
            session_id=f"{user_id}",
            user_id=user_id,
            metadata={"agent_name": agent.name, "agent_version": agent.version},
            tags=[agent.name, agent.version, project],
        ):
            async for chunk in phoenix_agent_stream(
                project, agent.stream(request, current_user, db=db, response_mode="stream")
            ):
                yield chunk

    return StreamingResponse(_traced_stream(), media_type="text/event-stream")


@router.post("/stream/post-process")
async def post_process_stream(request: AgentRequest, current_user: CurrentUserDep, db: DatabaseDep):
    """후처리 후 스트리밍"""
    agent = get_agent(request.agent_name, request.version)
    project = f"{agent.name}-{agent.version}"
    user_id = current_user.get("user_id", "")

    async def _traced_stream():
        with using_attributes(
            session_id=f"{user_id}",
            user_id=user_id,
            metadata={"agent_name": agent.name, "agent_version": agent.version},
            tags=[agent.name, agent.version, project],
        ):
            async for chunk in phoenix_agent_stream(
                project,
                agent.post_process_stream(request, current_user, db=db, response_mode="post_process"),
            ):
                yield chunk

    return StreamingResponse(_traced_stream(), media_type="text/event-stream")
