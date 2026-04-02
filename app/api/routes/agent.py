"""/app/api/routes/agent.py"""

from api.deps import DatabaseDep
from common.util.search_client import list_indexes, search_documents
from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse
from infra.database.repository import agent_log as agent_log_repo
from service.agent.registry import get_agent, list_agents
from service.model.agent import AgentRequest

router = APIRouter()

# 인증 없이 사용할 기본 사용자 정보
_DEFAULT_USER = {
    "user_id": "anonymous",
    "email": "",
    "username": "anonymous",
    "department": "",
    "company": "",
}


@router.get("/health")
async def health_check():
    return {"success": True, "data": {"status": "ok"}}


@router.get("/agents")
async def get_available_agents():
    """등록된 에이전트 목록 조회"""
    agents = list_agents()
    return {"success": True, "data": agents}


@router.get("/history")
async def get_history(db: DatabaseDep, offset: int = Query(0), limit: int = Query(20)):
    """채팅 히스토리 목록 조회"""
    logs = agent_log_repo.get_by_user_id(db, user_id=_DEFAULT_USER["user_id"], offset=offset, limit=limit)
    return {
        "success": True,
        "data": [
            {
                "id": log.id,
                "traceId": log.trace_id,
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
    """채팅 히스토리 단건 조회"""
    log = agent_log_repo.get_by_trace_id(db, trace_id=trace_id)
    if not log:
        return {"success": False, "error": {"message": "not found"}}
    return {
        "success": True,
        "data": {
            "id": log.id,
            "traceId": log.trace_id,
            "query": log.query,
            "answer": log.answer,
            "agentName": log.agent_name,
            "agentVersion": log.agent_version,
            "sources": log.sources,
            "logMetadata": log.log_metadata,
            "createDt": log.create_dt.isoformat() if log.create_dt else None,
        },
    }


@router.get("/search/indexes")
async def get_indexes():
    """Azure AI Search 인덱스 목록 조회"""
    indexes = await list_indexes()
    return {"success": True, "data": indexes}


@router.get("/search")
async def search(query: str = Query(..., description="검색 쿼리"), top: int = Query(5, description="결과 수")):
    """Azure AI Search 검색 테스트"""
    context, sources = await search_documents(query, top=top)
    return {
        "success": True,
        "data": {
            "context": context,
            "sources": sources,
            "count": len(sources),
        },
    }


@router.post("/invoke")
async def invoke(request: AgentRequest, db: DatabaseDep):
    """비스트리밍 응답"""
    agent = get_agent(request.agent_name, request.version)
    response = await agent.invoke(request, _DEFAULT_USER, db=db, response_mode="invoke")
    return {"success": True, "data": response.model_dump()}


@router.post("/stream")
async def stream(request: AgentRequest, db: DatabaseDep):
    """스트리밍 응답"""
    agent = get_agent(request.agent_name, request.version)
    return StreamingResponse(
        agent.stream(request, _DEFAULT_USER, db=db, response_mode="stream"),
        media_type="text/event-stream",
    )


@router.post("/stream/post-process")
async def post_process_stream(request: AgentRequest, db: DatabaseDep):
    """후처리 후 스트리밍"""
    agent = get_agent(request.agent_name, request.version)
    return StreamingResponse(
        agent.post_process_stream(request, _DEFAULT_USER, db=db, response_mode="post_process"),
        media_type="text/event-stream",
    )
