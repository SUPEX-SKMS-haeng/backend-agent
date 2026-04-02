"""/app/api/routes/agent.py"""

from api.deps import DatabaseDep
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
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
