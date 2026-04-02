"""/app/service/model/agent.py"""

from enum import Enum

from pydantic import BaseModel


class ChatRole(str, Enum):
    user = "user"
    ai = "ai"
    assistant = "assistant"
    system = "system"


class ChatHistory(BaseModel):
    role: ChatRole
    content: str


class AgentRequest(BaseModel):
    """에이전트 호출 요청"""
    query: str
    chat_history: list[ChatHistory] = []
    agent_name: str = "rag"
    version: str = "v1"
    provider: str = "azure-openai"
    model: str = "gpt-4.1-mini"
    org_id: str | None = "1"
    metadata: dict | None = None


class AgentResponse(BaseModel):
    """에이전트 비스트리밍 응답"""
    answer: str
    sources: list[dict] = []
    metadata: dict = {}
