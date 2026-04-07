"""/app/service/model/agent.py"""

from enum import Enum

from pydantic import BaseModel, model_validator


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
    provider: str = ""
    model: str = ""
    org_id: str | int | None = None
    session_id: str | None = None
    metadata: dict | None = None

    @model_validator(mode="after")
    def set_defaults(self):
        """빈 값이 오면 기본값으로 채움"""
        if not self.provider:
            self.provider = "azure-openai"
        if not self.model:
            self.model = "gpt-4.1-mini"
        if not self.org_id or self.org_id == 0:
            self.org_id = "1"
        else:
            self.org_id = str(self.org_id)
        return self


class AgentResponse(BaseModel):
    """에이전트 비스트리밍 응답"""
    answer: str
    sources: list[dict] = []
    metadata: dict = {}
