"""/app/service/agent/registry.py"""

from core.error.service_exception import ErrorCode, ServiceException
from core.log.logging import get_logging
from service.agent.base import BaseAgent

logger = get_logging()

# {agent_name: {version: AgentClass}}
_registry: dict[str, dict[str, type[BaseAgent]]] = {}


def register(name: str, version: str):
    """에이전트 등록 데코레이터"""
    def decorator(cls: type[BaseAgent]):
        cls.name = name
        cls.version = version
        _registry.setdefault(name, {})[version] = cls
        logger.info(f"Agent registered: {name}/{version}")
        return cls
    return decorator


def get_agent(name: str, version: str | None = None) -> BaseAgent:
    """
    등록된 에이전트 인스턴스를 반환

    Args:
        name: 에이전트 이름 (예: "rag")
        version: 버전 (예: "v1"). None이면 최신 버전
    """
    versions = _registry.get(name)
    if not versions:
        raise ServiceException(
            ErrorCode.AGENT_NOT_FOUND,
            detail=f"에이전트 '{name}'을 찾을 수 없습니다",
        )

    if version is None:
        version = sorted(versions.keys())[-1]

    agent_cls = versions.get(version)
    if not agent_cls:
        available = list(versions.keys())
        raise ServiceException(
            ErrorCode.AGENT_NOT_FOUND,
            detail=f"에이전트 '{name}/{version}'을 찾을 수 없습니다. 사용 가능: {available}",
        )

    return agent_cls()


def list_agents() -> list[dict]:
    """등록된 모든 에이전트 목록"""
    result = []
    for name, versions in _registry.items():
        for version, cls in versions.items():
            result.append({
                "name": name,
                "version": version,
                "description": cls.description,
            })
    return result
