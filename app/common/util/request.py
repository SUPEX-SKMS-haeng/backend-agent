"""/app/common/util/request.py"""

import base64

import httpx
from core.config import get_setting
from core.log.logging import get_logging

logger = get_logging()
settings = get_setting()

BASE_SERVICE_URI = settings.BASE_SERVICE_URI


def _get_system_auth_header() -> dict:
    """
    시스템 인증 헤더 생성 (MASTER_KEY를 base64 인코딩)
    """
    encoded_key = base64.b64encode(settings.MASTER_KEY.encode()).decode()
    return {"SYSTEM-AUTH": encoded_key}


def get_ai_resource_info():
    """
    AI 리소스 정보 조회
    """
    url = f"{BASE_SERVICE_URI}/api/v1/ai-resources"
    response = httpx.get(url)
    return response.json()


def resolve_resource_for_context(
    resource_type: str,
    org_id: int | None = None,
    scope_type: str | None = None,
    scope_value: str | None = None,
) -> dict:
    """
    컨텍스트 기반 최적 AI 리소스 조회

    우선순위: scope 매칭 > org 매칭 > default

    Args:
        resource_type: 리소스 타입 (llm, embedding, ai-search, di)
        org_id: 조직 ID (선택)
        scope_type: 스코프 타입 (선택) - user, project, environment 등
        scope_value: 스코프 값 (선택)

    Returns:
        AiResource 정보 (endpoint, access_key, model_deployment_name 등)
    """
    url = f"{BASE_SERVICE_URI}/api/v1/ai-resources/assignments/resolve"
    headers = _get_system_auth_header()

    payload = {
        "resource_type": resource_type,
        "org_id": org_id,
        "scope_type": scope_type,
        "scope_value": scope_value,
    }

    response = httpx.post(url, json=payload, headers=headers)
    response.raise_for_status()
    return response.json()  # type: ignore[no-any-return]
