"""/app/core/security.py"""

import base64
import time
from urllib.parse import unquote

import jwt
from core.config import get_setting
from core.log.logging import get_logging
from fastapi import Depends, Security
from fastapi.security import APIKeyHeader
from service.model.user import UserOrganizationRole

settings = get_setting()
logger = get_logging()


username_header = APIKeyHeader(name="username", scheme_name="username", auto_error=False)
company_header = APIKeyHeader(name="company", scheme_name="company", auto_error=False)
department_header = APIKeyHeader(name="department", scheme_name="department", auto_error=False)
user_id_header = APIKeyHeader(name="user_id", scheme_name="user_id", auto_error=False)
email_header = APIKeyHeader(name="email", scheme_name="email", auto_error=False)
role_header = APIKeyHeader(name="role", scheme_name="role", auto_error=False)


def get_current_user(
    username=Depends(username_header),
    company=Depends(company_header),
    department=Depends(department_header),
    user_id=Depends(user_id_header),
    email=Depends(email_header),
    role=Depends(role_header),
):
    try:
        return {
            "user_id": user_id,
            "email": email,
            "department": department,
            "username": unquote(username) if username else None,
            "company": company,
            "role": UserOrganizationRole.model_validate_json(unquote(role)) if role else UserOrganizationRole(),
        }
    except Exception as e:
        logger.error(f"Get current user error: {e}")
        raise e


def verify_master_key(token=Security(APIKeyHeader(name="SYSTEM-AUTH"))):
    try:
        key = settings.MASTER_KEY

        decoded_bytes = base64.b64decode(token)
        decoded_token = decoded_bytes.decode("utf-8")

        return key == decoded_token
    except Exception as e:
        raise e


def create_system_token() -> str:
    """
    시스템 간 통신을 위한 JWT 토큰을 생성합니다.
    MASTER_KEY를 사용하여 인증 토큰을 생성합니다.

    Returns:
        str: JWT 토큰
    """
    iat = time.time()
    exp = iat + 60 * 60  # 1시간 유효
    payload = {
        "exp": exp,
        "iat": iat,
        "oid": "system",
        "name": "System Service",
        "company": "internal",
        "role": ["system"],
    }
    key = settings.MASTER_KEY
    encoded_jwt = jwt.encode(payload, key, algorithm="HS256")

    return encoded_jwt  # type: ignore[no-any-return]
