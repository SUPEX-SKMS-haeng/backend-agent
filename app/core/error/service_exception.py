"""/app/core/error/service_exception.py"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Any

from fastapi import HTTPException, status


@dataclass(frozen=True)
class ErrorInfo:
    """에러 정보를 담는 불변 데이터 클래스"""
    http_status: int
    code: int
    message: str


class ErrorCode(Enum):
    """
    에러 코드 체계:
    - 1XXXX: 클라이언트 오류 (4XX)
    - 9XXXX: 서버 오류 (5XX)
    
    세부 분류:
    - X0XXX: 공통
    - X1XXX: 인증(Auth)
    - X2XXX: 사용자(User)
    - X3XXX: 리소스(Resource)
    - X4XXX: 파일(File)
    - X5XXX: AI/모델
    - X9XXX: 시스템
    """
    
    # ============================================
    # 400 Bad Request - 잘못된 요청
    # ============================================
    BAD_REQUEST = ErrorInfo(status.HTTP_400_BAD_REQUEST, 10000, "잘못된 요청입니다")
    VALIDATION_ERROR = ErrorInfo(status.HTTP_400_BAD_REQUEST, 10001, "입력값 검증에 실패했습니다")
    DUPLICATE_ENTITY = ErrorInfo(status.HTTP_400_BAD_REQUEST, 10002, "이미 존재하는 항목입니다")
    INVALID_PARAMETER = ErrorInfo(status.HTTP_400_BAD_REQUEST, 10003, "잘못된 파라미터입니다")
    
    # File 관련
    UNSUPPORTED_FILE_TYPE = ErrorInfo(status.HTTP_400_BAD_REQUEST, 14001, "지원하지 않는 파일 형식입니다")
    FILE_SIZE_EXCEEDED = ErrorInfo(status.HTTP_400_BAD_REQUEST, 14002, "파일 크기가 초과되었습니다")
    FILE_COUNT_EXCEEDED = ErrorInfo(status.HTTP_400_BAD_REQUEST, 14003, "파일 개수가 초과되었습니다")
    
    # AI/Model 관련 (클라이언트 측 문제)
    INPUT_TOKEN_EXCEEDED = ErrorInfo(status.HTTP_400_BAD_REQUEST, 15001, "입력 토큰 수가 초과되었습니다")
    
    # ============================================
    # 401 Unauthorized - 인증 실패
    # ============================================
    UNAUTHORIZED = ErrorInfo(status.HTTP_401_UNAUTHORIZED, 11000, "인증이 필요합니다")
    TOKEN_MISSING = ErrorInfo(status.HTTP_401_UNAUTHORIZED, 11001, "인증 토큰이 없습니다")
    TOKEN_EXPIRED = ErrorInfo(status.HTTP_401_UNAUTHORIZED, 11002, "인증 토큰이 만료되었습니다")
    TOKEN_INVALID = ErrorInfo(status.HTTP_401_UNAUTHORIZED, 11003, "유효하지 않은 토큰입니다")
    INVALID_CREDENTIALS = ErrorInfo(status.HTTP_401_UNAUTHORIZED, 11004, "아이디 또는 비밀번호가 올바르지 않습니다")
    
    # ============================================
    # 403 Forbidden - 권한 없음
    # ============================================
    FORBIDDEN = ErrorInfo(status.HTTP_403_FORBIDDEN, 13000, "접근 권한이 없습니다")
    ACCESS_DENIED = ErrorInfo(status.HTTP_403_FORBIDDEN, 13001, "접근이 거부되었습니다")
    INSUFFICIENT_PERMISSIONS = ErrorInfo(status.HTTP_403_FORBIDDEN, 13002, "권한이 부족합니다")
    INACTIVE_USER = ErrorInfo(status.HTTP_403_FORBIDDEN, 12001, "비활성화된 사용자입니다")
    
    # ============================================
    # 404 Not Found - 리소스 없음
    # ============================================
    NOT_FOUND = ErrorInfo(status.HTTP_404_NOT_FOUND, 13000, "요청한 리소스를 찾을 수 없습니다")
    USER_NOT_FOUND = ErrorInfo(status.HTTP_404_NOT_FOUND, 12002, "사용자를 찾을 수 없습니다")
    RESOURCE_NOT_FOUND = ErrorInfo(status.HTTP_404_NOT_FOUND, 13001, "리소스를 찾을 수 없습니다")
    AI_RESOURCE_NOT_FOUND = ErrorInfo(status.HTTP_404_NOT_FOUND, 13002, "AI 리소스를 찾을 수 없습니다")
    AI_RESOURCE_ASSIGNMENT_NOT_FOUND = ErrorInfo(status.HTTP_404_NOT_FOUND, 13003, "리소스 할당을 찾을 수 없습니다")
    ORGANIZATION_NOT_FOUND = ErrorInfo(status.HTTP_404_NOT_FOUND, 13004, "조직을 찾을 수 없습니다")
    
    # ============================================
    # 409 Conflict - 충돌
    # ============================================
    CONFLICT = ErrorInfo(status.HTTP_409_CONFLICT, 10900, "요청이 현재 상태와 충돌합니다")
    DUPLICATE_VALUE = ErrorInfo(status.HTTP_409_CONFLICT, 10901, "중복된 값이 존재합니다")
    ALREADY_EXISTS = ErrorInfo(status.HTTP_409_CONFLICT, 12901, "이미 존재하는 사용자입니다")
    DUPLICATE_ORGANIZATION_CODE = ErrorInfo(status.HTTP_409_CONFLICT, 13901, "이미 존재하는 조직 코드입니다")
    
    # ============================================
    # 429 Too Many Requests - 요청 과다
    # ============================================
    RATE_LIMITED = ErrorInfo(status.HTTP_429_TOO_MANY_REQUESTS, 10429, "요청이 너무 많습니다. 잠시 후 다시 시도해주세요")
    
    # ============================================
    # 500 Internal Server Error - 서버 오류
    # ============================================
    INTERNAL_ERROR = ErrorInfo(status.HTTP_500_INTERNAL_SERVER_ERROR, 90000, "서버 내부 오류가 발생했습니다")
    
    # Database
    DATABASE_ERROR = ErrorInfo(status.HTTP_500_INTERNAL_SERVER_ERROR, 99001, "데이터베이스 오류가 발생했습니다")
    DATABASE_CONNECTION_ERROR = ErrorInfo(status.HTTP_500_INTERNAL_SERVER_ERROR, 99002, "데이터베이스 연결에 실패했습니다")
    
    # AI/Model 관련 (서버 측 문제)
    MODEL_ERROR = ErrorInfo(status.HTTP_500_INTERNAL_SERVER_ERROR, 95000, "AI 모델 오류가 발생했습니다")
    MODEL_TIMEOUT = ErrorInfo(status.HTTP_500_INTERNAL_SERVER_ERROR, 95001, "AI 모델 응답 시간이 초과되었습니다")
    MODEL_NOT_FOUND = ErrorInfo(status.HTTP_500_INTERNAL_SERVER_ERROR, 95002, "AI 모델을 찾을 수 없습니다")
    MODEL_RATE_LIMITED = ErrorInfo(status.HTTP_500_INTERNAL_SERVER_ERROR, 95003, "AI 모델 요청 한도를 초과했습니다")
    
    # External Service
    EXTERNAL_SERVICE_ERROR = ErrorInfo(status.HTTP_500_INTERNAL_SERVER_ERROR, 98000, "외부 서비스 오류가 발생했습니다")
    CONNECTION_ERROR = ErrorInfo(status.HTTP_500_INTERNAL_SERVER_ERROR, 98001, "연결 오류가 발생했습니다")
    
    # File Export
    FILE_EXPORT_FAILED = ErrorInfo(status.HTTP_500_INTERNAL_SERVER_ERROR, 94001, "파일 내보내기에 실패했습니다")

    @property
    def http_status(self) -> int:
        return self.value.http_status
    
    @property
    def code(self) -> int:
        return self.value.code
    
    @property
    def message(self) -> str:
        return self.value.message


class ServiceException(HTTPException):
    """
    서비스 예외 클래스
    
    사용 예시:
        # 기본 사용 - 기본 메시지 사용
        raise ServiceException(ErrorCode.USER_NOT_FOUND)
        
        # 커스텀 메시지
        raise ServiceException(ErrorCode.USER_NOT_FOUND, detail="사용자 ID: abc123을 찾을 수 없습니다")
        
        # 추가 데이터 포함
        raise ServiceException(ErrorCode.VALIDATION_ERROR, detail="이메일 형식이 올바르지 않습니다", data={"field": "email"})
    """
    
    def __init__(
        self,
        error_code: ErrorCode,
        detail: Optional[str] = None,
        data: Optional[dict[str, Any]] = None,
        headers: Optional[dict[str, str]] = None,
    ):
        self.error_code = error_code
        self.error_data = data
        
        # detail이 없으면 기본 메시지 사용
        message = detail if detail else error_code.message
        
        super().__init__(
            status_code=error_code.http_status,
            detail=message,
            headers=headers,
        )

    def to_dict(self) -> dict[str, Any]:
        """에러 응답을 딕셔너리로 변환"""
        response = {
            "success": False,
            "error": {
                "code": self.error_code.code,
                "name": self.error_code.name,
                "message": self.detail,
            }
        }
        if self.error_data:
            response["error"]["data"] = self.error_data
        return response
