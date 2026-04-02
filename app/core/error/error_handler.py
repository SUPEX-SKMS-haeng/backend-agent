"""/app/core/error/error_handler.py"""

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from core.config import get_setting
from core.error.service_exception import ErrorCode, ServiceException
from core.log.logging import get_logging

settings = get_setting()
logger = get_logging()


def set_error_handlers(app: FastAPI):
    """FastAPI 앱에 에러 핸들러 등록"""
    
    @app.exception_handler(ServiceException)
    async def service_exception_handler(request: Request, exc: ServiceException):
        """ServiceException 핸들러 - 정의된 비즈니스 예외"""
        logger.error(
            f"ServiceException: {exc.error_code.name} | "
            f"path={request.url.path} | "
            f"detail={exc.detail}"
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.to_dict(),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Pydantic 검증 에러 핸들러"""
        errors = exc.errors()
        logger.warning(f"ValidationError: path={request.url.path} | errors={errors}")
        
        # 첫 번째 에러의 상세 정보 추출
        first_error = errors[0] if errors else {}
        field = ".".join(str(loc) for loc in first_error.get("loc", []))
        
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "success": False,
                "error": {
                    "code": ErrorCode.VALIDATION_ERROR.code,
                    "name": ErrorCode.VALIDATION_ERROR.name,
                    "message": f"입력값 검증 실패: {field}",
                    "data": {"details": errors},
                }
            },
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        """일반 HTTP 예외 핸들러"""
        logger.error(
            f"HTTPException: status={exc.status_code} | "
            f"path={request.url.path} | "
            f"detail={exc.detail}"
        )
        
        # 상태 코드별 기본 에러 코드 매핑
        error_code_map = {
            400: ErrorCode.BAD_REQUEST,
            401: ErrorCode.UNAUTHORIZED,
            403: ErrorCode.FORBIDDEN,
            404: ErrorCode.NOT_FOUND,
            409: ErrorCode.CONFLICT,
            429: ErrorCode.RATE_LIMITED,
        }
        
        error_code = error_code_map.get(exc.status_code, ErrorCode.INTERNAL_ERROR)
        
        # 1XX, 204, 304는 body 없이 응답
        if exc.status_code < 200 or exc.status_code in (204, 304):
            return JSONResponse(status_code=exc.status_code, content=None)
        
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "success": False,
                "error": {
                    "code": error_code.code,
                    "name": error_code.name,
                    "message": exc.detail or error_code.message,
                }
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        """처리되지 않은 예외 핸들러"""
        logger.exception(
            f"UnhandledException: type={type(exc).__name__} | "
            f"path={request.url.path} | "
            f"message={str(exc)}"
        )
        
        # 프로덕션에서는 상세 에러 숨김
        is_debug = settings.ENVIRONMENT in ("local", "dev", "development")
        message = (
            str(exc) if is_debug 
            else ErrorCode.INTERNAL_ERROR.message
        )
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "error": {
                    "code": ErrorCode.INTERNAL_ERROR.code,
                    "name": ErrorCode.INTERNAL_ERROR.name,
                    "message": message,
                }
            },
        )
