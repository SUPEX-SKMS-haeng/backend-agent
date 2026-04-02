"""/app/main.py"""

from contextlib import asynccontextmanager

import uvicorn
from api.router import api_router
from core.config import get_setting
from core.error.error_handler import set_error_handlers
from core.log.logging import get_logging
from fastapi import FastAPI
from fastapi.routing import APIRoute

settings = get_setting()
logger = get_logging()


def create_app():
    def custom_generate_unique_id(route: APIRoute) -> str:
        return f"{route.tags[0]}-{route.name}"

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info(f"{settings.APP_NAME}[{settings.APP_PORT}] service is initializing...")

        # DB 초기화
        logger.info("  └── Database init...")
        from infra.database.database import get_db
        db = next(get_db())
        db.close()

        # 에이전트 등록 로드
        import service.agent  # noqa: F401

        from service.agent.registry import list_agents
        for agent_info in list_agents():
            logger.info(f"  └── Agent loaded: {agent_info['name']}/{agent_info['version']}")

        # Phoenix 트레이싱 초기화
        if settings.PHOENIX_ENABLED:
            try:
                from core.middleware.phoenix import PhoenixSafeProjectManager

                phoenix_manager = PhoenixSafeProjectManager(collector_endpoint=settings.PHOENIX_URI)
                phoenix_manager.get_instrumentor(settings.APP_NAME)
                logger.info(f"  └── Phoenix instrumented: project={settings.APP_NAME}, endpoint={settings.PHOENIX_URI}")
            except Exception as e:
                logger.error(f"  └── Phoenix 초기화 실패: {e}")

        logger.info(f"{settings.APP_NAME}[{settings.APP_PORT}] service is ready and now running!!")
        yield

    app = FastAPI(
        title=settings.APP_NAME,
        openapi_url=f"{settings.API_V1_STR}/openapi.json",
        lifespan=lifespan,
        generate_unique_id_function=custom_generate_unique_id,
    )
    app.include_router(api_router, prefix=settings.API_V1_STR)

    if settings.PHOENIX_ENABLED:
        try:
            from core.middleware.phoenix import PhoenixMiddleware

            app.add_middleware(PhoenixMiddleware)
            logger.info("Phoenix 추적이 활성화되었습니다.")
        except Exception as e:
            logger.error(f"Phoenix 추적 설정 중 오류 발생: {e}")
    else:
        logger.info("Phoenix 추적이 비활성화되었습니다.")

    set_error_handlers(app)

    return app


app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "__main__:app",
        host="0.0.0.0",
        port=int(settings.APP_PORT),
        workers=int(settings.WORKER),
    )
