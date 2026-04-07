"""/app/main.py"""

import asyncio
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


async def warmup_llm() -> None:
    """앱 시작 시 Azure OpenAI TCP 연결을 미리 수립합니다."""
    try:
        from common.util.llm_gateway_client import LLMGatewayClient
        from service.model.agent import ChatHistory

        client = LLMGatewayClient(llm_gateway_url=settings.LLM_GATEWAY_URL)
        await client.call_completions_non_stream(
            user_id="system",
            org_id=None,
            provider="azure-openai",
            model="gpt-4o",
            messages=[ChatHistory(role="user", content="ping")],
            prompt_variables=None,
            agent_name="warmup",
        )
        logger.info("[warmup] LLM connection pre-warmed successfully")
    except Exception as e:
        logger.warning(f"[warmup] pre-warm failed (non-critical, app continues): {e}")


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

        # Phoenix 트레이싱 초기화 (에이전트별 프로젝트 분리)
        if settings.PHOENIX_ENABLED:
            try:
                from core.middleware.phoenix import init_phoenix_tracing

                init_phoenix_tracing(
                    default_project=settings.APP_NAME,
                    collector_endpoint=settings.PHOENIX_URI,
                )
                logger.info(f"  └── Phoenix instrumented: endpoint={settings.PHOENIX_URI}")
            except Exception as e:
                logger.error(f"  └── Phoenix 초기화 실패: {e}")

        # LLM 커넥션 프리워밍 (비차단, 백그라운드 실행, 참조 보관으로 GC 방지)
        _warmup_task = asyncio.create_task(warmup_llm())  # noqa: F841

        logger.info(f"{settings.APP_NAME}[{settings.APP_PORT}] service is ready and now running!!")
        yield

    app = FastAPI(
        title=settings.APP_NAME,
        openapi_url=f"{settings.API_V1_STR}/openapi.json",
        lifespan=lifespan,
        generate_unique_id_function=custom_generate_unique_id,
    )
    app.include_router(api_router, prefix=settings.API_V1_STR)

    if not settings.PHOENIX_ENABLED:
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
