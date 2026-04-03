"""/app/core/middleware/phoenix.py

Phoenix(Arize) 트레이싱 — 에이전트별 프로젝트 분리 지원

- contextvars 로 요청별 에이전트 프로젝트명을 전달
- AgentProjectSpanProcessor 가 on_end 에서 span 의 Resource.PROJECT_NAME 을 오버라이드
- Phoenix /projects 페이지에 에이전트별 별도 프로젝트로 표시됨
  (예: rag-v1, rag-v2, mentor-v1)
"""

import contextvars
from contextlib import contextmanager
from typing import AsyncGenerator
from urllib.parse import urljoin

from core.config import get_setting
from core.log.logging import get_logging
from openinference.instrumentation.langchain import LangChainInstrumentor
from openinference.semconv.resource import ResourceAttributes
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

logger = get_logging()
settings = get_setting()

# ── 에이전트별 프로젝트 컨텍스트 ──

_current_agent_project = contextvars.ContextVar("phoenix_agent_project", default="agent")


@contextmanager
def phoenix_agent_context(project_name: str):
    """동기 코드에서 에이전트 프로젝트 컨텍스트 설정"""
    token = _current_agent_project.set(project_name)
    try:
        yield
    finally:
        _current_agent_project.reset(token)


async def phoenix_agent_stream(
    project_name: str, gen: AsyncGenerator
) -> AsyncGenerator:
    """비동기 스트림 제너레이터를 에이전트 프로젝트 컨텍스트로 래핑"""
    token = _current_agent_project.set(project_name)
    try:
        async for item in gen:
            yield item
    finally:
        _current_agent_project.reset(token)


# ── 커스텀 SpanProcessor: Resource.PROJECT_NAME 을 에이전트별로 오버라이드 ──

_PROJECT_ATTR = "openinference.project.name"


class AgentProjectSpanProcessor(SimpleSpanProcessor):
    """span 생성 시 contextvars 에서 프로젝트명을 기록하고,
    span 종료 시 Resource 의 PROJECT_NAME 을 오버라이드하여
    Phoenix 가 에이전트별 프로젝트로 분류하게 한다.

    Phoenix 는 Resource 의 openinference.project.name 으로 프로젝트를 구분하므로
    span attribute 만 바꿔서는 프로젝트가 분리되지 않는다.
    on_end 에서 span._resource 를 직접 교체하는 방식으로 해결한다.
    """

    def on_start(self, span, parent_context=None) -> None:
        project = _current_agent_project.get()
        span.set_attribute(_PROJECT_ATTR, project)

    def on_end(self, span) -> None:
        project = (span.attributes or {}).get(_PROJECT_ATTR)
        if project:
            span._resource = span.resource.merge(
                Resource({ResourceAttributes.PROJECT_NAME: project})
            )
        super().on_end(span)


# ── 초기화 ──


def init_phoenix_tracing(
    default_project: str = "agent",
    collector_endpoint: str | None = None,
) -> None:
    """Phoenix 트레이싱을 1회 초기화한다.

    - LangChain 전역 계측 활성화
    - AgentProjectSpanProcessor 로 에이전트별 프로젝트 분리
    """
    endpoint = collector_endpoint or settings.PHOENIX_URI
    traces_endpoint = urljoin(endpoint, "/v1/traces")

    tracer_provider = trace_sdk.TracerProvider(
        resource=Resource({ResourceAttributes.PROJECT_NAME: default_project}),
        span_limits=trace_sdk.SpanLimits(max_attributes=10_000),
    )

    exporter = OTLPSpanExporter(endpoint=traces_endpoint)
    tracer_provider.add_span_processor(AgentProjectSpanProcessor(exporter))

    LangChainInstrumentor().instrument(
        skip_dep_check=True,
        tracer_provider=tracer_provider,
    )

    logger.info(
        f"Phoenix tracing initialized: default_project={default_project}, endpoint={endpoint}"
    )
