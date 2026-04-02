"""/app/core/middleware/phoenix.py"""

from threading import Lock
from typing import Any, Dict
from urllib.parse import urljoin

import requests
from core.config import get_setting
from core.log.logging import get_logging
from fastapi import FastAPI, Request
from openinference.instrumentation.langchain import LangChainInstrumentor
from openinference.semconv.resource import ResourceAttributes
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from phoenix.config import get_env_collector_endpoint, get_env_host, get_env_port
from starlette.middleware.base import BaseHTTPMiddleware

logger = get_logging()
settings = get_setting()


class CustomOpenInferenceExporter(OTLPSpanExporter):
    def __init__(self, custom_endpoint: str | None = None) -> None:
        host = get_env_host()
        if host == "0.0.0.0":
            host = "127.0.0.1"

        endpoint = custom_endpoint or get_env_collector_endpoint() or f"http://{host}:{get_env_port()}"
        endpoint = urljoin(endpoint, "/v1/traces")

        super().__init__(endpoint=endpoint)


class CustomLangChainInstrumentor(LangChainInstrumentor):
    def __init__(
        self,
        project_name: str = "default",
        collector_endpoint: str | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        try:
            super().__init__(*args, **kwargs)
            self.project_name = project_name
            self.collector_endpoint = collector_endpoint
        except Exception as e:
            logger.error(f"Failed to initialize CustomLangChainInstrumentor: {e}")
            raise

    def instrument(self) -> None:
        try:
            tracer_provider = trace_sdk.TracerProvider(
                resource=Resource({ResourceAttributes.PROJECT_NAME: self.project_name}),
                span_limits=trace_sdk.SpanLimits(max_attributes=10_000),
            )

            exporter = CustomOpenInferenceExporter(custom_endpoint=self.collector_endpoint)
            tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))

            super(LangChainInstrumentor, self).instrument(skip_dep_check=True, tracer_provider=tracer_provider)

            logger.info(f"Successfully instrumented for project: {self.project_name}")
        except Exception as e:
            logger.error(f"Failed to instrument project {self.project_name}: {e}")
            raise


class PhoenixSafeProjectManager:
    def __init__(self, collector_endpoint: str | None = None):
        self._instrumentors: Dict[str, CustomLangChainInstrumentor] = {}
        self._lock = Lock()
        self.collector_endpoint = collector_endpoint

    def get_instrumentor(self, project_name: str) -> CustomLangChainInstrumentor:
        with self._lock:
            if project_name not in self._instrumentors:
                instrumentor = CustomLangChainInstrumentor(
                    project_name=project_name,
                    collector_endpoint=self.collector_endpoint,
                )
                instrumentor.instrument()
                self._instrumentors[project_name] = instrumentor
            return self._instrumentors[project_name]


class PhoenixMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: FastAPI, collector_endpoint: str = settings.PHOENIX_URI):
        super().__init__(app)
        self.project_manager = PhoenixSafeProjectManager(collector_endpoint=collector_endpoint)

    def _is_phoenix_available(self) -> bool:
        try:
            if not settings.PHOENIX_ENABLED:
                return False

            if not settings.PHOENIX_URI:
                return False

            return True

        except Exception as e:
            logger.warning(f"Failed to check Phoenix availability: {e}")
            return False

    async def dispatch(self, request: Request, call_next):
        if not self._is_phoenix_available():
            return await call_next(request)

        if not settings.APP_NAME:
            return await call_next(request)

        try:
            self.project_manager.get_instrumentor(settings.APP_NAME)
            response = await call_next(request)
            return response
        except Exception as e:
            logger.error(f"Error in Phoenix middleware for {settings.APP_NAME}: {e}")
            return await call_next(request)
