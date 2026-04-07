"""/app/core/config.py"""

from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_file": "../.env", "env_file_encoding": "utf-8", "extra": "ignore"}

    # Application
    APP_NAME: str = "agent"
    APP_PORT: str = "8006"
    WORKER: int = 1
    API_V1_STR: str = "/api/v1"
    ENVIRONMENT: str = "local"
    DATA_PATH: str = "/data"
    LOGIN_TYPE: str = "basic"

    MASTER_KEY: str = ""
    DUPLICATE_AUTH: bool = False
    BASE_SERVICE_URI: str = "http://localhost:8002"
    LLM_GATEWAY_URL: str = "http://localhost:8080"

    # Logging
    LOG_LEVEL: str = "INFO"
    HTTP_LOG_LEVEL: str = "WARNING"
    LOGGING_ENABLED: bool = True

    # Redis
    REDIS_HOST: str = "127.0.0.1"
    REDIS_PORT: int = 6379
    REDIS_DATABASE: int = 0
    REDIS_ACCESS_KEY: str = ""
    REDIS_USE_SSL: bool = False

    # Azure AI Search
    AZURE_SEARCH_ENDPOINT: str = ""
    AZURE_SEARCH_KEY: str = ""
    AZURE_SEARCH_INDEX_NAME: str = ""

    # Hybrid Search (BM25 + Vector + Weighted RRF)
    # TODO: 데이터 전량 적재 완료 후 Recall@K, MRR 기반으로 튜닝 필요
    HYBRID_SEARCH_VECTOR_FIELD: str = "content_vector"
    HYBRID_SEARCH_TOP_N: int = 10
    HYBRID_SEARCH_K_NEAREST: int = 50
    HYBRID_SEARCH_VECTOR_WEIGHT: float = 0.5
    HYBRID_SEARCH_KEYWORD_WEIGHT: float = 0.5
    HYBRID_SEARCH_RRF_K: int = 60

    # Arize Phoenix
    PHOENIX_ENABLED: bool = False
    PHOENIX_URI: str = "http://localhost:6006"

    # Database
    DB_ENGINE: str = "postgresql"
    DB_USER: str = "user"
    DB_PASSWORD: str = ""
    DB_HOST: str = "127.0.0.1"
    DB_NAME: str = "dev"
    DB_PORT: int = 5432
    DB_POOL_SIZE: int = 50
    DB_POOL_MAX_OVERFLOW: int = 30
    DB_POOL_RECYCLE: int = 14400
    DB_POOL_TIMEOUT: int = 10


@lru_cache()
def get_setting():
    return Settings()
