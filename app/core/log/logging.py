"""/app/core/log/logging.py"""

import os
import sys
import yaml
import time
import logging.config
from functools import lru_cache, wraps
from contextlib import contextmanager
from contextvars import ContextVar

from core.config import get_setting

settings = get_setting()

# Context variables for log information
history_id_context: ContextVar[str] = ContextVar("history_id", default="")
agent_context: ContextVar[str] = ContextVar("agent", default="")
node_context: ContextVar[str] = ContextVar("node", default="")

log_dir = settings.DATA_PATH + "/logs/" + settings.APP_NAME
if not os.path.exists(log_dir):
    os.makedirs(log_dir, exist_ok=True)

# Load the config file
logging_file = os.path.join(os.path.dirname(__file__), "logging_config.yaml")
with open(logging_file, "rt", encoding="utf-8") as f:
    config = yaml.safe_load(f.read())
    config["handlers"]["file"]["filename"] = log_dir + f"/{settings.APP_NAME}.log"


class SimpleExtendedFormatter(logging.Formatter):
    """
    표준 Formatter를 확장하여 로그 레코드에 history_id, agent, node 필드가 존재하도록 보장합니다. 
    'extra'를 통해 제공되지 않은 경우 빈 문자열로 기본값이 설정됩니다.
    또한 일관된 너비를 위해 'name_level' 필드를 생성하고, 필드가 지정된 최대 길이를 초과하면 잘라냅니다.
    """

    def format(self, record):
        max_name_level_len = 10  # Corresponds to %(name_level)-10s in YAML
        max_agent_len = 10       # Corresponds to %(agent)-10s in YAML
        max_node_len = 15        # Corresponds to %(node)-15s in YAML
        max_history_id_len = 6   # Corresponds to %(history_id)-6s in YAML

        # 레코드에서 값을 가져오고, 컨텍스트 변수가 없으면 빈 문자열로 기본값을 설정합니다.
        history_id_val = getattr(record, "history_id", "") or history_id_context.get()
        agent_val = getattr(record, "agent", "") or agent_context.get()
        node_val = getattr(record, "node", "") or node_context.get()
        name_level_val = f"{record.name}({record.levelname})"

        record.history_id = str(history_id_val)[:max_history_id_len]
        record.agent = agent_val[:max_agent_len]
        record.node = node_val[:max_node_len]
        record.name_level = name_level_val[:max_name_level_len]

        return super().format(record)


@lru_cache()
def get_logging():
    if "simple" in config.get("formatters", {}):
        config["formatters"]["simple"]["()"] = SimpleExtendedFormatter
    
    # Configure the logging module with the config file
    logging.config.dictConfig(config)
    
    if not settings.LOGGING_ENABLED:
        logging.disable(sys.maxsize)
    else:
        logger_levels = getattr(settings, "LOGGER_LEVELS", {})
        loggers_config = {
            "sqlalchemy.engine": settings.LOG_LEVEL,
            "httpx": settings.LOG_LEVEL,
            "httpcore": settings.HTTP_LOG_LEVEL,
            "urllib3": settings.LOG_LEVEL,
            "openai": settings.LOG_LEVEL,
            "azure": "WARNING",
            "NumExpr": "WARNING",
        }
        
        for logger_name, default_level in loggers_config.items():
            level = logger_levels.get(logger_name, default_level)
            if level == "DISABLED":
                logging.getLogger(logger_name).disabled = True
            else:
                logging.getLogger(logger_name).setLevel(level)
    
    # Get a logger object
    logger = logging.getLogger(settings.APP_NAME)
    logger.setLevel(settings.LOG_LEVEL)
    
    if settings.LOGGING_ENABLED:
        logger.info("Application logging initialized")

    return logger


logger = get_logging()


def log_method(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.info(f"[{func.__name__}] Starting...")
        result = func(*args, **kwargs)
        logger.info(f"[{func.__name__}] Finished.")
        return result

    return wrapper


@contextmanager
def time_measure(
    message, agent="", node="", history_id="", deployment="", batch_size="1", requested_model=""
):
    """
    실행 시간을 측정하고 로깅하는 컨텍스트 매니저

    Args:
        message: 로그 메시지
        agent: 에이전트 이름
        node: 노드 이름
        history_id: 히스토리 ID
        deployment: 모델 deployment 이름 (선택적)
        batch_size: 배치 사이즈 (선택적)
        requested_model: 요청한 모델명 (선택적)
    """
    start_time = time.time()
    start_datetime = time.strftime("%m-%d %H:%M:%S", time.localtime(start_time))

    # Set context variables
    history_id_token = history_id_context.set(history_id)
    agent_token = agent_context.set(agent)
    node_token = node_context.set(node)

    extra = {
        "agent": agent,
        "node": node,
        "history_id": history_id,
    }

    # [LLM] 메시지인 경우 deployment와 batch_count 정보 추가
    if message.startswith("[LLM]"):
        message = f"{message} (배치: {batch_size})"
        extra["batch_count"] = batch_size
        if deployment:
            if requested_model:
                # 요청한 모델과 실제 deployment 비교
                message = f"[{deployment}] {message} (요청: {requested_model})"
            else:
                message = f"[{deployment}] {message}"

    try:
        yield
    finally:
        end_time = time.time()
        execution_time = end_time - start_time
        extra["execution_time"] = f"{execution_time:.3f}s"

        logger.info(
            f"{message} - 시작: {start_datetime}, 소요시간: {execution_time:.3f}초",
            extra=extra,
        )

        # Reset context variables
        history_id_context.reset(history_id_token)
        agent_context.reset(agent_token)
        node_context.reset(node_token)


def get_deployment_name(model_instance):
    """
    모델 인스턴스에서 deployment 이름을 추출하는 헬퍼 함수

    Args:
        model_instance: LangChain 모델 인스턴스

    Returns:
        str: deployment 이름 또는 빈 문자열
    """
    if not model_instance:
        return ""
    
    for attr in ["model", "model_name", "engine", "deployment_name", "azure_deployment", "deployment"]:
        if hasattr(model_instance, attr):
            value = getattr(model_instance, attr)
            if value:
                return str(value)
    return ""


def set_log_context(history_id="", agent="", node=""):
    """
    로그 컨텍스트를 설정하는 유틸리티 함수

    Args:
        history_id: 히스토리 ID
        agent: 에이전트 이름
        node: 노드 이름

    Returns:
        tuple: context variable tokens
    """
    history_id_token = history_id_context.set(history_id)
    agent_token = agent_context.set(agent)
    node_token = node_context.set(node)
    return history_id_token, agent_token, node_token


def reset_log_context(tokens):
    """
    로그 컨텍스트를 해제하는 유틸리티 함수

    Args:
        tokens: set_log_context()에서 반환된 토큰들
    """
    history_id_token, agent_token, node_token = tokens
    history_id_context.reset(history_id_token)
    agent_context.reset(agent_token)
    node_context.reset(node_token)


@contextmanager
def log_context(history_id="", agent="", node=""):
    """
    로그 컨텍스트를 설정하는 컨텍스트 매니저

    Args:
        history_id: 히스토리 ID
        agent: 에이전트 이름
        node: 노드 이름
    """
    tokens = set_log_context(history_id, agent, node)
    try:
        yield
    finally:
        reset_log_context(tokens)