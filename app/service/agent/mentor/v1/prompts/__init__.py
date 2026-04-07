"""/app/service/agent/mentor/v1/prompts/__init__.py

SK 멘토링 에이전트 프롬프트 모음 — re-export
"""

from service.agent.mentor.v1.prompts.classifier import INTENT_PROMPT
from service.agent.mentor.v1.prompts.generator import (
    DYNAMIC_PROMPT_TEMPLATE,
    GENERATE_PROMPT,
    INTENT_TALK_TYPE_MAP,
    STATIC_GENERATOR_PROMPT,
    TALK_TYPE_GUIDES,
)
from service.agent.mentor.v1.prompts.grader import GRADE_PROMPT
from service.agent.mentor.v1.prompts.rewriter import REWRITE_PROMPT
from service.agent.mentor.v1.prompts.validator import VALIDATE_PROMPT

__all__ = [
    "INTENT_PROMPT",
    "GRADE_PROMPT",
    "REWRITE_PROMPT",
    "GENERATE_PROMPT",
    "STATIC_GENERATOR_PROMPT",
    "DYNAMIC_PROMPT_TEMPLATE",
    "TALK_TYPE_GUIDES",
    "INTENT_TALK_TYPE_MAP",
    "VALIDATE_PROMPT",
]
