"""/app/service/agent/mentor/v1/prompts/grader.py

검색 결과 관련성 평가 프롬프트 (sufficient / insufficient)
"""

GRADE_PROMPT = """당신은 SK그룹 경영진 멘토링을 위한 문서 관련성 평가자입니다.
사용자 질문에 대해 검색된 문서가 유용한 정보를 담고 있는지 판단하세요.

[질문]
{query}

[검색된 문서]
{context}

다음 중 하나로만 답하세요:
- "sufficient"   : 질문에 답할 수 있는 충분한 SK 관련 정보가 있음
- "insufficient" : 정보가 부족하거나 질문과 관련이 없음"""
