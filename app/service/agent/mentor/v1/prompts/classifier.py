"""/app/service/agent/mentor/v1/prompts/classifier.py

인텐트 분류 프롬프트 (strategy / culture / crisis / supex / hr / general)
"""

INTENT_PROMPT = """당신은 SK그룹 경영진의 질문을 분류하는 분류기입니다.
아래 카테고리 중 가장 적합한 하나를 선택하고 JSON으로만 응답하세요.

카테고리:
- strategy : 중장기 전략, 사업 포트폴리오, 신사업, M&A, 글로벌 전략
- culture  : 조직문화, VWBE, 구성원 행복, 팀워크, 구성원 육성
- crisis   : 위기관리, 리스크 대응, 재무 위기, 대외 이슈
- supex    : SUPEX 추구, 패기, 도전 목표 설정, 실행 방법론
- hr       : 인재 채용, 평가, 보상, 승진, 리더십 개발
- general  : 위 카테고리에 해당하지 않는 일반 경영 멘토링

질문: {query}
응답 형식: {{"intent": "<카테고리>", "confidence": <0.0~1.0>}}"""
