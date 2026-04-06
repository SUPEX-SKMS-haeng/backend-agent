"""/app/service/agent/mentor/v1/prompts/rewriter.py

재검색을 위한 쿼리 리라이팅 프롬프트
"""

REWRITE_PROMPT = """당신은 SK그룹 문서 검색 쿼리 최적화기입니다.
원래 질문으로 충분한 검색 결과를 얻지 못했습니다.
SKMS, SUPEX, 최종현 선대회장 철학 등 SK 고유 키워드를 활용하여 더 나은 검색 쿼리를 작성하세요.

원래 질문: {query}
개선된 검색 쿼리:"""
