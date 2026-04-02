"""/app/common/util/search_client.py

Azure AI Search 클라이언트
"""

import httpx
from core.config import get_setting
from core.log.logging import get_logging

logger = get_logging()
settings = get_setting()


async def search_documents(
    query: str,
    top: int = 5,
    index_name: str | None = None,
) -> tuple[str, list[dict]]:
    """
    Azure AI Search에서 문서를 검색합니다.

    Args:
        query: 검색 쿼리
        top: 반환할 문서 수
        index_name: 인덱스 이름 (None이면 설정값 사용)

    Returns:
        (context, sources) — context는 검색 결과를 합친 문자열, sources는 출처 목록
    """
    endpoint = settings.AZURE_SEARCH_ENDPOINT
    api_key = settings.AZURE_SEARCH_KEY
    index = index_name or settings.AZURE_SEARCH_INDEX_NAME

    if not endpoint or not api_key or not index:
        logger.warning("Azure AI Search 설정이 없습니다. 빈 결과를 반환합니다.")
        return "", []

    url = f"{endpoint}/indexes/{index}/docs/search?api-version=2024-07-01"
    headers = {
        "Content-Type": "application/json",
        "api-key": api_key,
    }
    body = {
        "search": query,
        "top": top,
        "queryType": "simple",
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(url, headers=headers, json=body)
            response.raise_for_status()
            data = response.json()

        results = data.get("value", [])
        if not results:
            logger.info(f"[search] query='{query[:30]}...' → 0 results")
            return "", []

        # context 조합 + sources 추출
        context_parts = []
        sources = []
        for i, doc in enumerate(results):
            content = doc.get("content", doc.get("chunk", ""))
            title = doc.get("title", doc.get("metadata_storage_name", f"문서 {i + 1}"))
            score = doc.get("@search.score", 0)

            if content:
                context_parts.append(f"[{title}]\n{content}")
                sources.append({
                    "title": title,
                    "score": score,
                    "content_preview": content[:200],
                })

        context = "\n\n---\n\n".join(context_parts)
        logger.info(f"[search] query='{query[:30]}...' → {len(sources)} results")
        return context, sources

    except httpx.HTTPStatusError as e:
        logger.error(f"Azure Search HTTP error: {e.response.status_code} - {e.response.text}")
        return "", []
    except Exception as e:
        logger.error(f"Azure Search error: {e}")
        return "", []
