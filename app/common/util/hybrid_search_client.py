"""/app/common/util/hybrid_search_client.py

하이브리드 검색 클라이언트 — BM25(키워드) + Vector(시맨틱) + Weighted RRF

기존 search_client.py의 simple 키워드 검색을 래핑하여
BM25 + Vector 병렬 검색 → Weighted RRF 병합으로 검색 품질 향상.

사용법:
    from common.util.hybrid_search_client import hybrid_search_documents
    context, sources = await hybrid_search_documents(query, index_name="sk_books")
"""

import asyncio
import hashlib
import time

from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError
from azure.search.documents.aio import SearchClient as AzureSearchClient
from azure.search.documents.models import VectorizableTextQuery

from core.config import get_setting
from core.log.logging import get_logging

logger = get_logging()


class HybridSearchClient:
    """
    BM25 + Vector 하이브리드 검색 클라이언트.

    - asyncio.gather()로 BM25/Vector 병렬 실행
    - Weighted RRF(Reciprocal Rank Fusion)로 결과 병합
    - 가중치(vector_weight, keyword_weight)는 환경변수/config에서 조정
    - Vector 검색 실패 시 BM25 단독 폴백
    """

    def __init__(self) -> None:
        s = get_setting()
        self._endpoint = s.AZURE_SEARCH_ENDPOINT
        self._key = s.AZURE_SEARCH_KEY
        self._default_index = s.AZURE_SEARCH_INDEX_NAME
        self._vector_field = s.HYBRID_SEARCH_VECTOR_FIELD
        self._k_nearest = s.HYBRID_SEARCH_K_NEAREST
        self._top_n = s.HYBRID_SEARCH_TOP_N
        self._vector_weight = s.HYBRID_SEARCH_VECTOR_WEIGHT
        self._keyword_weight = s.HYBRID_SEARCH_KEYWORD_WEIGHT
        self._rrf_k = s.HYBRID_SEARCH_RRF_K

    # ── public ───────────────────────────────────────────────

    async def search(
        self,
        query: str,
        top: int | None = None,
        index_name: str | None = None,
    ) -> tuple[str, list[dict]]:
        """
        하이브리드 검색 실행.

        Args:
            query: 검색 쿼리
            top: 최종 반환 문서 수 (None이면 HYBRID_SEARCH_TOP_N)
            index_name: 인덱스 이름 (None이면 AZURE_SEARCH_INDEX_NAME)

        Returns:
            (context, sources) — 기존 search_documents()와 동일한 시그니처
        """
        index = index_name or self._default_index
        top_n = top if top is not None else self._top_n

        if not self._endpoint or not self._key or not index:
            logger.warning("[hybrid_search] Azure AI Search 설정이 없습니다.")
            return "", []

        total_start = time.monotonic()

        # BM25 + Vector 병렬 실행
        bm25_results, vector_results = await asyncio.gather(
            self._keyword_search(query, index),
            self._vector_search(query, index),
            return_exceptions=True,
        )

        # 실패 처리: Vector 실패 → BM25 단독, BM25 실패 → 빈 결과
        if isinstance(vector_results, Exception):
            if isinstance(vector_results, HttpResponseError):
                logger.warning(
                    f"[hybrid_search] vector search failed (HTTP {vector_results.status_code}), "
                    f"index={index}, BM25 fallback. "
                    f"인덱스에 벡터라이저/벡터 필드({self._vector_field})가 설정되어 있는지 확인하세요."
                )
            else:
                logger.warning(
                    f"[hybrid_search] vector search failed, index={index}, "
                    f"error_type={type(vector_results).__name__}: {vector_results}"
                )
            vector_results = []
        if isinstance(bm25_results, Exception):
            logger.error(
                f"[hybrid_search] BM25 search failed, index={index}, "
                f"error_type={type(bm25_results).__name__}: {bm25_results}"
            )
            bm25_results = []

        # Weighted RRF 병합
        merged = self._rrf_merge(bm25_results, vector_results, top_n)

        total_ms = round((time.monotonic() - total_start) * 1000, 1)
        logger.info(
            f"[hybrid_search] query='{query[:30]}...' "
            f"bm25={len(bm25_results)} vector={len(vector_results)} "
            f"merged={len(merged)} elapsed={total_ms}ms"
        )

        return self._format_results(merged)

    # ── BM25 키워드 검색 ─────────────────────────────────────

    async def _keyword_search(self, query: str, index: str) -> list[dict]:
        start = time.monotonic()

        client = AzureSearchClient(
            self._endpoint, index, AzureKeyCredential(self._key),
        )
        try:
            results = []
            async for doc in client.search(
                search_text=query,
                top=self._k_nearest,
                query_type="simple",
            ):
                results.append(self._extract_doc(doc))

            elapsed_ms = round((time.monotonic() - start) * 1000, 1)
            logger.info(f"[hybrid_search:bm25] {len(results)} results, {elapsed_ms}ms")
            return results
        finally:
            await client.close()

    # ── Vector 시맨틱 검색 ───────────────────────────────────

    async def _vector_search(self, query: str, index: str) -> list[dict]:
        """
        Azure AI Search 통합 벡터라이저(integrated vectorizer) 사용.
        인덱스에 벡터라이저가 설정되어 있어야 동작합니다.
        """
        start = time.monotonic()

        vector_query = VectorizableTextQuery(
            text=query,
            k_nearest_neighbors=self._k_nearest,
            fields=self._vector_field,
            # TODO: Semantic Ranker 도입 시 weight 파라미터 활용 가능
        )

        client = AzureSearchClient(
            self._endpoint, index, AzureKeyCredential(self._key),
        )
        try:
            results = []
            async for doc in client.search(
                search_text=None,
                vector_queries=[vector_query],
                top=self._k_nearest,
            ):
                results.append(self._extract_doc(doc))

            elapsed_ms = round((time.monotonic() - start) * 1000, 1)
            logger.info(f"[hybrid_search:vector] {len(results)} results, {elapsed_ms}ms")
            return results
        finally:
            await client.close()

    # ── 문서 필드 추출 ───────────────────────────────────────

    @staticmethod
    def _extract_doc(doc: dict) -> dict:
        content = doc.get("content", doc.get("chunk", ""))
        title = doc.get("title", doc.get("metadata_storage_name", ""))
        return {
            "id": doc.get("id", doc.get("metadata_storage_path", "")),
            "content": content,
            "title": title,
            "score": doc.get("@search.score", 0),
            "reranker_score": doc.get("@search.reranker_score"),
        }

    # ── Weighted RRF 병합 ────────────────────────────────────

    def _rrf_merge(
        self,
        bm25_results: list[dict],
        vector_results: list[dict],
        top_n: int,
    ) -> list[dict]:
        """
        Weighted RRF(Reciprocal Rank Fusion).

        score = keyword_weight × 1/(k + rank_bm25) + vector_weight × 1/(k + rank_vector)

        k=60 고정값 (Azure AI Search 내부 RRF와 동일).
        """
        k = self._rrf_k
        kw_w = self._keyword_weight
        vec_w = self._vector_weight

        score_map: dict[str, dict] = {}

        for rank, doc in enumerate(bm25_results, start=1):
            doc_key = self._doc_key(doc)
            entry = score_map.setdefault(doc_key, self._empty_entry(doc))
            entry["rrf_score"] += kw_w * (1.0 / (k + rank))
            entry["bm25_rank"] = rank
            entry["bm25_score"] = doc.get("score", 0)

        for rank, doc in enumerate(vector_results, start=1):
            doc_key = self._doc_key(doc)
            entry = score_map.setdefault(doc_key, self._empty_entry(doc))
            entry["rrf_score"] += vec_w * (1.0 / (k + rank))
            entry["vector_rank"] = rank
            entry["vector_score"] = doc.get("score", 0)
            # reranker_score는 vector 결과에서만 존재 가능
            if doc.get("reranker_score") is not None:
                entry["reranker_score"] = doc["reranker_score"]

        sorted_results = sorted(
            score_map.values(), key=lambda x: x["rrf_score"], reverse=True,
        )
        return sorted_results[:top_n]

    @staticmethod
    def _doc_key(doc: dict) -> str:
        """문서 고유 키 생성 — id > title > content hash 순으로 폴백"""
        if doc.get("id"):
            return doc["id"]
        if doc.get("title"):
            return doc["title"]
        return hashlib.md5(doc.get("content", "")[:200].encode()).hexdigest()

    @staticmethod
    def _empty_entry(doc: dict) -> dict:
        return {
            "doc": doc,
            "rrf_score": 0.0,
            "bm25_rank": None,
            "bm25_score": None,
            "vector_rank": None,
            "vector_score": None,
            "reranker_score": None,
        }

    # ── 결과 포맷팅 (기존 search_documents 출력 형식 호환) ──

    @staticmethod
    def _format_results(merged: list[dict]) -> tuple[str, list[dict]]:
        if not merged:
            return "", []

        context_parts: list[str] = []
        sources: list[dict] = []

        for i, item in enumerate(merged):
            doc = item["doc"]
            content = doc.get("content", "")
            title = doc.get("title", f"문서 {i + 1}")

            if not content:
                logger.debug(f"[hybrid_search] RRF rank {i+1} 문서에 content 없음, skip: title={title}")
                continue

            context_parts.append(f"[자료 {i + 1}: {title}]\n{content}")
            sources.append({
                "index": i + 1,
                "title": title,
                "score": round(item["rrf_score"], 6),
                "bm25_score": item["bm25_score"],
                "bm25_rank": item["bm25_rank"],
                "vector_score": item["vector_score"],
                "vector_rank": item["vector_rank"],
                "reranker_score": item["reranker_score"],
                # TODO: Semantic Ranker(cross-encoder reranker) 도입 시
                #       reranker_score를 최종 정렬 기준으로 활용.
                #       Azure AI Search의 semanticConfiguration 설정 후
                #       search() 호출에 query_type="semantic" 추가하면 자동 반환됨.
                "content": content,
                "content_preview": content[:200],
            })

        context = "\n\n---\n\n".join(context_parts)
        return context, sources


# ── 모듈 레벨 편의 함수 (기존 search_documents 시그니처 호환) ────

_client: HybridSearchClient | None = None


def _get_hybrid_client() -> HybridSearchClient:
    global _client
    if _client is None:
        _client = HybridSearchClient()
    return _client


async def hybrid_search_documents(
    query: str,
    top: int | None = None,
    index_name: str | None = None,
) -> tuple[str, list[dict]]:
    """
    하이브리드 검색 편의 함수.

    기존 search_documents(query, top, index_name)과 동일한 시그니처로
    드롭인 교체 가능.
    """
    return await _get_hybrid_client().search(query, top=top, index_name=index_name)
