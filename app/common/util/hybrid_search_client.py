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

    # 인텐트별 tags_topic 부스팅 태그 매핑
    _INTENT_TOPIC_BOOST: dict[str, list[str]] = {
        "strategy": ["경영전략", "경영체계", "사업전략", "경영철학"],
        "culture": ["기업문화", "구성원행복", "조직문화", "인간중심경영"],
        "crisis": ["위기관리", "리스크", "경영위기", "위기"],
        "supex": ["SUPEX", "경영체계", "경영철학", "경영관리체계"],
        "hr": ["인재육성", "리더십", "인사관리", "인재"],
    }

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
        vector_query: str | None = None,
        intent: str | None = None,
    ) -> tuple[str, list[dict]]:
        """
        하이브리드 검색 실행.

        Args:
            query: 검색 쿼리 (BM25용)
            top: 최종 반환 문서 수 (None이면 HYBRID_SEARCH_TOP_N)
            index_name: 인덱스 이름 (None이면 AZURE_SEARCH_INDEX_NAME)
            vector_query: Vector 검색 전용 쿼리 (None이면 query 사용)
            intent: 인텐트 (tags_topic 부스팅용)

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
            self._vector_search(vector_query or query, index),
            return_exceptions=True,
        )

        # 실패 처리: Vector 실패 → BM25 단독, BM25 실패 → 빈 결과
        vector_failed = isinstance(vector_results, Exception)
        bm25_failed = isinstance(bm25_results, Exception)

        if vector_failed:
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
        if bm25_failed:
            logger.error(
                f"[hybrid_search] BM25 search failed, index={index}, "
                f"error_type={type(bm25_results).__name__}: {bm25_results}"
            )
            bm25_results = []

        # 한쪽 검색만 성공 시 해당 가중치를 1.0으로 보정
        kw_override = 1.0 if vector_failed and not bm25_failed else None
        vec_override = 1.0 if bm25_failed and not vector_failed else None

        # Weighted RRF 병합
        merged = self._rrf_merge(
            bm25_results,
            vector_results,
            top_n,
            kw_weight_override=kw_override,
            vec_weight_override=vec_override,
            intent=intent,
        )

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
            self._endpoint,
            index,
            AzureKeyCredential(self._key),
        )
        try:
            results = []
            try:
                async for doc in await client.search(
                    search_text=query,
                    top=self._k_nearest,
                    query_type="simple",
                    search_fields=["text_content", "content", "title", "tags_topic"],
                    filter="content_type ne 'image'",
                ):
                    results.append(self._extract_doc(doc))
            except HttpResponseError as e:
                if e.status_code == 400:
                    logger.warning(
                        f"[hybrid_search:bm25] search_fields/filter 오류 (HTTP 400), "
                        f"index={index}, 필드/필터 없이 재시도"
                    )
                    async for doc in await client.search(
                        search_text=query,
                        top=self._k_nearest,
                        query_type="simple",
                    ):
                        results.append(self._extract_doc(doc))
                else:
                    raise

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

        vq = VectorizableTextQuery(
            text=query,
            k_nearest_neighbors=self._k_nearest,
            fields=self._vector_field,
        )

        client = AzureSearchClient(
            self._endpoint,
            index,
            AzureKeyCredential(self._key),
        )
        try:
            results = []
            try:
                async for doc in await client.search(
                    search_text=query,
                    vector_queries=[vq],
                    top=self._k_nearest,
                    query_type="semantic",
                    semantic_configuration_name="default-semantic-config",
                    filter="content_type ne 'image'",
                ):
                    results.append(self._extract_doc(doc))
            except HttpResponseError as e:
                if e.status_code == 400:
                    logger.warning(
                        f"[hybrid_search:vector] semantic ranker 오류 (HTTP 400), "
                        f"index={index}, 기본 벡터 검색으로 폴백"
                    )
                    async for doc in await client.search(
                        search_text=None,
                        vector_queries=[vq],
                        top=self._k_nearest,
                    ):
                        results.append(self._extract_doc(doc))
                else:
                    raise

            elapsed_ms = round((time.monotonic() - start) * 1000, 1)
            logger.info(f"[hybrid_search:vector] {len(results)} results, {elapsed_ms}ms")
            return results
        finally:
            await client.close()

    # ── 문서 필드 추출 ───────────────────────────────────────

    @staticmethod
    def _extract_doc(doc: dict) -> dict:
        content = doc.get("text_content") or doc.get("content") or doc.get("chunk") or ""
        title = doc.get("title", doc.get("metadata_storage_name", ""))
        return {
            "id": doc.get("id", doc.get("metadata_storage_path", "")),
            "content": content,
            "title": title,
            "score": doc.get("@search.score", 0),
            "reranker_score": doc.get("@search.reranker_score"),
            "document_path": doc.get("document_path", ""),
            "page_number": doc.get("page_number"),
            "content_type": doc.get("content_type", ""),
            "context": doc.get("context", ""),
            "tags_topic": doc.get("tags_topic", ""),
            "author": doc.get("author", ""),
            "issue": doc.get("issue", ""),
        }

    # ── Weighted RRF 병합 ────────────────────────────────────

    def _rrf_merge(
        self,
        bm25_results: list[dict],
        vector_results: list[dict],
        top_n: int,
        *,
        kw_weight_override: float | None = None,
        vec_weight_override: float | None = None,
        intent: str | None = None,
    ) -> list[dict]:
        """
        Weighted RRF(Reciprocal Rank Fusion).

        score = keyword_weight × 1/(k + rank_bm25) + vector_weight × 1/(k + rank_vector)

        k=60 고정값 (Azure AI Search 내부 RRF와 동일).
        한쪽 검색 실패 시 *_weight_override로 나머지 가중치를 1.0으로 보정.
        """
        k = self._rrf_k
        kw_w = kw_weight_override if kw_weight_override is not None else self._keyword_weight
        vec_w = vec_weight_override if vec_weight_override is not None else self._vector_weight

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
            score_map.values(),
            key=lambda x: x["rrf_score"],
            reverse=True,
        )

        # 부스트 전 최고 점수 기록 (품질 게이트 기준값으로 사용)
        pre_boost_best = sorted_results[0]["rrf_score"] if sorted_results else 0.0

        # 3) tags_topic 기반 인텐트 매칭 부스트
        if intent and intent in self._INTENT_TOPIC_BOOST:
            boost_tags = self._INTENT_TOPIC_BOOST[intent]
            for item in sorted_results:
                doc_tags = item["doc"].get("tags_topic", "")
                if doc_tags and any(tag in doc_tags for tag in boost_tags):
                    item["rrf_score"] *= 1.2  # 20% 부스트
            # 부스트 후 재정렬
            sorted_results.sort(key=lambda x: x["rrf_score"], reverse=True)

        # ── 품질 게이트 ──────────────────────────────────────
        # 1) 상대적 품질 필터: 부스트 전 최고 점수의 30% 미만인 문서 제거
        #    (부스트 후 best_score로 하면 비매칭 문서가 부당하게 탈락할 수 있음)
        if sorted_results:
            quality_threshold = pre_boost_best * 0.3
            sorted_results = [r for r in sorted_results if r["rrf_score"] >= quality_threshold]

        # 2) 콘텐츠 중복 제거: 70% 이상 텍스트 오버랩인 문서 제거
        sorted_results = self._deduplicate_content(sorted_results)

        top_results = sorted_results[:top_n]

        # RRF 점수 정규화: 최고 점수를 1.0으로 스케일링
        if top_results:
            max_score = top_results[0]["rrf_score"]
            if max_score > 0:
                for item in top_results:
                    item["normalized_score"] = round(item["rrf_score"] / max_score, 4)
            else:
                for item in top_results:
                    item["normalized_score"] = 0.0

        return top_results

    @staticmethod
    def _deduplicate_content(results: list[dict], overlap_threshold: float = 0.7) -> list[dict]:
        """콘텐츠 기반 중복 제거 — 상위 문서 우선 유지, 70% 이상 오버랩 시 하위 문서 제거."""
        if not results:
            return results

        kept: list[dict] = []
        for candidate in results:
            candidate_content = candidate["doc"].get("content", "")
            if not candidate_content:
                kept.append(candidate)
                continue

            # 이미 선택된 문서들과 오버랩 체크 (단어 집합 기반)
            candidate_words = set(candidate_content.split())
            is_duplicate = False
            for existing in kept:
                existing_content = existing["doc"].get("content", "")
                if not existing_content:
                    continue
                existing_words = set(existing_content.split())
                if not candidate_words or not existing_words:
                    continue
                overlap = len(candidate_words & existing_words) / min(len(candidate_words), len(existing_words))
                if overlap >= overlap_threshold:
                    is_duplicate = True
                    logger.debug(
                        f"[quality_gate] 중복 제거: overlap={overlap:.0%} "
                        f"title='{candidate['doc'].get('title', '')[:30]}'"
                    )
                    break

            if not is_duplicate:
                kept.append(candidate)

        return kept

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
    def _compute_absolute_relevance(item: dict) -> float:
        """
        0~1 범위의 절대적 관련도 점수.
        상대적 정규화(normalized_score)와 달리, 검색 결과가 무관해도 낮은 점수를 반환.
        - vector_score: Azure AI Search 코사인 유사도 (0~1)
        - rrf_score: 양쪽 검색 모두 상위권이어야 높은 값
        - bm25_rank: 키워드 매칭 순위 (낮을수록 좋음)
        """
        signals: list[float] = []

        # Signal 1: 벡터 유사도 (있으면, 이미 0~1 범위)
        vec_score = item.get("vector_score")
        if vec_score is not None and vec_score > 0:
            signals.append(vec_score)

        # Signal 2: RRF raw score 정규화
        # 이론적 최댓값: 양쪽 rank 1일 때 = 2 × (0.5 / 61) ≈ 0.016393
        rrf_raw = item.get("rrf_score", 0)
        _RRF_THEORETICAL_MAX = (0.5 / (60 + 1)) * 2
        rrf_normalized = min(rrf_raw / _RRF_THEORETICAL_MAX, 1.0)
        signals.append(rrf_normalized)

        # Signal 3: BM25 순위 기반 (rank 1~5: 좋음, 10+: 나쁨)
        bm25_rank = item.get("bm25_rank")
        if bm25_rank is not None:
            rank_signal = max(0.0, 1.0 - (bm25_rank - 1) / 20.0)
            signals.append(rank_signal)

        # Signal 4: Semantic Reranker 점수 (0~4 범위, 4로 나눠 0~1 정규화)
        reranker = item.get("reranker_score")
        if reranker is not None and reranker > 0:
            signals.append(min(reranker / 4.0, 1.0))

        return round(sum(signals) / len(signals), 4) if signals else 0.0

    @staticmethod
    def _format_results(merged: list[dict]) -> tuple[str, list[dict]]:
        if not merged:
            return "", []

        context_parts: list[str] = []
        sources: list[dict] = []
        source_idx = 0

        for i, item in enumerate(merged):
            doc = item["doc"]
            content = doc.get("content", "")
            title = doc.get("title", "")

            if not content:
                logger.debug(f"[hybrid_search] RRF rank {i + 1} 문서에 content 없음, skip: title={title}")
                continue

            source_idx += 1
            title = doc.get("title", f"문서 {source_idx}")
            abs_relevance = HybridSearchClient._compute_absolute_relevance(item)

            # 절대 관련도 기반 신뢰도 등급
            if abs_relevance >= 0.7:
                confidence_level = "high"
            elif abs_relevance >= 0.4:
                confidence_level = "medium"
            else:
                confidence_level = "low"

            ctx_desc = doc.get("context", "")

            # low-confidence 문서는 LLM 컨텍스트에서 제외 (sources에는 포함)
            if confidence_level != "low":
                if ctx_desc:
                    context_parts.append(f"[자료 {source_idx}: {title}]\n[맥락: {ctx_desc}]\n{content}")
                else:
                    context_parts.append(f"[자료 {source_idx}: {title}]\n{content}")

            sources.append(
                {
                    "included_in_context": confidence_level != "low",
                    "index": source_idx,
                    "title": title,
                    "context_desc": ctx_desc,
                    "score": item.get("normalized_score", round(item["rrf_score"], 6)),
                    "rrf_score_raw": round(item["rrf_score"], 6),
                    "absolute_relevance": abs_relevance,
                    "confidence_level": confidence_level,
                    "bm25_score": item["bm25_score"],
                    "bm25_rank": item["bm25_rank"],
                    "vector_score": item["vector_score"],
                    "vector_rank": item["vector_rank"],
                    "reranker_score": item["reranker_score"],
                    "content": content,
                    "content_preview": content[:200],
                    "document_path": doc.get("document_path", ""),
                    "page_number": doc.get("page_number"),
                    "tags_topic": doc.get("tags_topic", ""),
                    "author": doc.get("author", ""),
                    "issue": doc.get("issue", ""),
                }
            )

        # 폴백: 모든 문서가 low-confidence면 최상위 1개를 컨텍스트에 포함
        if not context_parts and sources:
            logger.warning("[hybrid_search] 모든 문서가 low-confidence — 최상위 문서 1개를 컨텍스트에 포함")
            top_src = sources[0]
            ctx_desc = top_src.get("context_desc", "")
            if ctx_desc:
                context_parts.append(f"[자료 {top_src['index']}: {top_src['title']}]\n[맥락: {ctx_desc}]\n{top_src['content']}")
            else:
                context_parts.append(f"[자료 {top_src['index']}: {top_src['title']}]\n{top_src['content']}")
            top_src["included_in_context"] = True

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
    vector_query: str | None = None,
    intent: str | None = None,
) -> tuple[str, list[dict]]:
    """
    하이브리드 검색 편의 함수.

    기존 search_documents(query, top, index_name)과 동일한 시그니처로
    드롭인 교체 가능.

    Args:
        query: BM25 키워드 검색용 쿼리
        top: 최종 반환 문서 수
        index_name: 인덱스 이름
        vector_query: Vector 검색 전용 쿼리 (None이면 query 사용)
        intent: 인텐트 (tags_topic 부스팅용)
    """
    return await _get_hybrid_client().search(
        query,
        top=top,
        index_name=index_name,
        vector_query=vector_query,
        intent=intent,
    )
