# backend-agent 개발 규칙

## 공통 규칙
@infra/CLAUDE.md

## 개발표준
- IMPORTANT: 백엔드 개발표준 숙지 필수 → `@infra/docs/standards/backend-standards.md`
- 에러 처리: ServiceException + ErrorCode 사용. bare except 금지
- API 응답: `{"success": bool, "error": {"code": int, "name": str, "message": str}}`
- 아키텍처: routes(요청처리) → service(비즈니스) → repository(DB접근)
- 보안: get_current_user() 의존성으로 사용자 정보 주입

## 빌드 & 테스트
- 의존성 설치: `uv sync`
- 테스트 실행: `uv run pytest --tb=short -q`
- 단일 테스트: `uv run pytest tests/test_xxx.py -k "test_name" -v`
- 린트: `uv run ruff check .`
- 포맷: `uv run ruff format .`
- 서버 실행: `uv run uvicorn app.main:app --reload --port 8006`

## 디렉토리 구조
- `app/api/routes/` - API 라우터 (요청/응답 처리만)
- `app/service/` - 비즈니스 로직 (에이전트, RAG 등)
- `app/service/model/` - Pydantic v2 모델
- `app/core/` - 설정, 에러처리, 로깅, 보안
- `app/core/error/` - ErrorCode enum + ServiceException + error_handler
- `app/common/util/` - LLM Gateway 클라이언트, 외부 서비스 요청, 프롬프트 관리
- `tests/` - 테스트 코드

## 이 앱의 규칙
- FastAPI router prefix: `/api/v1/agent`
- 포트: 8006
- RAG 기반 문서 검색 + LLM 응답 생성
- 프론트엔드 채팅 응답 (SSE 스트리밍) + 백그라운드 작업 (DB 저장)
- LLM 호출: LLM Gateway 경유 또는 Azure OpenAI 직접 호출
- LangChain/LangGraph 기반 에이전트 구성
- Gateway 헤더에서 사용자 정보 추출 (JWT 직접 검증 안 함)
- 에이전트 관련 에러 코드: 17XXX 대역 사용
- 새 엔드포인트 추가 시 반드시 테스트 작성
- 서비스 레이어에 비즈니스 로직 분리 (라우터에 직접 작성 금지)
