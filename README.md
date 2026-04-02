# backend-chat

채팅/LLM 스트리밍 — 포트 :8003

## 빠른 시작

> 최초 셋업이 안 됐다면 [개발자 가이드 — 최초 셋업](https://github.com/agent-template-apps/shared-infra/blob/main/docs/DEVELOPER_GUIDE.md#2-%EC%B5%9C%EC%B4%88-%EC%85%8B%EC%97%85-%EC%8B%A0%EA%B7%9C-%EA%B0%9C%EB%B0%9C%EC%9E%90)을 먼저 진행하세요.

```bash
# 서버 실행
uv run uvicorn app.main:app --port 8003 --reload
```

## 개발 가이드

모든 개발 워크플로우, 코드 표준, 스크립트 사용법은 shared-infra에서 관리합니다.

- [개발자 가이드](https://github.com/agent-template-apps/shared-infra/blob/main/docs/DEVELOPER_GUIDE.md) — 셋업, 일일 개발, 워크플로우, 트러블슈팅
- [백엔드 개발 표준](https://github.com/agent-template-apps/shared-infra/blob/main/docs/standards/backend-standards.md)
- [프론트엔드 개발 표준](https://github.com/agent-template-apps/shared-infra/blob/main/docs/standards/frontend-standards.md)

## 주요 명령어

| 명령어 | 설명 |
|--------|------|
| `claude` → `/start-task {이슈번호}` | 태스크 시작 (브랜치 생성) |
| `claude` → `/finish-task` | 린트 → 커밋 → PR 자동 생성 |
| `./infra/scripts/setup-repo.sh` | shared-infra 설정 재배포 |
