"""/app/common/util/llm_gateway_client.py"""

import json
from typing import Any, AsyncGenerator, Dict, List

import httpx
from core.config import get_setting
from core.log.logging import get_logging
from core.security import create_system_token
from service.model.agent import ChatHistory

logger = get_logging()
settings = get_setting()


class LLMGatewayClient:
    """
    LLM Gateway 통신을 위한 통합 클라이언트

    특징:
    - 헤더 생성, 메시지 변환 등 공통 로직 제공
    - HTTP 통신 로직 포함
    - Agent에서 간편하게 사용할 수 있는 통합 API
    """

    def __init__(self, llm_gateway_url: str):
        """
        Args:
            llm_gateway_url: LLM Gateway 베이스 URL
        """
        self.llm_gateway_url = llm_gateway_url

    def create_headers(
        self,
        user_id: str,
        org_id: str | None,
        agent_name: str,
    ) -> Dict[str, str]:
        """
        LLM Gateway 호출을 위한 헤더 생성

        Args:
            user_id: 사용자 ID
            org_id: 조직 ID (없을 수 있음)
            agent_name: 에이전트 이름

        Returns:
            HTTP 헤더 딕셔너리
        """
        system_token = create_system_token()
        headers = {
            "Content-Type": "application/json",
            "Internal-Key": system_token,
            "x-user-id": user_id,
            "x-org-id": org_id if org_id else "",
            "x-agent-name": agent_name,
        }
        return headers

    def convert_to_openai_format(
        self,
        messages: List[ChatHistory],
    ) -> List[Dict[str, Any]]:
        """
        ChatHistory를 OpenAI API 형식으로 변환
        role: ai -> assistant로 변환
        """
        openai_messages = []
        for message in messages:
            role = message.role.value
            if role == "ai":
                role = "assistant"
            openai_messages.append({"role": role, "content": message.content})
        return openai_messages

    async def call_completions_stream(
        self,
        user_id: str,
        org_id: str | None,
        provider: str,
        model: str,
        messages: List[ChatHistory],
        prompt_variables: dict[str, str] | None,
        agent_name: str,
    ) -> AsyncGenerator[str, None]:
        """
        LLM Gateway의 /api/v1/chat/completions 엔드포인트 호출 (스트리밍)

        Args:
            user_id: 사용자 ID
            org_id: 조직 ID
            provider: LLM 제공자 (예: "openai", "anthropic")
            model: 모델명 (예: "gpt-4", "claude-3")
            messages: 채팅 메시지 히스토리
            agent_name: 에이전트 이름

        Returns:
            AsyncGenerator[str, None]: SSE 스트림
        """
        headers = self.create_headers(user_id=user_id, org_id=org_id, agent_name=agent_name)

        openai_messages = self.convert_to_openai_format(messages)

        body = {
            "provider": provider,
            "model": model,
            "messages": openai_messages,
            "prompt_variables": prompt_variables,
            "stream": True,
            "skip_prompt_injection": True,
        }

        logger.info(f"Calling completions (stream): provider={provider}, model={model}")

        url = f"{self.llm_gateway_url}/api/v1/chat/completions"
        async for chunk in self._post_streaming(url, headers, body):
            yield chunk

    async def call_completions_non_stream(
        self,
        user_id: str,
        org_id: str | None,
        provider: str,
        model: str,
        messages: List[ChatHistory],
        prompt_variables: dict[str, str] | None,
        agent_name: str,
    ) -> Dict[str, Any]:
        """
        LLM Gateway의 /api/v1/chat/completions 엔드포인트 호출 (비스트리밍)

        Args:
            user_id: 사용자 ID
            org_id: 조직 ID
            provider: LLM 제공자 (예: "openai", "anthropic")
            model: 모델명 (예: "gpt-4", "claude-3")
            messages: 채팅 메시지 히스토리
            agent_name: 에이전트 이름

        Returns:
            Dict[str, Any]: JSON 응답
        """
        headers = self.create_headers(user_id=user_id, org_id=org_id, agent_name=agent_name)

        openai_messages = self.convert_to_openai_format(messages)

        body = {
            "provider": provider,
            "model": model,
            "messages": openai_messages,
            "prompt_variables": prompt_variables,
            "stream": False,
            "skip_prompt_injection": True,
        }

        logger.info(f"Calling completions (non-stream): provider={provider}, model={model}")

        url = f"{self.llm_gateway_url}/api/v1/chat/completions"
        return await self._post_non_streaming(url, headers, body)

    async def _post_non_streaming(
        self,
        url: str,
        headers: Dict[str, str],
        body: Dict[str, Any],
    ) -> Dict[str, Any]:
        """비스트리밍 POST 요청"""
        logger.info(f"Calling LLM Gateway (non-streaming): {url}")

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(url, headers=headers, json=body)

                if response.status_code != 200:
                    error_msg = f"LLM Gateway error: {response.status_code}"
                    logger.error(error_msg)
                    try:
                        error_detail = response.text
                        logger.error(f"Error detail: {error_detail}")
                    except Exception as e:
                        logger.error(f"Error detail: {e}")
                    return {"error": error_msg, "content": f"오류가 발생했습니다: {error_msg}"}

                return response.json()  # type: ignore[no-any-return]

        except httpx.TimeoutException:
            error_msg = "LLM Gateway 호출 시간 초과"
            logger.error(error_msg)
            return {"error": error_msg, "content": f"오류가 발생했습니다: {error_msg}"}
        except httpx.RequestError as e:
            error_msg = f"LLM Gateway 연결 오류: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg, "content": f"오류가 발생했습니다: {error_msg}"}
        except Exception as e:
            error_msg = f"예기치 않은 오류: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg, "content": f"오류가 발생했습니다: {error_msg}"}

    async def _post_streaming(
        self,
        url: str,
        headers: Dict[str, str],
        body: Dict[str, Any],
    ) -> AsyncGenerator[str, None]:
        """스트리밍 POST 요청 (SSE)"""
        logger.info(f"Calling LLM Gateway (streaming): {url}")

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream(
                    "POST",
                    url,
                    headers=headers,
                    json=body,
                ) as response:
                    if response.status_code != 200:
                        error_msg = f"LLM Gateway error: {response.status_code}"
                        logger.error(error_msg)
                        try:
                            error_detail = await response.aread()
                            logger.error(f"Error detail: {error_detail.decode()}")
                        except Exception as e:
                            logger.error(f"Error detail: {e}")
                        yield f"오류가 발생했습니다: {error_msg}"
                        return

                    # SSE 스트림 읽기
                    async for line in response.aiter_lines():
                        if line.strip():
                            # SSE 형식: "data: {...}"
                            if line.startswith("data: "):
                                data_str = line[6:]  # "data: " 제거
                                if data_str == "[DONE]":
                                    break
                                try:
                                    # LLM Gateway가 반환하는 SSE 데이터를 그대로 yield
                                    yield line + "\n\n"
                                except json.JSONDecodeError:
                                    logger.warning(f"Failed to parse SSE data: {data_str}")

        except httpx.TimeoutException:
            error_msg = "LLM Gateway 호출 시간 초과"
            logger.error(error_msg)
            yield f"data: {json.dumps({'content': f'오류가 발생했습니다: {error_msg}'}, ensure_ascii=False)}\n\n"
        except httpx.RequestError as e:
            error_msg = f"LLM Gateway 연결 오류: {str(e)}"
            logger.error(error_msg)
            yield f"data: {json.dumps({'content': f'오류가 발생했습니다: {error_msg}'}, ensure_ascii=False)}\n\n"
        except Exception as e:
            error_msg = f"예기치 않은 오류: {str(e)}"
            logger.error(error_msg)
            yield f"data: {json.dumps({'content': f'오류가 발생했습니다: {error_msg}'}, ensure_ascii=False)}\n\n"
