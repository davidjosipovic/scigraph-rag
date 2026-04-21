"""
OpenAI LLM client.

Wraps the openai Python SDK to conform to BaseLLMClient.
Supports any OpenAI-compatible endpoint (OpenAI, Azure, local vLLM, etc.)
via the base_url setting.

Requires: pip install openai
"""

from loguru import logger

from backend.config import settings
from backend.llm.base import BaseLLMClient, SYSTEM_PROMPT


class OpenAIClient(BaseLLMClient):
    """
    LLM client for OpenAI-compatible APIs.

    Uses the official openai SDK. Compatible with:
      - OpenAI (gpt-4o, gpt-4o-mini, gpt-3.5-turbo, ...)
      - Any OpenAI-compatible local server (vLLM, LM Studio, ...)
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout: int | None = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package is required for OpenAIClient. "
                "Install it with: pip install openai"
            )

        self.model = model or settings.openai_model
        self._timeout = timeout or settings.ollama_timeout
        resolved_key = api_key or settings.openai_api_key
        resolved_url = base_url or settings.openai_base_url or None

        self._client = OpenAI(
            api_key=resolved_key,
            base_url=resolved_url,
            timeout=self._timeout,
        )
        logger.info(f"OpenAIClient initialized: model={self.model}")

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> str:
        system = system or SYSTEM_PROMPT
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            text = response.choices[0].message.content or ""
            logger.info(f"OpenAI generation complete: {len(text)} chars, model={self.model}")
            return text.strip()

        except Exception as e:
            logger.error(f"OpenAI generation error: {e}")
            return f"Error: OpenAI generation failed: {e}"

    def is_available(self) -> bool:
        try:
            models = self._client.models.list()
            available = any(self.model in m.id for m in models.data)
            if not available:
                logger.warning(
                    f"Model '{self.model}' not found in OpenAI model list."
                )
            return available
        except Exception as e:
            logger.error(f"OpenAI health check failed: {e}")
            return False
