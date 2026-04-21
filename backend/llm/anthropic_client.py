"""
Anthropic LLM client.

Wraps the anthropic Python SDK to conform to BaseLLMClient.

Requires: pip install anthropic
"""

from loguru import logger

from backend.config import settings
from backend.llm.base import BaseLLMClient, SYSTEM_PROMPT


class AnthropicClient(BaseLLMClient):
    """
    LLM client for the Anthropic API (Claude models).

    Uses the official anthropic SDK. Compatible with:
      - claude-opus-4-6
      - claude-sonnet-4-6
      - claude-haiku-4-5-20251001
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout: int | None = None,
    ) -> None:
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError(
                "anthropic package is required for AnthropicClient. "
                "Install it with: pip install anthropic"
            )

        self.model = model or settings.anthropic_model
        self._timeout = timeout or settings.ollama_timeout
        resolved_key = api_key or settings.anthropic_api_key

        self._client = Anthropic(
            api_key=resolved_key,
            timeout=self._timeout,
        )
        logger.info(f"AnthropicClient initialized: model={self.model}")

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> str:
        system = system or SYSTEM_PROMPT
        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            text = response.content[0].text
            logger.info(f"Anthropic generation complete: {len(text)} chars, model={self.model}")
            return text.strip()

        except Exception as e:
            logger.error(f"Anthropic generation error: {e}")
            return f"Error: Anthropic generation failed: {e}"

    def is_available(self) -> bool:
        # Anthropic has no free model-list endpoint — just verify the API key is set.
        try:
            from anthropic import Anthropic
            resolved_key = settings.anthropic_api_key
            if not resolved_key:
                logger.warning("Anthropic API key is not configured.")
                return False
            # Minimal probe: list models (available in anthropic SDK >= 0.40)
            models = self._client.models.list()
            available = any(self.model in m.id for m in models.data)
            if not available:
                logger.warning(f"Model '{self.model}' not found in Anthropic model list.")
            return available
        except Exception as e:
            logger.error(f"Anthropic health check failed: {e}")
            return False
