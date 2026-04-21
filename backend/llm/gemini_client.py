"""
Google Gemini LLM client.

Wraps the google-genai SDK to conform to BaseLLMClient.
Gemini has a free tier (no credit card needed) — get your API key at
https://aistudio.google.com/app/apikey

Free tier rate limits (as of 2025):
  gemini-2.0-flash:    15 RPM, 1 500 req/day
  gemini-1.5-flash:    15 RPM, 1 500 req/day
  gemini-1.5-flash-8b: 15 RPM, 1 500 req/day

Requires: pip install google-genai
"""

from loguru import logger

from backend.config import settings
from backend.llm.base import BaseLLMClient, SYSTEM_PROMPT


class GeminiClient(BaseLLMClient):
    """
    LLM client for the Google Gemini API.

    Uses the official google-genai SDK. Recommended free models:
      - gemini-2.0-flash      (fast, free tier)
      - gemini-1.5-flash      (fast, free tier)
      - gemini-1.5-flash-8b   (smallest, highest free rate limit)
      - gemini-1.5-pro        (more capable, lower free rate limit)
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout: int | None = None,
    ) -> None:
        try:
            from google import genai
        except ImportError:
            raise ImportError(
                "google-genai package is required for GeminiClient. "
                "Install it with: pip install google-genai"
            )

        self.model = model or settings.gemini_model
        self._timeout = timeout or settings.ollama_timeout
        resolved_key = api_key or settings.gemini_api_key

        from google import genai
        self._client = genai.Client(api_key=resolved_key)

        logger.info(f"GeminiClient initialized: model={self.model}")

    @property
    def _supports_system_instruction(self) -> bool:
        """Gemma models don't support system_instruction — only Gemini does."""
        return "gemma" not in self.model.lower()

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> str:
        from google.genai import types

        system = system or SYSTEM_PROMPT
        try:
            if self._supports_system_instruction:
                config = types.GenerateContentConfig(
                    system_instruction=system,
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                )
                contents = prompt
            else:
                # Gemma: prepend system prompt directly into the user message
                config = types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                )
                contents = f"{system}\n\n{prompt}"

            response = self._client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config,
            )
            text = response.text or ""
            logger.info(f"Gemini generation complete: {len(text)} chars, model={self.model}")
            return text.strip()

        except Exception as e:
            logger.error(f"Gemini generation error: {e}")
            return f"Error: Gemini generation failed: {e}"

    def is_available(self) -> bool:
        try:
            models = [m.name for m in self._client.models.list()]
            available = any(self.model in m for m in models)
            if not available:
                logger.warning(f"Model '{self.model}' not found in Gemini model list.")
            return available
        except Exception as e:
            logger.error(f"Gemini health check failed: {e}")
            return False
