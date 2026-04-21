"""
Ollama LLM client for local language model inference.

Communicates with a locally running Ollama server to generate text completions.

Ollama API reference: https://github.com/ollama/ollama/blob/main/docs/api.md
"""

import httpx
from loguru import logger

from backend.config import settings
from backend.llm.base import BaseLLMClient, SYSTEM_PROMPT

# Re-export so existing imports from this module continue to work
from backend.llm.base import get_prompt_template  # noqa: F401


class OllamaClient(BaseLLMClient):
    """
    Client for the Ollama local LLM API.

    Sends generation requests to a locally running Ollama instance
    and returns the model's text output.
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: int | None = None,
    ) -> None:
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.model = model or settings.ollama_model
        self.timeout = timeout or settings.ollama_timeout
        # Persistent client — reuses TCP connections across requests (connection pooling).
        # httpx.Client is thread-safe so it can be shared across run_in_executor calls.
        self._http = httpx.Client(timeout=self.timeout)

        logger.info(f"OllamaClient initialized: model={self.model}, url={self.base_url}")

    def __del__(self) -> None:
        try:
            self._http.close()
        except Exception:
            pass

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> str:
        """
        Generate a text completion using the Ollama API.

        Args:
            prompt:      The user/context prompt.
            system:      Optional system prompt (defaults to SYSTEM_PROMPT).
            temperature: Sampling temperature (lower = more deterministic).
            max_tokens:  Maximum tokens to generate.

        Returns:
            The generated text string.
        """
        system = system or SYSTEM_PROMPT
        url = f"{self.base_url}/api/generate"

        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        logger.debug(f"Sending request to Ollama ({self.model})...")

        try:
            response = self._http.post(url, json=payload)
            response.raise_for_status()

            result = response.json()
            text = result.get("response", "")
            logger.info(
                f"LLM generation complete: {len(text)} chars, "
                f"model={result.get('model', '?')}"
            )
            return text.strip()

        except httpx.ConnectError:
            logger.error(
                "Cannot connect to Ollama. Is it running? "
                f"Tried: {self.base_url}"
            )
            return (
                "Error: Cannot connect to the local LLM (Ollama). "
                "Please ensure Ollama is running with: ollama serve"
            )
        except httpx.HTTPStatusError as e:
            logger.error(f"Ollama API error: {e.response.status_code} - {e.response.text}")
            return f"Error: LLM returned status {e.response.status_code}"
        except Exception as e:
            logger.error(f"Unexpected LLM error: {e}")
            return f"Error: Unexpected error during LLM generation: {e}"

    def is_available(self) -> bool:
        """Check if the Ollama server is reachable and the model is loaded."""
        try:
            resp = self._http.get(f"{self.base_url}/api/tags", timeout=5)
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]
            available = any(self.model in m for m in models)
            if not available:
                logger.warning(
                    f"Model '{self.model}' not found. "
                    f"Available models: {models}"
                )
            return available
        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            return False


# Module-level convenience instance used by the query classifier and entity extractor.
# These infrastructure components always run on the local Ollama model regardless
# of which provider is configured for answer generation.
ollama_client = OllamaClient()
