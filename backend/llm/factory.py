"""
LLM client factory.

Creates the appropriate BaseLLMClient based on the configured provider.
The provider is controlled by the LLM_PROVIDER environment variable
(or settings.llm_provider), defaulting to "ollama".

Usage:
    from backend.llm.factory import create_llm_client

    # Uses settings.llm_provider and the matching model/key settings
    client = create_llm_client()

    # Or override explicitly for evaluation runs
    client = create_llm_client(provider="openai", model="gpt-4o-mini")
"""

from loguru import logger

from backend.config import settings
from backend.llm.base import BaseLLMClient

_SUPPORTED_PROVIDERS = ("ollama", "openai", "anthropic", "gemini")


def create_llm_client(
    provider: str | None = None,
    model: str | None = None,
) -> BaseLLMClient:
    """
    Instantiate and return an LLM client for the given provider.

    Args:
        provider: One of "ollama", "openai", "anthropic".
                  Defaults to settings.llm_provider (env: LLM_PROVIDER).
        model:    Model name override. If None, uses the provider's default
                  model from settings.

    Returns:
        A BaseLLMClient ready to call .generate().

    Raises:
        ValueError: If the provider name is not recognised.
        ImportError: If the provider's SDK package is not installed.
    """
    resolved_provider = (provider or settings.llm_provider).lower().strip()
    logger.info(f"Creating LLM client: provider={resolved_provider}, model={model or '(default)'}")

    if resolved_provider == "ollama":
        from backend.llm.ollama_client import OllamaClient
        return OllamaClient(model=model)

    if resolved_provider == "openai":
        from backend.llm.openai_client import OpenAIClient
        return OpenAIClient(model=model)

    if resolved_provider == "anthropic":
        from backend.llm.anthropic_client import AnthropicClient
        return AnthropicClient(model=model)

    if resolved_provider == "gemini":
        from backend.llm.gemini_client import GeminiClient
        return GeminiClient(model=model)

    raise ValueError(
        f"Unknown LLM provider: '{resolved_provider}'. "
        f"Supported providers: {_SUPPORTED_PROVIDERS}"
    )
