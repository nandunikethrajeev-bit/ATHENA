"""Model-agnostic LLM interface and adapters for ATHENA (Milestone M3).

Supports:
- MockLLMClient: Deterministic, zero-cost, fully offline testing without network or keys.
- OpenAILikeClient: Works with any OpenAI-compatible REST endpoint (OpenAI, OpenRouter,
  Ollama, Groq, vLLM, LM Studio) using the existing httpx dependency.
"""

from collections.abc import Callable
import json
import os
from typing import Any, Protocol, runtime_checkable

import httpx


class LLMConfigurationError(Exception):
    """Raised when an LLM provider is not configured or configured improperly."""


class LLMProviderError(Exception):
    """Raised when an LLM API call fails, times out, or returns an HTTP error."""


@runtime_checkable
class LLMClient(Protocol):
    """Protocol for model-agnostic LLM completion backends."""

    def complete(self, prompt: str, system_prompt: str | None = None) -> str:
        """Send a prompt to the model and return the generated text response."""
        ...


class MockLLMClient:
    """Deterministic, zero-network mock client for offline development and testing."""

    def __init__(
        self,
        response: str | dict[str, Any] | None = None,
        response_generator: Callable[[str, str | None], str] | None = None,
    ) -> None:
        """Initialize MockLLMClient.

        Args:
            response: Static string or dictionary to return as JSON string.
            response_generator: Optional callable receiving (prompt, system_prompt) and returning string.
        """
        if isinstance(response, dict):
            self._default_response = json.dumps(response, indent=2)
        else:
            self._default_response = response
        self._response_generator = response_generator
        self.call_history: list[tuple[str, str | None]] = []

    def complete(self, prompt: str, system_prompt: str | None = None) -> str:
        """Record the call and return mock output."""
        self.call_history.append((prompt, system_prompt))
        if self._response_generator is not None:
            return self._response_generator(prompt, system_prompt)
        if self._default_response is not None:
            return self._default_response

        # Dynamic mock generation if prompt contains evidence blocks
        import re
        chunk_ids = re.findall(r"=== EVIDENCE ITEM:\s*([^\s=]+)\s*===", prompt)
        if chunk_ids:
            mock_payload: dict[str, Any] = {
                "overview": f"Synthesized findings across {len(chunk_ids)} retrieved evidence item(s) addressing the research question.",
                "key_findings": [
                    {
                        "claim_id": "C1",
                        "text": "Reported findings in the indexed evidence support the primary experimental outcomes.",
                        "evidence_ids": [chunk_ids[0]],
                    }
                ],
                "conflicting_findings": [],
                "limitations": [
                    "Synthesis generated using mock provider for offline demonstration and testing."
                ],
            }
            if len(chunk_ids) > 1:
                mock_payload["key_findings"].append({
                    "claim_id": "C2",
                    "text": "Secondary study results corroborate the observed baseline characteristics.",
                    "evidence_ids": [chunk_ids[1]],
                })
            return json.dumps(mock_payload, indent=2)

        return "{}"


class OpenAILikeClient:
    """Client for any OpenAI-compatible /v1/chat/completions HTTP endpoint."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 60.0,
    ) -> None:
        """Initialize OpenAILikeClient.

        Args:
            model: Model identifier (e.g. 'gpt-4o-mini', 'llama3.1:8b').
            api_key: API authorization key (optional for local endpoints like Ollama).
            base_url: Base URL of the OpenAI-compatible API endpoint.
            timeout: HTTP timeout in seconds.
        """
        self.model = model
        self.api_key = api_key or "no-key"
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def complete(self, prompt: str, system_prompt: str | None = None) -> str:
        """Send chat completion request to the OpenAI-compatible endpoint."""
        endpoint = f"{self.base_url}/chat/completions"

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(endpoint, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                if content is None:
                    raise LLMProviderError("LLM response contained no content in choices[0].message.")
                return str(content)
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            error_body = exc.response.text[:300]
            raise LLMProviderError(f"LLM API returned HTTP {status}: {error_body}") from exc
        except httpx.RequestError as exc:
            raise LLMProviderError(f"Network error communicating with LLM API at {endpoint}: {exc}") from exc
        except (KeyError, IndexError, ValueError) as exc:
            raise LLMProviderError(f"Unexpected response schema from LLM provider: {exc}") from exc


def get_llm_client(
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    mock_response: str | dict[str, Any] | None = None,
) -> LLMClient:
    """Factory creating an LLM client based on arguments or environment variables.

    Supported providers:
        - 'mock': Offline testing, requires no keys or network.
        - 'openai': OpenAI API (requires LLM_API_KEY or OPENAI_API_KEY).
        - 'ollama': Local Ollama instance (default base URL: http://localhost:11434/v1).
        - 'openrouter': OpenRouter API (default base URL: https://openrouter.ai/api/v1).

    Raises:
        LLMConfigurationError: If no provider is configured or an unsupported provider is requested.
    """
    resolved_provider = (
        provider
        or os.environ.get("ATHENA_LLM_PROVIDER")
        or ""
    ).strip().lower()

    if not resolved_provider:
        raise LLMConfigurationError(
            "No LLM provider is configured in ATHENA.\n"
            "To enable evidence synthesis, specify a provider using the CLI flag or environment variables:\n"
            "  CLI:          --llm-provider mock (for offline testing)\n"
            "  Environment:  ATHENA_LLM_PROVIDER=mock\n"
            "                ATHENA_LLM_PROVIDER=ollama (for local free models)\n"
            "                ATHENA_LLM_PROVIDER=openai\n"
            "                ATHENA_LLM_PROVIDER=openrouter\n"
            "Optional keys/models:\n"
            "  ATHENA_LLM_MODEL=llama3.1:8b (or gpt-4o-mini)\n"
            "  LLM_API_KEY=your_key_here\n"
            "  LLM_BASE_URL=http://localhost:11434/v1"
        )

    supported_providers = {"mock", "openai", "ollama", "openrouter"}
    if resolved_provider not in supported_providers:
        raise LLMConfigurationError(
            f"Unsupported LLM provider: '{resolved_provider}'. "
            f"Supported providers are: {', '.join(sorted(supported_providers))}."
        )

    if resolved_provider == "mock":
        return MockLLMClient(response=mock_response)

    resolved_model = model or os.environ.get("ATHENA_LLM_MODEL")
    resolved_api_key = api_key or os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    resolved_base_url = base_url or os.environ.get("LLM_BASE_URL")

    if resolved_provider == "ollama":
        target_model = resolved_model or "llama3.1:8b"
        target_url = resolved_base_url or "http://localhost:11434/v1"
        target_key = resolved_api_key or "ollama"
        return OpenAILikeClient(model=target_model, api_key=target_key, base_url=target_url)

    if resolved_provider == "openrouter":
        if not resolved_api_key:
            raise LLMConfigurationError(
                "OpenRouter provider requires an API key. Set LLM_API_KEY environment variable or pass --api-key."
            )
        target_model = resolved_model or "anthropic/claude-3-haiku"
        target_url = resolved_base_url or "https://openrouter.ai/api/v1"
        return OpenAILikeClient(model=target_model, api_key=resolved_api_key, base_url=target_url)

    if resolved_provider == "openai":
        if not resolved_api_key:
            raise LLMConfigurationError(
                "OpenAI provider requires an API key. Set LLM_API_KEY or OPENAI_API_KEY environment variable."
            )
        target_model = resolved_model or "gpt-4o-mini"
        target_url = resolved_base_url or "https://api.openai.com/v1"
        return OpenAILikeClient(model=target_model, api_key=resolved_api_key, base_url=target_url)

    raise LLMConfigurationError(f"Unhandled provider: '{resolved_provider}'.")
