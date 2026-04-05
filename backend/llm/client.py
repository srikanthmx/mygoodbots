from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator
from typing import Any

import httpx

from backend.models import LLMError

# Defaults loaded from environment
_DEFAULT_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
_DEFAULT_OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
_DEFAULT_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
_DEFAULT_OLLAMA_MODELS = set(
    m for m in os.getenv("OLLAMA_MODELS", "llama3,mistral,mixtral").split(",") if m.strip()
)
_DEFAULT_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))
_DEFAULT_BACKOFF_SECONDS = float(os.getenv("LLM_BACKOFF_SECONDS", "1.0"))


class LLMClient:
    """Unified async LLM client routing to Ollama or OpenAI-compatible backends."""

    def __init__(
        self,
        base_url: str = _DEFAULT_BASE_URL,
        openai_base_url: str = _DEFAULT_OPENAI_BASE_URL,
        openai_api_key: str = _DEFAULT_OPENAI_API_KEY,
        ollama_models: set[str] | None = None,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        backoff_seconds: float = _DEFAULT_BACKOFF_SECONDS,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.openai_base_url = openai_base_url.rstrip("/")
        self.openai_api_key = openai_api_key
        self.ollama_models: set[str] = (
            ollama_models if ollama_models is not None else _DEFAULT_OLLAMA_MODELS
        )
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds

    def _resolve_provider(self, model: str) -> str:
        """Return 'ollama' if model is in ollama_models set, else 'openai'."""
        return "ollama" if model in self.ollama_models else "openai"

    async def _ollama_complete(
        self,
        model: str,
        messages: list[dict[str, Any]],
        stream: bool,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """POST to Ollama /api/chat and return the response content."""
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=120.0,
            )
            response.raise_for_status()
            data = response.json()
            return data["message"]["content"]

    async def _ollama_stream(
        self,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float,
        max_tokens: int,
    ) -> AsyncIterator[str]:
        """Stream from Ollama /api/chat yielding content chunks."""
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=120.0,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    content = chunk.get("message", {}).get("content", "")
                    if content:
                        yield content
                    if chunk.get("done", False):
                        break

    async def _openai_complete(
        self,
        model: str,
        messages: list[dict[str, Any]],
        stream: bool,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """POST to OpenAI-compatible /chat/completions and return the response content."""
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {"Authorization": f"Bearer {self.openai_api_key}"}
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.openai_base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=120.0,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]

    async def _openai_stream(
        self,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float,
        max_tokens: int,
    ) -> AsyncIterator[str]:
        """Stream from OpenAI-compatible /chat/completions yielding content chunks."""
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {"Authorization": f"Bearer {self.openai_api_key}"}
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"{self.openai_base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=120.0,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    raw = line[len("data: "):]
                    if raw.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        yield content

    async def complete(
        self,
        model: str,
        messages: list[dict[str, Any]],
        stream: bool = False,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str | AsyncIterator[str]:
        """
        Main completion method.

        Returns a full string when stream=False, or an AsyncIterator[str] when stream=True.
        Raises ValueError for invalid temperature.
        Raises LLMError after MAX_RETRIES exhausted.
        """
        if not (0.0 <= temperature <= 2.0):
            raise ValueError(
                f"temperature must be in [0.0, 2.0], got {temperature}"
            )

        provider = self._resolve_provider(model)

        if stream:
            # For streaming, return the async generator directly (no retry wrapper)
            if provider == "ollama":
                return self._ollama_stream(model, messages, temperature, max_tokens)
            else:
                return self._openai_stream(model, messages, temperature, max_tokens)

        # Non-streaming path with retry logic
        retry_count = 0
        last_exc: Exception | None = None

        while retry_count <= self.max_retries:
            try:
                if provider == "ollama":
                    return await self._ollama_complete(
                        model, messages, stream, temperature, max_tokens
                    )
                else:
                    return await self._openai_complete(
                        model, messages, stream, temperature, max_tokens
                    )
            except (httpx.HTTPStatusError,) as exc:
                # Only retry on 5xx server errors
                if exc.response.status_code >= 500:
                    last_exc = exc
                    retry_count += 1
                    if retry_count <= self.max_retries:
                        await asyncio.sleep(self.backoff_seconds * retry_count)
                else:
                    raise LLMError(str(exc)) from exc
            except httpx.TimeoutException as exc:
                last_exc = exc
                retry_count += 1
                if retry_count <= self.max_retries:
                    await asyncio.sleep(self.backoff_seconds * retry_count)

        raise LLMError(
            f"Max retries ({self.max_retries}) exceeded for model {model}"
        ) from last_exc
